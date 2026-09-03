"""Can ATEN/autograd reach upsample_nearest2d_backward's symbolic path?

If a real backward pass can produce a symbolic input_size, this is a LIVE
user-visible bug, not just hardening -- a much stronger PR.
"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
import torch._inductor.lowering as L
DEV="cuda"

hits={"n":0,"symbolic":0}
orig=L.lowerings[torch.ops.aten.upsample_nearest2d_backward.default]
def spy(x, output_size=None, input_size=None, scales_h=None, scales_w=None):
    import sympy
    hits["n"]+=1
    oh, ow = input_size[-2], input_size[-1]
    if any(isinstance(v, sympy.Expr) and not v.is_number for v in (oh,ow)):
        hits["symbolic"]+=1
    return orig(x, output_size, input_size, scales_h, scales_w)
L.lowerings[torch.ops.aten.upsample_nearest2d_backward.default]=spy

def trial(label, in_hw, size=None, sf=None, dynamic=True, mark=()):
    hits["n"]=0; hits["symbolic"]=0
    torch._dynamo.reset()
    x=torch.randn(2,3,*in_hw,device=DEV,requires_grad=True)
    for d in mark: torch._dynamo.maybe_mark_dynamic(x,d)
    def f(a):
        if sf is not None: return F.interpolate(a,scale_factor=sf,mode="nearest").sum()
        return F.interpolate(a,size=size,mode="nearest").sum()
    try:
        torch.compile(f,dynamic=dynamic)(x).backward()
        print(f"  {label}: ok  hits={hits['n']} symbolic={hits['symbolic']}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[0][:80]}  hits={hits['n']} symbolic={hits['symbolic']}")

print("autograd through F.interpolate(mode='nearest'), backward:")
trial("size=, no marks",            (32,32), size=(64,64))
trial("size=, mark H,W",            (32,32), size=(64,64), mark=(2,3))
trial("size=, mark batch",          (32,32), size=(64,64), mark=(0,))
trial("scale_factor=2, mark H,W",   (32,32), sf=2.0, mark=(2,3))
trial("scale_factor=2, mark all",   (32,32), sf=2.0, mark=(0,2,3))
trial("scale_factor=1.5, mark H,W", (32,32), sf=1.5, mark=(2,3))
trial("size non-divisible, mark HW",(32,32), size=(63,63), mark=(2,3))

# nn.Upsample inside a module, more realistic
print("nn.Upsample module, backward:")
class M(torch.nn.Module):
    def __init__(s, sf): super().__init__(); s.u=torch.nn.Upsample(scale_factor=sf, mode="nearest")
    def forward(s,x): return s.u(x).sum()
for sf in (2.0, 1.5):
    hits["n"]=0; hits["symbolic"]=0
    torch._dynamo.reset()
    x=torch.randn(2,3,32,32,device=DEV,requires_grad=True)
    torch._dynamo.maybe_mark_dynamic(x,2); torch._dynamo.maybe_mark_dynamic(x,3)
    try:
        torch.compile(M(sf),dynamic=True)(x).backward()
        print(f"  nn.Upsample sf={sf}: ok hits={hits['n']} symbolic={hits['symbolic']}")
    except Exception as e:
        print(f"  nn.Upsample sf={sf}: {type(e).__name__}: {str(e).splitlines()[0][:70]} hits={hits['n']} symbolic={hits['symbolic']}")
