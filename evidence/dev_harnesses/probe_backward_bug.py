import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch
import torch._inductor.lowering as L
DEV=os.environ.get("DEV","cuda")

hits={"n":0}
orig=L.lowerings[torch.ops.aten.upsample_nearest2d_backward.default]
def spy(*a,**k):
    hits["n"]+=1
    return orig(*a,**k)
L.lowerings[torch.ops.aten.upsample_nearest2d_backward.default]=spy

# full 4-element input_size, output_size = grad's spatial dims
def g(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad,
        [grad.shape[-2], grad.shape[-1]],                      # output_size (of fwd out)
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]],  # input_size (4 elems)
    )

for label, g_hw, r_hw, mark in [
    ("64->32 divisible, static",       (64,64),(32,32), None),
    ("64->32 divisible, ref H,W dyn",  (64,64),(32,32), (2,3)),
    ("63->32 non-divisible, static",   (63,63),(32,32), None),
    ("63->32 non-divisible, ref dyn",  (63,63),(32,32), (2,3)),
    ("96->32 divisible, ref dyn",      (96,96),(32,32), (2,3)),
]:
    hits["n"]=0
    torch._dynamo.reset()
    grad=torch.randn(1,3,*g_hw,device=DEV)
    ref =torch.randn(1,3,*r_hw,device=DEV)
    if mark:
        for d in mark: torch._dynamo.maybe_mark_dynamic(ref,d)
    try:
        exp=g(grad,ref)
        out=torch.compile(g,dynamic=True)(grad,ref)
        print(f"  {label}: hits={hits['n']} bit-exact={torch.equal(out,exp)}")
    except Exception as e:
        print(f"  {label}: hits={hits['n']} {type(e).__name__}: {chr(10).join(str(e).splitlines()[:6])}")
