"""Is upsample_nearest2d_backward reachable with a symbolic input_size, and does it break?

It guards inp_h/inp_w (the grad_output spatial dims) but NOT out_h/out_w, which come from
`input_size`, then does `inp_h % out_h` and `ceildiv(inp_h, out_h)` on them.
"""
import os, traceback
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
DEV = os.environ.get("DEV","cuda")

# A. reachability via autograd through F.interpolate(mode="nearest")
hits = {"n":0}
import torch._inductor.lowering as L
orig = L.lowerings.get(torch.ops.aten.upsample_nearest2d_backward.default)
def spy(*a, **k):
    hits["n"] += 1
    return orig(*a, **k)
L.lowerings[torch.ops.aten.upsample_nearest2d_backward.default] = spy

def run(label, in_hw, out_hw, dynamic, mark=None):
    hits["n"]=0
    torch._dynamo.reset()
    x = torch.randn(1,3,*in_hw, device=DEV, requires_grad=True)
    if mark is not None:
        for d in mark: torch._dynamo.maybe_mark_dynamic(x, d)
    def f(a):
        return F.interpolate(a, size=out_hw, mode="nearest").sum()
    try:
        torch.compile(f, dynamic=dynamic)(x).backward()
        print(f"  {label}: OK, backward lowering hits={hits['n']}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[-1][:110]}  hits={hits['n']}")

print("A. autograd through interpolate(nearest)")
run("static 32->64", (32,32), (64,64), False)
run("dynamic 32->64", (32,32), (64,64), True)
run("dynamic+mark batch", (32,32), (64,64), True, mark=[0])
run("dynamic+mark H,W (input dims symbolic)", (32,32), (64,64), True, mark=[2,3])
run("dynamic non-divisible 32->63", (32,32), (63,63), True, mark=[2,3])

# B. call the backward op DIRECTLY with a symbolic input_size
print("B. direct aten.upsample_nearest2d_backward with symbolic input_size")
def g(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad, [grad.shape[-2], grad.shape[-1]], [ref.shape[-2], ref.shape[-1]])
for label, in_hw, out_hw in [("64->32 divisible",(64,64),(32,32)), ("63->32 NOT divisible",(63,63),(32,32))]:
    torch._dynamo.reset()
    grad = torch.randn(1,3,*in_hw, device=DEV)
    ref  = torch.randn(1,3,*out_hw, device=DEV)
    torch._dynamo.maybe_mark_dynamic(ref,2); torch._dynamo.maybe_mark_dynamic(ref,3)
    try:
        out = torch.compile(g, dynamic=True)(grad, ref)
        exp = g(grad, ref)
        print(f"  {label}: compiled, bit-exact={torch.equal(out,exp)}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[-1][:110]}")
