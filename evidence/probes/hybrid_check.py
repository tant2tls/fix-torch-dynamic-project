"""Would a HYBRID be better: keep today's folded constant when the size is already
concrete (done), AND let the user's existing guard behaviour apply when the caller has
already specialized? i.e. is there any case where the deferred path fires but a guard
would have been free?

Key question for the PR: does the deferred path ever fire when o_sizes is ALREADY concrete?
If not, the fix only affects cases that today CRASH -- so there is no regression for anyone,
and the perf tradeoff applies only to newly-working code.
"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F, sympy
import torch._inductor.lowering as LL
DEV="cuda"

log={"symbolic":0,"concrete":0}
real=LL.upsample_nearestnd
def spy(x, output_size, scales_x, n=2, exact=False):
    for o in output_size:
        if isinstance(o,sympy.Expr) and not o.is_number: log["symbolic"]+=1
        else: log["concrete"]+=1
    return real(x, output_size, scales_x, n=n, exact=exact)
LL.upsample_nearestnd=spy
for k,v in list(LL.lowerings.items()):
    if "upsample_nearest" in str(k):
        pass  # aten entries call the module-level name indirectly via closures; patch below

# rebuild the aten registrations to point at the spy
import torch._inductor.lowering as L
L.lowerings[torch.ops.aten.upsample_nearest2d.default] = (
    lambda x, output_size, scales_h=None, scales_w=None: spy(x, output_size, (scales_h,scales_w), n=2))

print("ATen interpolate paths -- is output_size symbolic or concrete at the lowering?")
for label, kw, mark in [
    ("size=, static",            dict(size=(64,64)),      ()),
    ("size=, dynamic",           dict(size=(64,64)),      (2,3)),
    ("scale_factor=2, dynamic",  dict(scale_factor=2.0),  (2,3)),
    ("scale_factor=1.5, dynamic",dict(scale_factor=1.5),  (2,3)),
]:
    log["symbolic"]=0; log["concrete"]=0
    torch._dynamo.reset()
    x=torch.randn(1,3,32,32,device=DEV)
    for d in mark: torch._dynamo.maybe_mark_dynamic(x,d)
    try:
        torch.compile(lambda a: F.interpolate(a,mode="nearest",**kw),dynamic=True)(x)
    except Exception as e:
        pass
    print(f"  {label}: symbolic={log['symbolic']} concrete={log['concrete']}")
