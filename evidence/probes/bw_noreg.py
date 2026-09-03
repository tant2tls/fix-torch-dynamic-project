"""No-regression: real autograd backward through interpolate must be unchanged."""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
DEV="cuda"
ok=bad=0
for mode in ("nearest","nearest-exact"):
    for kw in (dict(size=(64,64)), dict(scale_factor=2.0), dict(scale_factor=1.5),
               dict(size=(63,63)), dict(size=(17,17))):
        for dyn in (False, True):
            torch._dynamo.reset()
            x=torch.randn(2,3,32,32,device=DEV,requires_grad=True)
            xe=x.detach().clone().requires_grad_(True)
            if dyn:
                torch._dynamo.maybe_mark_dynamic(x,2); torch._dynamo.maybe_mark_dynamic(x,3)
            f=lambda a: F.interpolate(a,mode=mode,**kw).sum()
            try:
                torch.compile(f,dynamic=dyn)(x).backward()
                f(xe).backward()
                if torch.equal(x.grad, xe.grad): ok+=1
                else: bad+=1; print(f"  GRAD MISMATCH {mode} {kw} dyn={dyn}")
            except Exception as e:
                bad+=1; print(f"  ERR {mode} {kw} dyn={dyn}: {type(e).__name__}: {str(e).splitlines()[0][:60]}")
print(f"autograd backward no-regression: {ok} ok, {bad} bad, of {ok+bad}")
# and the direct 3d/1d backward ops still fall back fine
for op,shape,osz,isz in [
  (torch.ops.aten.upsample_nearest2d_backward.default,(1,3,64,64),[64,64],[1,3,32,32]),
]:
    torch._dynamo.reset()
    g=torch.randn(*shape,device=DEV)
    try:
        e=op(g,osz,isz); c=torch.compile(lambda a: op(a,osz,isz))(g)
        print("direct backward op static:", torch.equal(e,c))
    except Exception as ex: print("direct backward op:", type(ex).__name__)
