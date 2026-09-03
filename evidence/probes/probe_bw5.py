import os, traceback
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch
DEV=os.environ.get("DEV","cuda")
def g(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad, [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]])

for dyn in (None, False, True):
    torch._dynamo.reset()
    grad=torch.randn(1,3,64,64,device=DEV); ref=torch.randn(1,3,32,32,device=DEV)
    kw = {} if dyn is None else {"dynamic":dyn}
    try:
        exp=g(grad,ref); out=torch.compile(g,**kw)(grad,ref)
        print(f"dynamic={dyn}: OK bit-exact={torch.equal(out,exp)}")
    except Exception as e:
        msg=str(e).splitlines()[0]
        print(f"dynamic={dyn}: {type(e).__name__}: {msg[:90]}")
        tb=traceback.format_exc()
        frames=[l.strip() for l in tb.splitlines() if "_inductor" in l and ", in " in l]
        for f in frames[-6:]:
            import re
            m=re.search(r"(_inductor/[\w/]+\.py)\", line (\d+), in (\S+)", f)
            print("     ", m.group(1)+":"+m.group(2)+" in "+m.group(3) if m else f[:120])
