"""The real user-facing tradeoff: guard_int specializes (fast kernel, N recompiles, then EAGER)
vs deferred divide (one graph, slower kernel). Measure total wall time over a realistic
multi-resolution workload, including compile time."""
import os, time
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F, sympy
from torch._inductor.lowering import register_lowering, lowerings
from torch._inductor.ir import Pointwise
from torch._inductor.virtualized import V, ops
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_attrib.py")).read().split("def bench")[0].split('"""',2)[2])

import os as _o
SIZES=[(512,512)] if _o.environ.get("ONE") else [(256,256),(320,320),(384,384),(448,448),(512,512),(576,576),(640,640),(704,704),(768,768),(832,832),(896,896),(960,960)]

def run(variant, reps=5):
    tag=f"to{variant}"
    lib=torch.library._scoped_library(tag,"FRAGMENT"); l=lib.__enter__()
    l.define("u(Tensor x, SymInt h, SymInt w) -> Tensor")
    ref=lambda x,h,w: F.interpolate(x,size=(int(h),int(w)),mode="nearest")
    l.impl("u", lambda x,h,w: x.new_empty((x.shape[0],x.shape[1],h,w)),"Meta")
    l.impl("u", ref,"CUDA")
    op=getattr(torch.ops,tag).u
    calls={"n":0}
    base=make(variant)
    def counted(x,h,w):
        calls["n"]+=1
        return base(x,[h,w],(None,None),n=2)
    register_lowering(op)(counted)
    torch._dynamo.reset()
    x=torch.randn(4,16,128,128,device="cuda")
    f=torch.compile(lambda a,s: op(a,s[0],s[1]),dynamic=True)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(reps):
        for hw in SIZES:
            rt=torch.randn(1,1,*hw,device="cuda")
            torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
            f(x,rt.shape[-2:])
    torch.cuda.synchronize(); total=time.perf_counter()-t0
    lib.__exit__(None,None,None)
    for k in [k for k in list(lowerings) if tag in str(k)]: lowerings.pop(k,None)
    return total, calls["n"]

for v in ("guard_int","div_rn"):
    t,n = run(v)
    print(f"  {v:10s} total {t:7.2f} s over {len(SIZES)}x5 calls, {n} graphs compiled")
