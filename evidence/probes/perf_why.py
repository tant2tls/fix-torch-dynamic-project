"""Why is div_rn 8x slower? Look at the generated kernels side by side."""
import os, re
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F, sympy
from torch._inductor.lowering import register_lowering, lowerings
from torch._inductor.ir import Pointwise
from torch._inductor.virtualized import V, ops
from torch._inductor.utils import run_and_get_code
# Reuse the setup preamble (lowering variants) from perf.py, which sits beside this
# file. Resolved relative to __file__ -- an absolute path here used to point outside
# the repo, at the superseded prwork/ tree, and broke as soon as this moved.
_perf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf.py")
exec(open(_perf).read().split("def bench")[0].split('"""', 2)[2])

def code_for(variant):
    lib=torch.library._scoped_library(f"cw{variant}","FRAGMENT"); l=lib.__enter__()
    l.define("u(Tensor x, SymInt h, SymInt w) -> Tensor")
    ref=lambda x,h,w: F.interpolate(x,size=(int(h),int(w)),mode="nearest")
    l.impl("u", lambda x,h,w: x.new_empty((x.shape[0],x.shape[1],h,w)),"Meta")
    l.impl("u", ref,"CUDA")
    op=getattr(torch.ops,f"cw{variant}").u
    register_lowering(op)(lambda x,h,w: make(variant)(x,[h,w],(None,None),n=2))
    torch._dynamo.reset()
    x=torch.randn(8,32,512,512,device="cuda"); rt=torch.randn(1,1,2048,2048,device="cuda")
    torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
    _,code=run_and_get_code(torch.compile(lambda a,s: op(a,s[0],s[1]),dynamic=True), x, rt.shape[-2:])
    lib.__exit__(None,None,None)
    for k in [k for k in list(lowerings) if f"cw{variant}" in str(k)]: lowerings.pop(k,None)
    return "\n".join(code)

for v in ("guard_int","div_rn"):
    blob=code_for(v)
    print(f"===== {v} =====")
    for line in blob.splitlines():
        s=line.strip()
        if any(k in s for k in ("def triton_","xnumel","XBLOCK","tl.load","div_rn","tmp","x0 =","x1 =","x2 =","num_warps","grid")) and len(s)<160:
            print("   ",s[:150])
    print()
