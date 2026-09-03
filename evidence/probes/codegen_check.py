"""What does the generated kernel actually contain for the symbolic path?"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
from torch._inductor.lowering import register_lowering, upsample_nearestnd, lowerings
from torch._inductor.utils import run_and_get_code

with torch.library._scoped_library("cg","FRAGMENT") as lib:
    lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
    ref = lambda x,h,w: F.interpolate(x,size=(int(h),int(w)),mode="nearest")
    lib.impl("ups2d", lambda x,h,w: x.new_empty((x.shape[0],x.shape[1],h,w)), "Meta")
    lib.impl("ups2d", ref, "CPU")
    register_lowering(torch.ops.cg.ups2d)(lambda x,h,w: upsample_nearestnd(x,[h,w],(None,None),n=2))
    torch._dynamo.reset()
    x=torch.randn(1,3,32,32); rt=torch.randn(1,3,64,64)
    torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
    _, code = run_and_get_code(torch.compile(lambda a,s: torch.ops.cg.ups2d(a,s[0],s[1]), dynamic=True), x, rt.shape[-2:])
    blob="\n".join(code)
    # print the index math lines
    for line in blob.splitlines():
        if any(k in line for k in ("div_rn","/ ","tmp0","tmp1","ks")) and "def " not in line:
            print("   ", line.strip()[:130])
    print("\ncontains div_rn:", "div_rn" in blob)
    for k in [k for k in list(lowerings) if "cg" in str(k)]: lowerings.pop(k,None)
