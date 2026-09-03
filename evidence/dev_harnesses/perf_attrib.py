"""Attribute the 8x: is it div_rn, or is it dynamic-shape index math?

Add a 4th variant: 'truediv_dynamic' -- same dynamic shapes, but a plain (approximate)
fp32 divide instead of div_rn. If it is ~as slow as div_rn, the cost is the dynamic shape,
not the rounding mode.
"""
import os, time
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F, sympy
from torch._inductor.lowering import register_lowering, lowerings
from torch._inductor.ir import Pointwise
from torch._inductor.virtualized import V, ops

def make(variant):
    def low(x, output_size, scales_x, n=2):
        x.realize_hint(); x_loader=x.make_loader()
        i_sizes=x.get_size()[-n:]; batch=x.get_size()[:-n]
        i_sizes=[V.graph.sizevars.guard_int(i) for i in i_sizes]
        o_sizes=output_size
        inv=[]
        for i,o in zip(i_sizes,o_sizes):
            sym=isinstance(o,sympy.Expr) and not o.is_number
            if variant=="guard_int" and sym: inv.append(i/V.graph.sizevars.guard_int(o))
            elif variant in ("div_rn","truediv") and sym: inv.append(o)
            else: inv.append(i/o)
        def sfn(c,s,size):
            c=ops.index_expr(c,torch.float32)
            if isinstance(s,sympy.Expr):
                a=ops.constant(size,torch.float32); b=ops.index_expr(s,torch.float32)
                sv=ops.div_rn(a,b) if variant=="div_rn" else ops.truediv(a,b)
            else: sv=ops.constant(s,torch.float32)
            c=ops.mul(c,sv); c=ops.to_dtype(c,torch.int32)
            return ops.indirect_indexing(c,size,check=False)
        def fn(idx):
            cs=idx[-n:]; b=idx[:-n]
            return x_loader([*b,*[sfn(c,s,sz) for c,s,sz in zip(cs,inv,i_sizes)]])
        return Pointwise.create(device=x.get_device(),dtype=x.get_dtype(),inner_fn=fn,ranges=[*batch,*o_sizes])
    return low

def bench(variant, iters=200):
    tag=f"pa{variant}"
    lib=torch.library._scoped_library(tag,"FRAGMENT"); l=lib.__enter__()
    l.define("u(Tensor x, SymInt h, SymInt w) -> Tensor")
    ref=lambda x,h,w: F.interpolate(x,size=(int(h),int(w)),mode="nearest")
    l.impl("u", lambda x,h,w: x.new_empty((x.shape[0],x.shape[1],h,w)),"Meta")
    l.impl("u", ref,"CUDA")
    op=getattr(torch.ops,tag).u
    register_lowering(op)(lambda x,h,w: make(variant)(x,[h,w],(None,None),n=2))
    torch._dynamo.reset()
    x=torch.randn(8,32,512,512,device="cuda"); rt=torch.randn(1,1,2048,2048,device="cuda")
    torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
    f=torch.compile(lambda a,s: op(a,s[0],s[1]),dynamic=True)
    for _ in range(20): f(x,rt.shape[-2:])
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): f(x,rt.shape[-2:])
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/iters*1e3
    lib.__exit__(None,None,None)
    for k in [k for k in list(lowerings) if tag in str(k)]: lowerings.pop(k,None)
    return dt

# also: eager, and static-shape compiled, for reference
x=torch.randn(8,32,512,512,device="cuda")
for _ in range(10): F.interpolate(x,size=(2048,2048),mode="nearest")
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(200): F.interpolate(x,size=(2048,2048),mode="nearest")
torch.cuda.synchronize(); eager=(time.perf_counter()-t0)/200*1e3

print(f"  eager F.interpolate      {eager:7.3f} ms")
for v in ("guard_int","div_rn","truediv"):
    print(f"  {v:22s} {bench(v):7.3f} ms")
