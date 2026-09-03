"""Does guarding out_h/out_w fix upsample_nearest2d_backward? And is guarding CORRECT here?

Unlike the forward case, kernel_maxes feeds `range()` -- a Python loop bound baked into the
generated code. There is no way to keep that dynamic, so guarding is the ONLY option here.
That asymmetry is worth stating in the report.
"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, sympy
import torch._inductor.lowering as LL
from torch._inductor.virtualized import V, ops
from torch._inductor.ir import Pointwise
from torch._inductor.utils import ceildiv
from torch._inductor import ir
DEV="cuda"

# a copy of the lowering with out_h/out_w guarded
def fixed(x, output_size=None, input_size=None, scales_h=None, scales_w=None):
    from torch._inductor.lowering import (_adaptive_pooling_fn, pad_adaptive_loader,
                                          avg_pool2d)
    from torch.utils._sympy.functions import CeilDiv, FloorDiv
    x.realize_hint()
    *_batch, inp_h, inp_w = x.get_size()
    inp_h = V.graph.sizevars.guard_int(inp_h)
    inp_w = V.graph.sizevars.guard_int(inp_w)
    *_batch, out_h, out_w = input_size
    # THE FIX: guard the output sizes too -- kernel_maxes feeds range()
    out_h = V.graph.sizevars.guard_int(out_h)
    out_w = V.graph.sizevars.guard_int(out_w)
    if inp_h % out_h == 0 and inp_w % out_w == 0:
        return avg_pool2d(x, [FloorDiv(inp_h,out_h), FloorDiv(inp_w,out_w)], divisor_override=1)
    h_kernel_max = ceildiv(inp_h, out_h); w_kernel_max = ceildiv(inp_w, out_w)
    def start_index(index, out_dim, inp_dim):
        return CeilDiv(index*inp_dim, sympy.sympify(out_dim))
    def end_index(index, out_dim, inp_dim):
        return start_index((index+1), out_dim, inp_dim)
    fn_sum = _adaptive_pooling_fn(start_index=start_index, end_index=end_index,
        kernel_maxes=[h_kernel_max,w_kernel_max], in_sizes=[inp_h,inp_w],
        out_sizes=[out_h,out_w], pooling_fn=ops.add)
    def fn(idx): return fn_sum(idx, pad_adaptive_loader(x))
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn,
                            ranges=list(input_size))

def g(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad, [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]])

for label, use_fix in (("BASELINE", False), ("FIXED", True)):
    if use_fix:
        LL.lowerings[torch.ops.aten.upsample_nearest2d_backward.default] = fixed
    ok=bad=crash=0
    for g_hw, r_hw in [((64,64),(32,32)),((63,63),(32,32)),((96,96),(32,32)),
                       ((64,64),(21,21)),((100,70),(50,35)),((45,45),(15,15))]:
        torch._dynamo.reset()
        grad=torch.randn(1,3,*g_hw,device=DEV); ref=torch.randn(1,3,*r_hw,device=DEV)
        torch._dynamo.maybe_mark_dynamic(ref,2); torch._dynamo.maybe_mark_dynamic(ref,3)
        try:
            exp=g(grad,ref); out=torch.compile(g,dynamic=True)(grad,ref)
            if torch.equal(out,exp): ok+=1
            else: bad+=1
        except Exception as e:
            crash+=1
    print(f"  {label}: bit-exact {ok}/6, wrong {bad}, crashed {crash}")
