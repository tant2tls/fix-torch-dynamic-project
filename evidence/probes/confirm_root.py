"""Confirm: kernel_maxes = ceildiv(inp, out) is symbolic when out_h/out_w are unguarded."""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, sympy
import torch._inductor.lowering as L
from torch._inductor.utils import ceildiv
DEV="cuda"

seen={}
real=L.lowerings[torch.ops.aten.upsample_nearest2d_backward.default]
import torch._inductor.lowering as LL
orig_fn = LL.upsample_nearest2d_backward
def spy(x, output_size=None, input_size=None, scales_h=None, scales_w=None):
    from torch._inductor.virtualized import V
    *_b, ih, iw = x.get_size()
    ih = V.graph.sizevars.guard_int(ih); iw = V.graph.sizevars.guard_int(iw)
    *_b2, oh, ow = input_size
    seen["inp"]=(ih,iw); seen["out"]=(oh,ow)
    seen["out_types"]=(type(oh).__name__, type(ow).__name__)
    seen["kernel_max_h"]=ceildiv(ih,oh)
    seen["kernel_max_h_type"]=type(ceildiv(ih,oh)).__name__
    seen["mod_ok"]=None
    try:
        seen["mod_ok"] = (ih % oh == 0)
    except Exception as e:
        seen["mod_ok"] = f"{type(e).__name__}"
    return orig_fn(x, output_size, input_size, scales_h, scales_w)
LL.upsample_nearest2d_backward = spy
L.lowerings[torch.ops.aten.upsample_nearest2d_backward.default] = spy

def g(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad, [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]])
grad=torch.randn(1,3,64,64,device=DEV); ref=torch.randn(1,3,32,32,device=DEV)
try: torch.compile(g,dynamic=True)(grad,ref)
except Exception as e: pass
for k,v in seen.items(): print(f"  {k} = {v}")
