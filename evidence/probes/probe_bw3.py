import os, traceback
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch
DEV=os.environ.get("DEV","cuda")
def g(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad, [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]])
grad=torch.randn(1,3,64,64,device=DEV); ref=torch.randn(1,3,32,32,device=DEV)
try:
    torch.compile(g)(grad,ref)
except Exception:
    tb = traceback.format_exc()
    # print only frames inside _inductor
    for line in tb.splitlines():
        if "_inductor" in line or "TypeError" in line or "FloorDiv" in line:
            print(line.strip()[:160])
