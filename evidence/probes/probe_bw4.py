import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
os.environ["TORCHDYNAMO_VERBOSE"]="1"
import torch
DEV=os.environ.get("DEV","cuda")
def g(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad, [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]])
grad=torch.randn(1,3,64,64,device=DEV); ref=torch.randn(1,3,32,32,device=DEV)
torch.compile(g)(grad,ref)   # let it blow up uncaught
