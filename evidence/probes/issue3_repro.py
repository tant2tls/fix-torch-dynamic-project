import torch

def fn(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad,
        [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]],
    )

grad = torch.randn(1, 3, 64, 64)
ref = torch.randn(1, 3, 32, 32)
torch._dynamo.maybe_mark_dynamic(ref, 2)
torch._dynamo.maybe_mark_dynamic(ref, 3)
torch.compile(fn, dynamic=True)(grad, ref)
