#!/usr/bin/env python
"""Is the upsample_nearest2d_backward mismatch caused by PR3, or pre-existing?

Found while quantifying PR3: with a symbolic input_size, grad 128x128 -> input
16x16 and 8x8 disagree with eager (653/768 and 177/192 elements). 32x32 and 64x64
are bit-exact. This MUST be resolved before PR3 is sent -- if the guard makes the
lowering reachable and the lowering is wrong for large pooling windows, then PR3
converts a clean crash into a silent wrong answer, which is worse.

Discriminating runs:
  1. STATIC input_size (no PR3 involvement -- the guard is a no-op when the size is
     already concrete). If static also mismatches, the defect is in the pooling
     math, not in PR3.
  2. Sweep the ratio to find where it breaks.
  3. Compare against the exactly-divisible avg_pool2d path vs the general
     adaptive path, since the lowering branches on `inp_h % out_h == 0`.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fn(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad,
        [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]],
    )


def probe(grad_hw, in_hw, dynamic):
    torch._dynamo.reset()
    torch.manual_seed(0)
    grad = torch.randn(1, 3, *grad_hw, device=DEV)
    ref = torch.randn(1, 3, *in_hw, device=DEV)
    if dynamic:
        torch._dynamo.maybe_mark_dynamic(ref, 2)
        torch._dynamo.maybe_mark_dynamic(ref, 3)
    try:
        got = torch.compile(fn, dynamic=dynamic)(grad, ref)
    except Exception as e:
        return f"RAISED {type(e).__name__}: {str(e)[:44]}"
    want = fn(grad, ref)
    n = (got != want).sum().item()
    if n == 0:
        return "bit-exact"
    mx = (got - want).abs().max().item()
    return f"MISMATCH {n}/{want.numel()} max|d|={mx:.4g}"


print(f"torch {torch.__version__}  {DEV}")
print()
print("Question 1: does the STATIC path (PR3 not involved) also mismatch?")
print(f"{'grad':<12}{'input':<12}{'ratio':<9}{'divisible':<11}{'static':<32}{'dynamic'}")
print("-" * 104)
for grad_hw, in_hw in [
    ((128, 128), (64, 64)),
    ((128, 128), (32, 32)),
    ((128, 128), (16, 16)),
    ((128, 128), (8, 8)),
    ((128, 128), (4, 4)),
    ((64, 64), (32, 32)),
    ((64, 64), (16, 16)),
    ((64, 64), (8, 8)),
    ((96, 96), (32, 32)),
    ((100, 70), (50, 35)),
    ((63, 63), (32, 32)),
    ((127, 127), (16, 16)),
]:
    r = grad_hw[0] / in_hw[0]
    div = "yes" if grad_hw[0] % in_hw[0] == 0 else "no"
    s = probe(grad_hw, in_hw, False)
    d = probe(grad_hw, in_hw, True)
    print(
        f"{str(grad_hw):<12}{str(in_hw):<12}{r:<9.2f}{div:<11}{s:<32}{d}"
    )
print()
print("If `static` mismatches too, PR3 is not the cause -- the pooling math is")
print("wrong for that window size regardless of how the size was obtained, and")
print("PR3 merely makes it reachable. That still blocks PR3 as written.")
