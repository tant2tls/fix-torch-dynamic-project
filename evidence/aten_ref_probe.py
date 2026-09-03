#!/usr/bin/env python
"""Exact reference model of ATen's nearest index, then compare Triton's options.

ATen (UpSample.h):
    scale     = static_cast<float>(input_size) / output_size   // float32 divide
    src_index = min(int64(floorf(dst_index * scale)), input_size - 1)

`dst_index * scale` is int64 * float -> float (single precision) in C++.
Model that exactly with numpy float32, then ask which Triton divide reproduces
it: the approximate `/` that inductor emits today, or `div_rn`.
"""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def k(o_approx, o_rn, I, O, n, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    m = off < n
    s_approx = (I / O).to(tl.float32)  # what inductor emits for dynamic shapes
    s_rn = tl.math.div_rn(I.to(tl.float32), O.to(tl.float32))
    d = off.to(tl.float32)
    tl.store(o_approx + off, (d * s_approx).to(tl.int32), mask=m)
    tl.store(o_rn + off, (d * s_rn).to(tl.int32), mask=m)


def aten_ref(i, o):
    """float32 scale, float32 multiply, floor -- exactly UpSample.h."""
    scale = np.float32(np.float32(i) / np.float32(o))
    d = np.arange(o, dtype=np.float32)
    return np.minimum(np.floor(d * scale).astype(np.int64), i - 1)


RATIOS = [
    (448, 192),
    (384, 363),
    (37, 74),
    (32, 64),
    (32, 100),
    (63, 31),
    (100, 70),
    (1000, 999),
    (255, 256),
]

print(f"torch {torch.__version__}  triton {triton.__version__}")
print(f"gpu   {torch.cuda.get_device_name(0)}\n")
print(f"{'ratio':<12}{'inductor `/`':<18}{'div_rn':<18}")
print("-" * 48)
ta = tr = 0
for i, o in RATIOS:
    B = triton.next_power_of_2(o)
    a = torch.zeros(B, dtype=torch.int32, device="cuda")
    r = torch.zeros(B, dtype=torch.int32, device="cuda")
    k[(1,)](a, r, i, o, o, BLOCK=B)
    ref = aten_ref(i, o)
    ap = np.minimum(a[:o].cpu().numpy().astype(np.int64), i - 1)
    rn = np.minimum(r[:o].cpu().numpy().astype(np.int64), i - 1)
    da = int((ap != ref).sum())
    dr = int((rn != ref).sum())
    ta += da
    tr += dr
    print(f"{f'{i}->{o}':<12}{f'{da}/{o} differ':<18}{f'{dr}/{o} differ':<18}")
print("-" * 48)
print(f"{'TOTAL':<12}{f'{ta} differ':<18}{f'{tr} differ':<18}")
print()
print("Column 2 = today's behaviour vs eager. Column 3 = with a correctly")
print("rounded divide. Zero in column 3 means div_rn reproduces eager exactly.")
