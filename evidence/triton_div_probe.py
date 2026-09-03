#!/usr/bin/env python
"""Is Triton's `/` (div.full) what makes the dynamic path pick a different pixel?

Inductor's dynamic upsample decomposition emits `(ks0 / 74).to(tl.float32)` -- an
approximate divide evaluated on the GPU. Eager (ATen `compute_scales_value`) uses
a correctly-rounded float32 divide on the host. Compare all three directly:

    approx  : (I / O)                     <- what inductor emits today
    div_rn  : tl.math.div_rn(I, O)        <- correctly rounded
    eager   : float32(I/O) on the host    <- the reference
"""
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def k(out_approx, out_rn, I, O, n, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    m = off < n
    s_approx = (I / O).to(tl.float32)
    s_rn = tl.math.div_rn(I.to(tl.float32), O.to(tl.float32))
    d = off.to(tl.float32)
    tl.store(out_approx + off, (d * s_approx).to(tl.int32), mask=m)
    tl.store(out_rn + off, (d * s_rn).to(tl.int32), mask=m)


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


print(f"torch {torch.__version__}  triton {triton.__version__}")
print(f"gpu   {torch.cuda.get_device_name(0)}")
print()
print(f"{'ratio':<12}{'inductor `/` vs eager':<26}{'div_rn vs eager':<20}")
print("-" * 58)
tot_a = tot_r = 0
for i, o in [(448, 192), (384, 363), (37, 74), (32, 64), (32, 100), (63, 31), (100, 70)]:
    n = o
    B = triton.next_power_of_2(n)
    a = torch.zeros(B, dtype=torch.int32, device="cuda")
    r = torch.zeros(B, dtype=torch.int32, device="cuda")
    k[(1,)](a, r, i, o, n, BLOCK=B)
    eager = [min(int(d * f32(f32(i) / f32(o))), i - 1) for d in range(n)]
    ap = [min(v, i - 1) for v in a[:n].tolist()]
    rn = [min(v, i - 1) for v in r[:n].tolist()]
    da = sum(1 for x, y in zip(ap, eager) if x != y)
    dr = sum(1 for x, y in zip(rn, eager) if x != y)
    tot_a += da
    tot_r += dr
    print(f"{f'{i}->{o}':<12}{f'{da}/{n} differ':<26}{f'{dr}/{n} differ':<20}")
print("-" * 58)
print(f"{'TOTAL':<12}{f'{tot_a} differ':<26}{f'{tot_r} differ':<20}")
print()
print("If column 2 is non-zero and column 3 is zero, the approximate divide is")
print("the mechanism and div_rn is the fix.")
