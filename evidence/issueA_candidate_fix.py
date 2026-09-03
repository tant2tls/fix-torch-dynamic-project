#!/usr/bin/env python
"""Candidate fix for Issue A, tested in-process (never touches installed torch).

`_compute_upsample_nearest_indices` computes `scale = isize / osize` on SymInts,
which Inductor lowers to an approximate on-device divide, so the floored index can
differ from eager by one. Eager (`compute_scales_value`, ATen/native/UpSample.h)
does a correctly-rounded float32 divide on the host.

Two candidate fixes, both one line:

  A) build the scale from float32 tensors so the divide is a real fp32 op
     -> lowers to ops.truediv on tensors, still approximate in Triton
  B) compute the reciprocal-free form: input_indices = (out_idx * isize) // osize
     -> integer math, no float divide at all

Option B is exact by construction for exact=False. For exact=True the +0.5
offset makes it (2*out_idx + 1) * isize // (2 * osize).

Monkey-patches the decomposition in-process and re-runs the four failing cases.
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F
from torch._decomp import decompositions as D

DEV = "cuda"
CASES = [
    ("nearest", 448, 192),
    ("nearest", 384, 363),
    ("nearest", 37, 74),
    ("nearest-exact", 384, 363),
    ("nearest", 32, 64),
    ("nearest", 32, 100),
    ("nearest", 1000, 999),
    ("nearest-exact", 448, 192),
]

orig = D._compute_upsample_nearest_indices


def integer_indices(input, output_size, scales, exact=False):
    """Exact integer formulation -- no float divide, so nothing to round wrong."""
    indices = []
    n = len(output_size)
    for d in range(n):
        osize = output_size[d]
        isize = input.shape[-n + d]
        out_idx = torch.arange(osize, dtype=torch.int64, device=input.device)
        if scales[d] is not None and scales[d] > 0:
            # keep the existing float path when an explicit scale was given:
            # `1/scale` is a genuine float and not derived from sizes
            scale = isize / (isize * scales[d])
            offset = 0.5 if exact else 0.0
            inp = ((out_idx.to(torch.float32) + offset) * scale).to(torch.int64)
        elif exact:
            # floor((i + 0.5) * isize/osize) == (2*i + 1) * isize // (2*osize)
            inp = (2 * out_idx + 1) * isize // (2 * osize)
        else:
            # floor(i * isize/osize) == i * isize // osize
            inp = out_idx * isize // osize
        for _ in range(n - 1 - d):
            inp = inp.unsqueeze(-1)
        indices.append(inp)
    return indices


def rows(t):
    return t[0, 0, :, 0].to(torch.int64).tolist()


def check(label):
    bad = 0
    print(f"\n=== {label} ===")
    for mode, i, o in CASES:
        torch._dynamo.reset()
        x = (
            torch.arange(i, device=DEV, dtype=torch.float32)
            .view(1, 1, i, 1)
            .expand(1, 1, i, 4)
            .contiguous()
        )
        f = lambda t: F.interpolate(t, size=(o, 4), mode=mode)
        eager = rows(f(x))
        got = rows(torch.compile(f, dynamic=True)(x))
        wrong = [(d, e, g) for d, (e, g) in enumerate(zip(eager, got)) if e != g]
        bad += bool(wrong)
        status = "exact" if not wrong else f"{len(wrong)}/{o} WRONG {wrong[:1]}"
        print(f"  {mode:<14} {i}->{o:<6} {status}")
    return bad


print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
n_before = check("BEFORE (stock decomposition)")

D._compute_upsample_nearest_indices = integer_indices
# the decomposition closures call the module-level name, so also patch the ref
# used inside upsample_nearest{1,2,3}d
import torch._decomp.decompositions as _d

_d._compute_upsample_nearest_indices = integer_indices
n_after = check("AFTER (integer index formulation)")

print()
print(f"failing configs: before={n_before}  after={n_after}")
print("Note: the compiled path is what changes; eager is the reference and is")
print("untouched, so 'exact' means compiled == eager bit for bit.")
