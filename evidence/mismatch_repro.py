#!/usr/bin/env python
"""Tight, dependency-free repro of the dynamic-shape index mismatch.

Claim under test: on stock torch, `F.interpolate(x, size=..., mode="nearest")`
compiled with dynamic shapes selects a different SOURCE PIXEL than eager for some
(input, output) size ratios -- an off-by-one in the index, not a tolerance issue.

Guards against the ways this could be an artifact of the harness:
  * no custom op, no inference_mode, no mark_dynamic games -- just torch.compile
  * input is `arange` along the sampled dim, so each output value IS the source
    index; a mismatch is an exact integer, not float noise
  * checks static compile too (must be bit-exact) to show it is dynamic-specific
  * checks CPU and CUDA separately
  * re-runs eager after compile to rule out input mutation
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

CASES = [
    ("nearest", 448, 192),
    ("nearest", 384, 363),
    ("nearest", 37, 74),
    ("nearest-exact", 384, 363),
    ("nearest", 32, 64),
    ("nearest", 32, 100),
]


def rows(t):
    return t[0, 0, :, 0].to(torch.int64).tolist()


def probe(device, mode, i, o, dynamic):
    torch._dynamo.reset()
    # value == source row index
    x = (
        torch.arange(i, device=device, dtype=torch.float32)
        .view(1, 1, i, 1)
        .expand(1, 1, i, 4)
        .contiguous()
    )
    f = lambda t: F.interpolate(t, size=(o, 4), mode=mode)
    eager_before = rows(f(x))
    got = rows(torch.compile(f, dynamic=dynamic)(x))
    eager_after = rows(f(x))
    assert eager_before == eager_after, "eager not reproducible -- harness bug"
    wrong = [(d, e, g) for d, (e, g) in enumerate(zip(eager_before, got)) if e != g]
    return wrong, len(eager_before)


print(f"torch {torch.__version__}")
devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
if torch.cuda.is_available():
    print(f"gpu   {torch.cuda.get_device_name(0)}")
print()
hdr = f"{'device':<7}{'mode':<15}{'in->out':<12}{'static':<12}{'dynamic':<12}"
print(hdr)
print("-" * len(hdr))
for device in devs:
    for mode, i, o in CASES:
        w_s, n = probe(device, mode, i, o, False)
        w_d, _ = probe(device, mode, i, o, True)
        s = "exact" if not w_s else f"{len(w_s)}/{n} WRONG"
        d = "exact" if not w_d else f"{len(w_d)}/{n} WRONG"
        print(f"{device:<7}{mode:<15}{f'{i}->{o}':<12}{s:<12}{d:<12}")
        if w_d:
            d0, e0, g0 = w_d[0]
            print(f"       first: out row {d0}: eager src {e0}, compiled src {g0}")
