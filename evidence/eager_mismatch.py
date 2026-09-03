#!/usr/bin/env python
"""Characterise the pre-existing `F.interpolate` dynamic-shape numerics mismatch.

The no-op digest found 4 cases where compiled != eager, identically with the
three PRs on and off -- so this is a bug in shipping PyTorch, reached through
plain `F.interpolate(..., size=...)` under `dynamic=True`, with no custom op and
no inference_mode.

Determine, for each: how many elements differ, whether the compiled result is a
shifted index (an off-by-one in the source coordinate) rather than arithmetic
noise, and which decomposition path is actually running.
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CASES = [
    ("nearest", (448, 448), (192, 192)),
    ("nearest", (384, 384), (363, 363)),
    ("nearest", (37, 41), (74, 82)),
    ("nearest-exact", (384, 384), (363, 363)),
    # controls that were clean in the digest
    ("nearest", (32, 32), (64, 64)),
    ("nearest", (32, 32), (100, 70)),
]

print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0) if DEV=='cuda' else 'cpu'}")
print()


def src_index(mode, i, o):
    """The source coordinate eager picks, per dim, as ATen computes it."""
    scale = float(i) / float(o)  # compute_scales_value, float32-ish in C++
    idx = []
    for d in range(o):
        if mode == "nearest-exact":
            idx.append(min(int((d + 0.5) * scale), i - 1))
        else:
            idx.append(min(int(d * scale), i - 1))
    return idx


for mode, in_hw, out_hw in CASES:
    torch._dynamo.reset()
    torch.manual_seed(0)
    # Use a rows-are-distinguishable input so a shifted index is detectable:
    # value == source row index, so a wrong pick is visible as an integer delta.
    h, w = in_hw
    x = (
        torch.arange(h, device=DEV, dtype=torch.float32)
        .view(1, 1, h, 1)
        .expand(1, 1, h, w)
        .contiguous()
    )
    torch._dynamo.mark_dynamic(x, 2)
    torch._dynamo.mark_dynamic(x, 3)
    f = lambda t, m=mode, o=out_hw: F.interpolate(t, size=o, mode=m)
    got = torch.compile(f, dynamic=True)(x)
    want = f(x)
    n = (got != want).sum().item()
    if n == 0:
        print(f"{mode:<14} {in_hw} -> {out_hw}: bit-exact")
        continue
    # rows only (dim 2): compare the source row each picked
    g_rows = got[0, 0, :, 0].to(torch.int64).tolist()
    w_rows = want[0, 0, :, 0].to(torch.int64).tolist()
    deltas = sorted({g - wv for g, wv in zip(g_rows, w_rows)})
    bad = [(d, wv, g) for d, (wv, g) in enumerate(zip(w_rows, g_rows)) if wv != g]
    print(f"{mode:<14} {in_hw} -> {out_hw}: {n} elems differ")
    print(f"    row-index deltas observed: {deltas}   ({len(bad)}/{out_hw[0]} rows wrong)")
    if bad:
        d, wv, g = bad[0]
        print(f"    first wrong output row {d}: eager picks src {wv}, compiled picks {g}")
        # show the float32 vs exact scale at that coordinate
        i, o = in_hw[0], out_hw[0]
        import struct

        s32 = struct.unpack("f", struct.pack("f", i / o))[0]
        exact = (d + 0.5) * (i / o) if mode == "nearest-exact" else d * (i / o)
        approx = (d + 0.5) * s32 if mode == "nearest-exact" else d * s32
        print(
            f"    scale {i}/{o}: fp64 {i/o!r}  fp32 {s32!r}"
            f"  floor(exact)={int(exact)} floor(fp32)={int(approx)}"
        )
