#!/usr/bin/env python
"""No-op proof: real F.interpolate / nn.Upsample paths must be UNCHANGED.

Emits a JSON digest of results + generated Triton source hashes over a wide grid
of real ATen upsample calls. Run once with the PRs on and once off; the two
digests must be byte-identical. That is what makes "strict no-op for every
existing caller" a measurement rather than an assertion.

    python noop_digest.py out.json
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

MODES = ["nearest", "nearest-exact"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def h(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()[:16]


def seed():
    # Every tensor must be identical across the ON and OFF runs or the two
    # digests are not comparable.
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)


def code_hash(codes):
    return hashlib.sha256("\n".join(codes).encode()).hexdigest()[:16]


def fresh():
    torch._dynamo.reset()
    seed()


# Validate the output path BEFORE doing ~63 GPU compilations. This used to be
# read as sys.argv[1] at the very end, so forgetting the argument burned the
# entire run and then raised IndexError with nothing written.
if len(sys.argv) < 2:
    raise SystemExit(f"usage: {os.path.basename(__file__)} OUT.json")
path = sys.argv[1]

out = {"torch": torch.__version__, "device": DEV, "cases": {}}

# ---- forward: size= and scale_factor=, static and dynamic, 1d/2d/3d ----
for mode in MODES:
    for dyn in (False, True):
        for in_hw, out_hw in [
            ((32, 32), (64, 64)),
            ((32, 32), (100, 70)),
            ((448, 448), (192, 192)),
            ((384, 384), (363, 363)),
            ((37, 41), (74, 82)),
        ]:
            fresh()
            x = torch.randn(2, 3, *in_hw, device=DEV)
            if dyn:
                torch._dynamo.mark_dynamic(x, 2)
                torch._dynamo.mark_dynamic(x, 3)
            f = lambda t, m=mode, o=out_hw: F.interpolate(t, size=o, mode=m)
            got, codes = run_and_get_code(torch.compile(f, dynamic=dyn), x)
            k = f"fwd2d_size_{mode}_dyn{int(dyn)}_{in_hw}_{out_hw}"
            out["cases"][k] = [h(got), code_hash(codes), h(f(x))]

        for sf in (2.0, 1.5, 0.5, 3.0):
            fresh()
            x = torch.randn(2, 3, 32, 32, device=DEV)
            if dyn:
                torch._dynamo.mark_dynamic(x, 2)
                torch._dynamo.mark_dynamic(x, 3)
            f = lambda t, m=mode, s=sf: F.interpolate(t, scale_factor=s, mode=m)
            got, codes = run_and_get_code(torch.compile(f, dynamic=dyn), x)
            out["cases"][f"fwd2d_sf_{mode}_dyn{int(dyn)}_{sf}"] = [
                h(got),
                code_hash(codes),
                h(f(x)),
            ]

        fresh()
        x1 = torch.randn(2, 3, 40, device=DEV)
        f1 = lambda t, m=mode: F.interpolate(t, size=(90,), mode=m)
        got, codes = run_and_get_code(torch.compile(f1, dynamic=dyn), x1)
        out["cases"][f"fwd1d_{mode}_dyn{int(dyn)}"] = [
            h(got),
            code_hash(codes),
            h(f1(x1)),
        ]

        fresh()
        x3 = torch.randn(1, 2, 8, 10, 12, device=DEV)
        f3 = lambda t, m=mode: F.interpolate(t, size=(16, 15, 25), mode=m)
        got, codes = run_and_get_code(torch.compile(f3, dynamic=dyn), x3)
        out["cases"][f"fwd3d_{mode}_dyn{int(dyn)}"] = [
            h(got),
            code_hash(codes),
            h(f3(x3)),
        ]

        fresh()
        x = torch.randn(2, 3, 24, 24, device=DEV)
        m = torch.nn.Upsample(size=(48, 48), mode=mode)
        got, codes = run_and_get_code(torch.compile(m, dynamic=dyn), x)
        out["cases"][f"mod2d_{mode}_dyn{int(dyn)}"] = [
            h(got),
            code_hash(codes),
            h(m(x)),
        ]

# ---- backward through real autograd ----
for mode in MODES:
    for dyn in (False, True):
        for in_hw, out_hw in [
            ((32, 32), (64, 64)),
            ((33, 31), (99, 62)),
            ((64, 64), (32, 32)),
        ]:
            fresh()
            x = torch.randn(2, 3, *in_hw, device=DEV, requires_grad=True)
            f = lambda t, m=mode, o=out_hw: F.interpolate(t, size=o, mode=m).sum()
            torch.compile(f, dynamic=dyn)(x).backward()
            g_compiled = x.grad.clone()
            x.grad = None
            f(x).backward()
            out["cases"][f"bwd_{mode}_dyn{int(dyn)}_{in_hw}_{out_hw}"] = [
                h(g_compiled),
                "n/a",
                h(x.grad),
            ]

# ---- adaptive_avg_pool2d, which PR3's exactly-divisible path routes into ----
# dynamic=True is skipped: `_adaptive_avg_pool2d` raises "cannot determine truth
# value of Relational" there on stock torch too (verified identical with all
# three PRs reverted), so it is a separate pre-existing bug, not a regression.
for dyn in (False,):
    for hw, osz in [((64, 64), (32, 32)), ((63, 61), (7, 7)), ((100, 70), (50, 35))]:
        fresh()
        x = torch.randn(2, 3, *hw, device=DEV)
        f = lambda t, o=osz: F.adaptive_avg_pool2d(t, o)
        got, codes = run_and_get_code(torch.compile(f, dynamic=dyn), x)
        out["cases"][f"adaptavg_{hw}_{osz}_dyn{int(dyn)}"] = [
            h(got),
            code_hash(codes),
            h(f(x)),
        ]

with open(path, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(f"wrote {path}: {len(out['cases'])} cases")
bad = [k for k, v in out["cases"].items() if v[0] != v[2]]
print(f"compiled == eager for {len(out['cases']) - len(bad)}/{len(out['cases'])} cases")
for k in bad:
    print("   DIFFERS:", k)
