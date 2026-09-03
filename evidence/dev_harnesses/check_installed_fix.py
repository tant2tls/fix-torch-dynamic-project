#!/usr/bin/env python3
"""Validate the installed patch: symbolic output sizes compile bit-exactly.

Unlike `variants.py` (which re-implements the lowering to compare candidate
fixes), this drives the *real installed* `upsample_nearestnd`, so it measures
the patch as shipped.

    python upstream/dev/check_installed_fix.py --device cpu
    python upstream/dev/check_installed_fix.py --device cuda

Exit 0 if every case is bit-exact vs eager, 1 otherwise.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch._inductor.lowering import (  # noqa: E402
    lowerings,
    register_lowering,
    upsample_nearestnd,
)


# Ratios that are exact, anisotropic, downsampling, and -- critically --
# 448->192 / 384->363 / 256->82, where a scale one ulp off flips a floored index.
PAIRS = [
    ((32, 32), (64, 64)),
    ((32, 32), (100, 70)),
    ((32, 32), (17, 17)),
    ((448, 448), (192, 192)),
    ((384, 384), (363, 363)),
    ((256, 256), (82, 82)),
    ((1, 1), (9, 9)),
    ((13, 13), (40, 39)),
    ((10, 10), (100, 100)),
    ((3, 4), (97, 53)),
    ((7, 5), (23, 11)),
    ((64, 64), (21, 85)),
]


def patch_state() -> str:
    src = inspect.getsource(upsample_nearestnd)
    return "patched" if "div_rn" in src else "unpatched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable", file=sys.stderr)
        return 3

    print(f"torch {torch.__version__}  device {dev}  lowering: {patch_state()}")

    # --- A. the defect path: symbolic output size, reached directly -----------
    ok = wrong = crashed = 0
    details: list[str] = []
    for exact in (False, True):
        mode = "nearest-exact" if exact else "nearest"
        for in_hw, out_hw in PAIRS:
            with torch.library._scoped_library("chk_ups", "FRAGMENT") as lib:
                lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")

                def ref(x, h, w, mode=mode):
                    return F.interpolate(x, size=(int(h), int(w)), mode=mode)

                lib.impl(
                    "ups2d",
                    lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)),
                    "Meta",
                )
                lib.impl("ups2d", ref, "CPU")
                if torch.cuda.is_available():
                    lib.impl("ups2d", ref, "CUDA")

                register_lowering(torch.ops.chk_ups.ups2d)(
                    lambda x, h, w: upsample_nearestnd(
                        x, [h, w], (None, None), n=2, exact=exact
                    )
                )

                torch._dynamo.reset()
                x = torch.randn(1, 3, *in_hw, device=dev)
                ref_t = torch.randn(1, 3, *out_hw, device=dev)
                torch._dynamo.maybe_mark_dynamic(ref_t, 2)
                torch._dynamo.maybe_mark_dynamic(ref_t, 3)
                expect = ref(x, *out_hw)
                try:
                    got = torch.compile(
                        lambda a, s: torch.ops.chk_ups.ups2d(a, s[0], s[1]),
                        dynamic=True,
                    )(x, ref_t.shape[-2:])
                    if torch.equal(got, expect):
                        ok += 1
                    else:
                        wrong += 1
                        details.append(
                            f"WRONG {in_hw}->{out_hw} {mode}: "
                            f"{int((got != expect).sum())}/{expect.numel()}"
                        )
                except Exception as e:  # noqa: BLE001
                    crashed += 1
                    if len(details) < 4:
                        details.append(
                            f"CRASH {in_hw}->{out_hw} {mode}: {type(e).__name__}"
                        )
                for k in [k for k in list(lowerings) if "chk_ups" in str(k)]:
                    lowerings.pop(k, None)

    total = 2 * len(PAIRS)
    print(f"A. symbolic output size : {ok}/{total} bit-exact, {wrong} wrong, {crashed} crashed")
    for d in details[:5]:
        print(f"   ! {d}")

    # --- B. one graph for many sizes (the fix must not specialize) ------------
    calls = {"n": 0}
    real = upsample_nearestnd
    with torch.library._scoped_library("chk_ups2", "FRAGMENT") as lib:
        lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")

        def ref2(x, h, w):
            return F.interpolate(x, size=(int(h), int(w)), mode="nearest")

        lib.impl(
            "ups2d", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta"
        )
        lib.impl("ups2d", ref2, "CPU")
        if torch.cuda.is_available():
            lib.impl("ups2d", ref2, "CUDA")

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        register_lowering(torch.ops.chk_ups2.ups2d)(
            lambda x, h, w: counting(x, [h, w], (None, None), n=2)
        )
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda a, s: torch.ops.chk_ups2.ups2d(a, s[0], s[1]), dynamic=True
        )
        x = torch.randn(1, 3, 32, 32, device=dev)
        sizes = [(48 + 8 * i, 48 + 8 * i) for i in range(20)]
        b_ok = 0
        for out_hw in sizes:
            ref_t = torch.randn(1, 3, *out_hw, device=dev)
            torch._dynamo.maybe_mark_dynamic(ref_t, 2)
            torch._dynamo.maybe_mark_dynamic(ref_t, 3)
            try:
                if torch.equal(compiled(x, ref_t.shape[-2:]), ref2(x, *out_hw)):
                    b_ok += 1
            except Exception:  # noqa: BLE001
                pass
        for k in [k for k in list(lowerings) if "chk_ups2" in str(k)]:
            lowerings.pop(k, None)
    print(
        f"B. {len(sizes)} distinct output sizes: {calls['n']} graph(s), "
        f"{b_ok}/{len(sizes)} bit-exact"
    )

    # --- C. ATen paths unchanged --------------------------------------------
    c_ok = c_bad = 0
    for mode in ("nearest", "nearest-exact"):
        for out_hw, sf in [
            ((64, 64), None),
            (None, 2.0),
            (None, 1.5),
            (None, 0.5),
            ((100, 70), None),
            ((17, 17), None),
        ]:
            torch._dynamo.reset()
            x = torch.randn(1, 3, 32, 32, device=dev)
            f = (
                (lambda a: F.interpolate(a, size=out_hw, mode=mode))
                if sf is None
                else (lambda a: F.interpolate(a, scale_factor=sf, mode=mode))
            )
            if torch.equal(torch.compile(f)(x), f(x)):
                c_ok += 1
            else:
                c_bad += 1
    print(f"C. ATen interpolate paths: {c_ok} match, {c_bad} mismatch")

    good = (
        wrong == 0
        and crashed == 0
        and ok == total
        and calls["n"] == 1
        and b_ok == len(sizes)
        and c_bad == 0
    )
    print("VERDICT:", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
