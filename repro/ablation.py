#!/usr/bin/env python3
"""Show *why* a naive standalone reproduction does not fail.

The bug needs three ingredients at once. This script removes them one at a time
and reports which variants fail -- turning "it works on my machine" into a
localized claim about which code path is required.

    python repro/ablation.py [--device cpu|cuda|auto]

Every row is compiled from scratch (`torch._dynamo.reset()`), so no row can
benefit from another's cache.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repro  # noqa: F401  -- sets TORCHINDUCTOR_CACHE_DIR before torch

import contextlib  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from repro.bug import pick_device, version_warning  # noqa: E402

MARKER = "Cannot convert expression to float"


def attempt(fn, mk_inputs, ctx_factory) -> str:
    """Compile+run one variant; return 'BUG', 'ok', or another error's name."""
    torch._dynamo.reset()
    compiled = torch.compile(fn)
    try:
        with ctx_factory():
            compiled(*mk_inputs())
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return "BUG" if MARKER in str(exc) else type(exc).__name__


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    warn = version_warning()
    if warn:
        print(warn + "\n")
    dev = pick_device(args.device)

    def dyn_2d():
        """ref's H/W marked dynamic -> symbolic output size."""
        x = torch.randn(1, 3, 32, 32, device=dev)
        ref = torch.randn(1, 3, 64, 64, device=dev)
        torch._dynamo.mark_dynamic(ref, 2)
        torch._dynamo.mark_dynamic(ref, 3)
        return x, ref

    def static_2d():
        """No mark_dynamic -> output size is a concrete int."""
        return (torch.randn(1, 3, 32, 32, device=dev),
                torch.randn(1, 3, 64, 64, device=dev))

    def dyn_1d():
        x = torch.randn(1, 3, 32, device=dev)
        ref = torch.randn(1, 3, 64, device=dev)
        torch._dynamo.mark_dynamic(ref, 2)
        return x, ref

    def dyn_3d():
        x = torch.randn(1, 3, 4, 8, 8, device=dev)
        ref = torch.randn(1, 3, 8, 16, 16, device=dev)
        for d in (2, 3, 4):
            torch._dynamo.mark_dynamic(ref, d)
        return x, ref

    def size_from_ref(mode, n=2):
        if n == 1:
            return lambda x, r: F.interpolate(x, size=(r.shape[-1],), mode=mode)
        if n == 3:
            return lambda x, r: F.interpolate(
                x, size=(r.shape[-3], r.shape[-2], r.shape[-1]), mode=mode)
        return lambda x, r: F.interpolate(
            x, size=(r.shape[-2], r.shape[-1]), mode=mode)

    inference = torch.inference_mode
    no_grad = torch.no_grad
    grad_on = contextlib.nullcontext

    # (label, fn, inputs, context, which ingredient is being removed)
    cases = [
        ("all three ingredients present",
         size_from_ref("nearest"), dyn_2d, inference, "-- baseline, must fail"),

        ("(1) mode='bilinear' instead of 'nearest'",
         size_from_ref("bilinear"), dyn_2d, inference, "different lowering"),

        ("(2) scale_factor=2.0 instead of size=",
         lambda x, r: F.interpolate(x, scale_factor=2.0, mode="nearest"),
         dyn_2d, inference, "hits the 1.0/scale float path"),

        ("(2) size= but shapes NOT marked dynamic",
         size_from_ref("nearest"), static_2d, inference, "output size is concrete"),

        ("(3) torch.no_grad() instead of inference_mode",
         size_from_ref("nearest"), dyn_2d, no_grad, "decomposition still fires"),

        ("(3) grad enabled (no context at all)",
         size_from_ref("nearest"), dyn_2d, grad_on, "decomposition still fires"),

        ("variant: mode='nearest-exact'",
         size_from_ref("nearest-exact"), dyn_2d, inference, "same lowering, exact=True"),

        ("variant: 1-D nearest",
         size_from_ref("nearest", 1), dyn_1d, inference, "upsample_nearest1d"),

        ("variant: 3-D nearest",
         size_from_ref("nearest", 3), dyn_3d, inference, "upsample_nearest3d"),
    ]

    print("=" * 78)
    print(f"Ingredient ablation  (torch {torch.__version__}, device={dev})")
    print("=" * 78)
    print(f"{'variant':<44} {'result':<8} note")
    print("-" * 78)

    results = {}
    for label, fn, mk, ctx, note in cases:
        verdict = attempt(fn, mk, ctx)
        results[label] = verdict
        flag = "FAIL" if verdict == "BUG" else ("pass" if verdict == "ok" else verdict)
        print(f"{label:<44} {flag:<8} {note}")

    print("-" * 78)
    print("\nReading of this table:")
    print("  * All three ingredients are REQUIRED. Remove any one -> compiles fine.")
    print("  * The bug is not about 'nearest' per se; it is about a symbolic value")
    print("    reaching ops.constant. bilinear has a lowering that never does that.")
    print("  * inference_mode vs no_grad is the subtle one: only inference_mode")
    print("    bypasses the DispatchKey.Autograd decomposition. Run")
    print("    `python repro/why_it_survives.py` to see the two graphs side by side.")
    print("  * All of nearest 1d/2d/3d and nearest-exact are affected -- the defect")
    print("    is in the shared `upsample_nearestnd` helper, not one entry point.")

    baseline = results["all three ingredients present"]
    if baseline != "BUG":
        print(f"\nNOTE: the baseline did not fail (got '{baseline}'). Either the fix is")
        print("      already applied (`python patch/patch.py status`) or this is not")
        print("      torch 2.3.1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
