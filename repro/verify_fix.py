#!/usr/bin/env python3
"""Verify the patched build is NUMERICALLY correct, not merely non-crashing.

A patch that makes compilation succeed but silently returns wrong pixels is
worse than the crash. This script compares the compiled output against eager
across a range of output sizes, including non-integer scale factors.

    python repro/verify_fix.py [--device cpu|cuda|auto]

Exit codes:
    0 = every case bit-exact against eager
    1 = a mismatch (serious -- report it)
    2 = compilation failed (the patch is not applied)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repro  # noqa: F401  -- sets TORCHINDUCTOR_CACHE_DIR before torch

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from repro.bug import pick_device, upsample_to_match, version_warning  # noqa: E402

# Deliberately mixed: exact 2x, anisotropic, non-integer ratios, a downsample,
# and an odd size. Non-integer ratios matter because that is where the symbolic
# index arithmetic and the float path could plausibly disagree.
CASES = [
    ((32, 32), (64, 64), "exact 2x upsample"),
    ((32, 32), (96, 48), "anisotropic, integer ratios"),
    ((32, 32), (100, 70), "non-integer ratios"),
    ((32, 32), (33, 33), "near-identity, non-integer"),
    ((32, 32), (17, 17), "downsample"),
    ((7, 13), (29, 31), "odd/prime sizes"),
    ((64, 64), (256, 256), "large 4x"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    warn = version_warning()
    if warn:
        print(warn + "\n")
    dev = pick_device(args.device)

    print("=" * 78)
    print(f"Numerical verification of the patched build "
          f"(torch {torch.__version__}, {dev})")
    print("=" * 78)

    # One compile, reused across every size -- this also exercises the dynamic
    # path properly: a single compiled artifact serving many output sizes is the
    # whole point of dynamic=True.
    torch._dynamo.reset()
    compiled = torch.compile(upsample_to_match)

    print(f"{'in -> out':<22} {'bit-exact':<11} {'max |diff|':<12} note")
    print("-" * 78)

    all_ok = True
    for in_hw, out_hw, note in CASES:
        x = torch.randn(1, 3, *in_hw, device=dev)
        ref = torch.randn(1, 3, *out_hw, device=dev)
        torch._dynamo.mark_dynamic(ref, 2)
        torch._dynamo.mark_dynamic(ref, 3)

        try:
            with torch.inference_mode():
                got = compiled(x, ref)
                expected = upsample_to_match(x, ref)
        except Exception as exc:  # noqa: BLE001
            if "Cannot convert expression to float" in str(exc):
                print("\nCompilation FAILED with the original bug.")
                print("The patch is not applied. Run: python patch/patch.py apply")
                return 2
            print(f"\nUnexpected {type(exc).__name__}: {str(exc)[:200]}")
            return 2

        if got.shape != expected.shape:
            print(f"{str(in_hw)+' -> '+str(out_hw):<22} {'SHAPE MISMATCH':<11} "
                  f"{'-':<12} got {tuple(got.shape)} want {tuple(expected.shape)}")
            all_ok = False
            continue

        exact = torch.equal(got, expected)
        maxdiff = (got.float() - expected.float()).abs().max().item()
        all_ok &= exact
        print(f"{str(in_hw)+' -> '+str(out_hw):<22} {str(exact):<11} "
              f"{maxdiff:<12.3g} {note}")

    print("-" * 78)
    if all_ok:
        print("\nPASS: every case is bit-exact against eager.")
        print("      The patch restores compilation without changing results.")
        print("\n      Note on semantics: the fallback re-routes the value through")
        print("      `index_expr`, so Inductor generates the symbolic index")
        print("      arithmetic it would have used anyway -- which is why the")
        print("      outputs match exactly rather than approximately.")
        return 0
    print("\nFAIL: at least one case did not match eager. Do not use this patch")
    print("      until the mismatch is understood.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
