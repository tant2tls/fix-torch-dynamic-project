#!/usr/bin/env python3
"""Compile the buggy operation and report what happened.

    python repro/repro.py              # auto-detect device
    python repro/repro.py --device cpu

Exit codes are meaningful so CI / a reader can script this:
    0 = compiled and produced numerically correct output (fix is installed)
    1 = compiled but the output was WRONG (should never happen; would be serious)
    2 = compilation failed with the expected TypeError (the bug is active)
    3 = compilation failed with some *other* error (environment problem)
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repro  # noqa: F401  -- sets TORCHINDUCTOR_CACHE_DIR before torch

import torch  # noqa: E402

from repro.bug import (  # noqa: E402
    compile_and_run,
    eager_reference,
    pick_device,
    version_warning,
)

RULE = "=" * 78

# The frames that make this traceback *this* bug rather than a generic failure.
# repro.py checks for them so a superficially similar error is not mistaken for
# a successful reproduction.
EXPECTED_FRAMES = [
    ("lowering.py", "scale_fn"),
    ("index_propagation.py", "propagate_sympy"),
    ("index_propagation.py", "constant"),
]


def patch_state() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "patch"))
    import patch as patch_mod

    return patch_mod.state(patch_mod.target_path())


def traceback_matches(exc: BaseException) -> bool:
    """Verify the failure went through the frames we claim it does."""
    frames = traceback.extract_tb(exc.__traceback__)
    seen = [(Path(f.filename).name, f.name) for f in frames]
    return all(f in seen for f in EXPECTED_FRAMES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--show-traceback", action="store_true",
                    help="print the full traceback on failure")
    args = ap.parse_args()

    warn = version_warning()
    if warn:
        print(warn + "\n")

    device = pick_device(args.device)
    state = patch_state()

    print(RULE)
    print("TorchInductor `ops.constant` bug -- reproduction")
    print(RULE)
    print(f"torch        : {torch.__version__}")
    print(f"device       : {device}")
    print(f"patch state  : {state}")
    print("operation    : F.interpolate(x, size=(ref.shape[-2], ref.shape[-1]),")
    print("               mode='nearest')  under torch.inference_mode(),")
    print("               with ref's H/W marked dynamic")
    print(RULE)
    print("\nCompiling ...\n")

    try:
        out = compile_and_run(device)
    except Exception as exc:  # noqa: BLE001 -- we classify it below
        is_typeerror = "Cannot convert expression to float" in str(exc)
        matched = traceback_matches(exc)

        if args.show_traceback:
            traceback.print_exc()
            print()

        if is_typeerror and matched:
            print("RESULT: compilation FAILED -- the bug reproduced.\n")
            print("  TypeError: Cannot convert expression to float\n")
            print("  Verified call chain (each frame confirmed present):")
            print("    lowering.py            upsample_nearestnd -> scale_fn")
            print("      ops.mul(x, ops.constant(scale, torch.float32))")
            print("    index_propagation.py   IndexPropagation.__getattr__.inner")
            print("      -> propagate_sympy -> SymPyOps.constant")
            print("      -> sympy.Float(float(<symbolic expr>))   *** raises ***\n")
            print("  `scale` here is a sympy expression, not a float, because the")
            print("  output size stayed symbolic. See README.md section 3.\n")
            print("  Next: `python patch/patch.py apply`, then re-run this script.")
            return 2

        print("RESULT: compilation failed, but NOT with the expected error.")
        print(f"        got {type(exc).__name__}: {str(exc)[:200]}")
        print(f"        expected-frames matched: {matched}")
        print("        This is an environment problem, not the bug. See README.md.")
        if not args.show_traceback:
            print("        Re-run with --show-traceback for detail.")
        return 3

    # Compiled. Now the part that matters: is the answer actually right?
    expected = eager_reference(device)
    if out.shape != expected.shape:
        print(f"RESULT: compiled, but SHAPE IS WRONG: {tuple(out.shape)} != "
              f"{tuple(expected.shape)}")
        return 1

    print("RESULT: compilation SUCCEEDED -- the fix is working.\n")
    print(f"  output shape : {tuple(out.shape)}  (expected {tuple(expected.shape)})")
    print("\n  Compiling is necessary but not sufficient -- verify numerics with:")
    print("      python repro/verify_fix.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
