#!/usr/bin/env python3
"""Run the test suite with or without pytest installed.

    python tests/run_tests.py          # uses real pytest if importable
    python tests/run_tests.py --mini   # force the bundled shim

This exists because the repo's only hard dependency is torch. Prefer real
pytest when you have it; the shim keeps the suite runnable in a bare
torch-2.3.1 environment (which is exactly what README.md tells you to build).

Exit code 0 = all collected tests passed (skips are not failures).
"""

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEST_FILE = HERE / "test_bug_and_fix.py"


def load_with_shim():
    """Install the shim under the name `pytest`, then import the test module."""
    sys.path.insert(0, str(HERE))
    import _minipytest

    sys.modules["pytest"] = _minipytest

    spec = importlib.util.spec_from_file_location("test_bug_and_fix", TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, _minipytest


def expand(fn):
    """Yield (display_name, callable) for a test, expanding @parametrize."""
    param_sets = getattr(fn, "_params", [])
    if not param_sets:
        yield fn.__name__, fn
        return

    # Build the cartesian product of stacked @parametrize decorators.
    combos = [{}]
    for names, values in param_sets:
        new = []
        for base in combos:
            for v in values:
                vals = v if len(names) > 1 and isinstance(v, tuple) else (v,)
                merged = dict(base)
                merged.update(dict(zip(names, vals)))
                new.append(merged)
        combos = new

    for kw in combos:
        label = ",".join(f"{k}={v}" for k, v in kw.items())
        yield f"{fn.__name__}[{label}]", (lambda _f=fn, _k=kw: _f(**_k))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mini", action="store_true",
                    help="force the bundled shim even if pytest is installed")
    args = ap.parse_args()

    if not args.mini:
        try:
            import pytest  # noqa: F401
        except ImportError:
            pass
        else:
            print("pytest is installed -- delegating to it.\n")
            import pytest as _pt
            return _pt.main(["-v", str(TEST_FILE)])

    print("Running with the bundled minimal pytest shim "
          "(install pytest for the real thing).\n")

    sys.path.insert(0, str(ROOT))
    mod, shim = load_with_shim()
    import torch

    # The module-level `pytestmark` gate (torch version). Under the shim,
    # mark.skipif returns a decorator, so applying it to a probe function tells
    # us whether the whole module should be skipped.
    module_gate = getattr(mod, "pytestmark", None)
    if module_gate is not None:
        def _probe():
            pass

        for cond, reason in getattr(module_gate(_probe), "_skipifs", []):
            if cond:
                print(f"SKIPPING THE WHOLE MODULE: {reason}")
                return 0

    names = [n for n in dir(mod) if n.startswith("test_")]
    tests = [getattr(mod, n) for n in sorted(names)]
    tests = [t for t in tests if callable(t) and not getattr(t, "_is_fixture", False)]

    passed = failed = skipped = 0
    failures = []

    for fn in tests:
        for label, call in expand(fn):
            # Honour @skipif conditions.
            skip_reason = None
            for cond, reason in getattr(fn, "_skipifs", []):
                if cond:
                    skip_reason = reason
                    break
            if skip_reason:
                print(f"  SKIP  {label}\n          ({skip_reason})")
                skipped += 1
                continue

            torch._dynamo.reset()  # the autouse fixture, applied by hand
            try:
                call()
            except shim.Skipped as s:
                print(f"  SKIP  {label}\n          ({s.reason})")
                skipped += 1
            except Exception:
                print(f"  FAIL  {label}")
                failed += 1
                failures.append((label, traceback.format_exc()))
            else:
                print(f"  ok    {label}")
                passed += 1
            finally:
                torch._dynamo.reset()

    print("\n" + "=" * 70)
    print(f"passed={passed}  failed={failed}  skipped={skipped}")
    print("=" * 70)

    if failures:
        print("\nFailure detail:")
        for label, tb in failures:
            print(f"\n--- {label} " + "-" * max(0, 66 - len(label)))
            print(tb)

    if skipped:
        print("\nSkips are expected: tests for the opposite patch state are skipped.")
        print("Run the suite in BOTH states to cover everything:")
        print("    python patch/patch.py revert && python tests/run_tests.py")
        print("    python patch/patch.py apply  && python tests/run_tests.py")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
