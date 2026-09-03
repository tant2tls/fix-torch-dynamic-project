#!/usr/bin/env python3
"""Apply, revert, or report the status of the one-hunk fix to TorchInductor.

The patch target is `torch/_inductor/index_propagation.py` inside whichever
torch the *active* interpreter imports -- it is located via `torch.__file__`,
never a hardcoded path.

    python patch/patch.py status     # which state is the installed torch in?
    python patch/patch.py apply      # install the fix (backs up first)
    python patch/patch.py revert     # restore the pristine file

`apply` is idempotent and always writes a `.orig` backup before its first edit,
so `revert` can restore byte-for-byte even if this repo is deleted.
"""

import argparse
import shutil
import sys
from pathlib import Path

# The exact pristine text of the code we replace. Matching on this full block
# (rather than a line number) means the script refuses to silently mangle a
# file that isn't the torch 2.3.1 we expect.
PRISTINE = """            if not all(v.is_symbolic for v in var_arguments):
                return self.fallback(name, args, kwargs)

            return self.propagate_sympy(name, args, kwargs)
"""

PATCHED = """            if not all(v.is_symbolic for v in var_arguments):
                return self.fallback(name, args, kwargs)

            try:
                return self.propagate_sympy(name, args, kwargs)
            except Exception:
                # A value that cannot be lowered as a float constant (e.g. a
                # symbolic output size arriving from the upsample_nearestnd
                # lowering) is treated as an index expression instead of
                # aborting the whole compilation. Inductor then emits the
                # symbolic index arithmetic it would have used anyway.
                return self.propagate_sympy("index_expr", args, kwargs)
"""


def target_path() -> Path:
    """Locate index_propagation.py in the torch the active interpreter imports."""
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: torch is not importable in this interpreter. "
                 "Activate the right venv first (see README.md).")
    p = Path(torch.__file__).parent / "_inductor" / "index_propagation.py"
    if not p.is_file():
        sys.exit(f"ERROR: expected file not found: {p}\n"
                 "       torch >= 2.4 restructured this module; the patch does not apply.")
    return p


def state(path: Path) -> str:
    """Return 'pristine', 'patched', or 'unknown' for the installed file."""
    text = path.read_text()
    if PATCHED in text:
        return "patched"
    if PRISTINE in text:
        return "pristine"
    return "unknown"


def describe(path: Path) -> str:
    import torch
    st = state(path)
    label = {
        "pristine": "PRISTINE  -> the bug is ACTIVE (compile will fail)",
        "patched": "PATCHED   -> the fix is INSTALLED (compile will succeed)",
        "unknown": "UNKNOWN   -> file matches neither expected form; inspect manually",
    }[st]
    backup = path.with_suffix(path.suffix + ".orig")
    return (
        f"torch version : {torch.__version__}\n"
        f"target file   : {path}\n"
        f"backup        : {backup if backup.exists() else '(none yet)'}\n"
        f"state         : {label}"
    )


def apply(path: Path) -> int:
    st = state(path)
    if st == "patched":
        print("Already patched -- nothing to do.")
        return 0
    if st == "unknown":
        print("ERROR: the target file matches neither the pristine nor the patched\n"
              "       form. Refusing to edit it. Restore a clean torch 2.3.1 first.",
              file=sys.stderr)
        return 1

    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backed up pristine file -> {backup}")

    path.write_text(path.read_text().replace(PRISTINE, PATCHED, 1))
    print(f"Applied fix to {path}")
    print("The next torch.compile will pick this up (Inductor's code cache keys on\n"
          "the graph, not on its own source -- run repro/repro.py to confirm).")
    return 0


def revert(path: Path) -> int:
    backup = path.with_suffix(path.suffix + ".orig")
    if backup.exists():
        shutil.copy2(backup, path)
        print(f"Restored {path}\n     from {backup}")
        return 0
    if state(path) == "patched":
        path.write_text(path.read_text().replace(PATCHED, PRISTINE, 1))
        print(f"Reverted the patch hunk in {path} (no .orig backup was present)")
        return 0
    print("Nothing to revert -- the file is already pristine.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "apply", "revert"])
    args = ap.parse_args()

    path = target_path()
    if args.action == "status":
        print(describe(path))
        return 0
    return apply(path) if args.action == "apply" else revert(path)


if __name__ == "__main__":
    raise SystemExit(main())
