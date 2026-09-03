#!/usr/bin/env python3
"""Check whether this bug still exists in another torch — it does not, since 2.4.

Run this with a DIFFERENT interpreter than the 2.3.1 one to confirm that modern
torch is unaffected. That matters for two reasons: it stops anyone (including me)
from claiming an open upstream bug, and it shows what upstream's actual fix was.

    /path/to/other/env/bin/python repro/check_upstream.py

Exit codes:
    0 = this torch is UNAFFECTED (expected for >= 2.4)
    2 = this torch still has the bug (expected only for 2.3.x)
"""

import os
import sys
import tempfile

# Same cache hygiene as repro/__init__.py, but standalone: this script is meant
# to be run by a foreign interpreter that cannot import this package.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
    tempfile.gettempdir(), f"torchinductor_upstream_check_{os.getuid()}"
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MARKER = "Cannot convert expression to float"


def main() -> int:
    print(f"torch {torch.__version__}")

    def fn(x, ref):
        return F.interpolate(x, size=(ref.shape[-2], ref.shape[-1]), mode="nearest")

    x = torch.randn(1, 3, 32, 32)
    ref = torch.randn(1, 3, 64, 64)
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)

    torch._dynamo.reset()
    with torch.inference_mode():
        try:
            got = torch.compile(fn)(x, ref)
            expected = fn(x, ref)
        except Exception as exc:  # noqa: BLE001
            if MARKER in str(exc):
                print("  AFFECTED: reproduces the ops.constant bug.")
                print("  (expected only on torch 2.3.x)")
                return 2
            print(f"  Inconclusive: unrelated {type(exc).__name__}: {str(exc)[:160]}")
            return 1

    print(f"  UNAFFECTED: compiles, shape {tuple(got.shape)}, "
          f"bit-exact={torch.equal(got, expected)}")

    # Show *why*. Upstream fixed this at two layers; report both.
    import inspect

    # (1) The dispatch registration -- the fix that closes the hole. If the
    # decomposition is keyed on CompositeImplicitAutograd it also fires under
    # inference_mode, so the buggy lowering is never reached.
    try:
        from torch._decomp import decompositions as d

        dsrc = inspect.getsource(d)
        idx = dsrc.find("def upsample_nearest2d(")
        window = dsrc[max(0, idx - 400):idx] if idx != -1 else ""
        has_composite = "CompositeImplicitAutograd" in window
        print(f"  decomposition also keyed on CompositeImplicitAutograd: {has_composite}")
        if has_composite:
            print("  -> so it fires under inference_mode too, and the buggy")
            print("     upsample_nearestnd lowering is never reached. (In 2.3.1 the")
            print("     decomposition was Autograd-only, which is the whole bug.)")
    except Exception:  # noqa: BLE001
        pass

    # (2) The callee -- defence in depth.
    try:
        from torch._inductor.index_propagation import SymPyOps

        coerces = "float(value)" in inspect.getsource(SymPyOps.constant)
        print(f"  SymPyOps.constant still coerces with float(): {coerces}")
        if not coerces:
            print("  -> and the eager sympy.Float(float(...)) coercion is gone, so a")
            print("     symbolic value could not raise there even if it arrived.")
    except Exception:  # noqa: BLE001
        pass

    # (3) What was NOT fixed: the caller-side asymmetry is still in the source.
    try:
        import torch._inductor.lowering as L

        lsrc = inspect.getsource(L.upsample_nearestnd)
        still_lax = ("inv_scales = [i / o" in lsrc
                     and "ops.constant(scale" in lsrc)
        print(f"  caller still passes a possibly-symbolic scale to ops.constant: "
              f"{still_lax}")
        if still_lax:
            print("  -> the type-contract violation itself was never repaired, only")
            print("     made unreachable. Latent, not fixed.")
    except Exception:  # noqa: BLE001
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
