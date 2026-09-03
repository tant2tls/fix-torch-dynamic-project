#!/usr/bin/env python3
"""Print the post-AOT graph under three grad contexts, side by side.

This is the script that actually explains the bug's elusiveness. Under normal
autograd the `aten.upsample_nearest2d` op is *decomposed away* before Inductor
sees it, so the buggy lowering is unreachable. Under `inference_mode` the op
survives -- and that is when it hits the lowering.

    python repro/why_it_survives.py [--device cpu|cuda|auto] [--full]

The mechanism: the decomposition is registered with

    @aten.upsample_nearest2d.default.py_impl(DispatchKey.Autograd)

`inference_mode` excludes the Autograd dispatch key, so that py_impl never
fires and the raw aten op reaches Inductor.
"""

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repro  # noqa: F401  -- sets TORCHINDUCTOR_CACHE_DIR before torch

import torch  # noqa: E402

from repro.bug import pick_device, upsample_to_match, version_warning  # noqa: E402


def graph_under(ctx_factory, device: str):
    """Compile under a grad context and capture the post-AOT forward graph."""
    captured = {}

    def fw_compiler(gm, example_inputs):
        captured["ops"] = [str(n.target) for n in gm.graph.nodes
                           if n.op == "call_function"]
        captured["readable"] = gm.print_readable(print_output=False)
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(gm, example_inputs)

    from torch._dynamo.backends.common import aot_autograd

    torch._dynamo.reset()
    compiled = torch.compile(
        upsample_to_match,
        backend=lambda gm, ex: aot_autograd(fw_compiler=fw_compiler)(gm, ex),
    )

    x = torch.randn(1, 3, 32, 32, device=device)
    ref = torch.randn(1, 3, 64, 64, device=device)
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)

    err = None
    try:
        with ctx_factory():
            compiled(x, ref)
    except Exception as exc:  # noqa: BLE001
        err = ("BUG" if "Cannot convert expression to float" in str(exc)
               else type(exc).__name__)
    return captured, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--full", action="store_true",
                    help="print the entire graph, not just the relevant ops")
    args = ap.parse_args()

    warn = version_warning()
    if warn:
        print(warn + "\n")
    dev = pick_device(args.device)

    contexts = [
        ("grad enabled (default)", contextlib.nullcontext),
        ("torch.no_grad()", torch.no_grad),
        ("torch.inference_mode()", torch.inference_mode),
    ]

    print("=" * 78)
    print(f"Does aten.upsample_nearest2d survive into Inductor? "
          f"(torch {torch.__version__}, {dev})")
    print("=" * 78)

    for label, ctx in contexts:
        cap, err = graph_under(ctx, dev)
        ops = cap.get("ops", [])
        survives = any("upsample_nearest" in o for o in ops)

        print(f"\n--- {label} " + "-" * (72 - len(label)))
        print(f"  aten.upsample_nearest2d survives : {survives}")
        print(f"  compile result                   : "
              f"{'FAILS with the bug' if err == 'BUG' else (err or 'compiles fine')}")

        interesting = [o for o in ops
                       if any(k in o for k in
                              ("upsample", "unsafe_index", "arange", "_to_copy", "mul"))]
        print(f"  relevant ops in the graph        : {interesting}")

        if args.full and cap.get("readable"):
            print("\n" + cap["readable"])

    print("\n" + "=" * 78)
    print("Conclusion")
    print("=" * 78)
    print("""
With grad enabled OR under no_grad, the op is decomposed into
  arange -> add -> mul -> _to_copy -> _unsafe_index
i.e. the index arithmetic becomes real tensor ops, and `upsample_nearestnd`
in _inductor/lowering.py is NEVER CALLED. No symbolic value ever reaches
ops.constant, so there is nothing to crash.

Under inference_mode, `aten.upsample_nearest2d.default` reaches Inductor
intact, gets lowered by `upsample_nearestnd`, and the symbolic output size
flows into ops.constant -> sympy.Float(float(expr)) -> TypeError.

Why: the decomposition is registered as
    @aten.upsample_nearest2d.default.py_impl(DispatchKey.Autograd)
and inference_mode excludes the Autograd key. The guard against this bug was
never in the lowering -- it was an accident of dispatch. A production
inference server, which of course runs under inference_mode, removes that
accidental protection.

That is the systems lesson: the safe path was load-bearing but undocumented,
and the configuration that removes it (inference_mode + dynamic shapes) is
exactly the configuration you deploy.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
