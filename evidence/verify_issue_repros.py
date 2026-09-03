#!/usr/bin/env python
"""Run the repro snippets from issues.md VERBATIM and report what they print.

An issue whose repro does not run is worse than no issue. These are copy-pasted
from issues.md; if the output here does not match what the issue body claims,
fix the issue body.

    python verify_issue_repros.py            # needs CUDA for Issue A
"""
import io
import contextlib
import traceback

import torch

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu   {torch.cuda.get_device_name(0)}")
print()

results = {}


def section(name):
    def deco(f):
        print(f"{'=' * 70}\n{name}\n{'=' * 70}")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                f()
            out = buf.getvalue()
            results[name] = ("ran", out)
            print(out.rstrip() or "(no output)")
            print(f"--> repro RAN as written")
        except Exception as e:
            out = buf.getvalue()
            if out:
                print(out.rstrip())
            results[name] = ("raised", f"{type(e).__name__}: {e}")
            print(f"--> repro RAISED {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(limit=2)
        print()
        return f

    return deco


# --------------------------------------------------------------- Issue A
if torch.cuda.is_available():

    @section("Issue A: wrong pixels, dynamic=True, CUDA")
    def issue_a():
        import torch.nn.functional as F

        def rows(t):
            return t[0, 0, :, 0].to(torch.int64).tolist()

        for mode, i, o in [
            ("nearest", 448, 192),
            ("nearest", 384, 363),
            ("nearest", 37, 74),
            ("nearest-exact", 384, 363),
        ]:
            x = (
                torch.arange(i, device="cuda", dtype=torch.float32)
                .view(1, 1, i, 1)
                .expand(1, 1, i, 4)
                .contiguous()
            )
            f = lambda t: F.interpolate(t, size=(o, 4), mode=mode)
            torch._dynamo.reset()
            eager = rows(f(x))
            compiled = rows(torch.compile(f, dynamic=True)(x))
            wrong = [
                (d, e, g) for d, (e, g) in enumerate(zip(eager, compiled)) if e != g
            ]
            print(f"{mode} {i}->{o}: {len(wrong)}/{o} rows differ; first {wrong[:1]}")


# --------------------------------------------------------------- Issue B
@section("Issue B: upsample_nearestnd symbolic output_size")
def issue_b():
    from torch._inductor.lowering import register_lowering, upsample_nearestnd

    with torch.library._scoped_library("demo", "FRAGMENT") as lib:
        lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
        lib.impl(
            "ups2d", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta"
        )
        lib.impl(
            "ups2d",
            lambda x, h, w: torch.nn.functional.interpolate(
                x, size=(int(h), int(w)), mode="nearest"
            ),
            "CPU",
        )
        register_lowering(torch.ops.demo.ups2d)(
            lambda x, h, w: upsample_nearestnd(x, [h, w], (None, None), n=2)
        )

        def fn(x, sizes):
            return torch.ops.demo.ups2d(x, sizes[0], sizes[1])

        x = torch.randn(1, 3, 32, 32)
        ref = torch.randn(1, 3, 100, 70)
        torch._dynamo.maybe_mark_dynamic(ref, 2)
        torch._dynamo.maybe_mark_dynamic(ref, 3)
        print(torch.compile(fn, dynamic=True)(x, ref.shape[-2:]).shape)


# --------------------------------------------------------------- Issue C
@section("Issue C: ops.constant accepts a symbolic value")
def issue_c():
    import sympy
    from torch._inductor.virtualized import ops

    ops.constant(32 * sympy.Symbol("s0", integer=True, positive=True), torch.float32)


# --------------------------------------------------------------- Issue D
@section("Issue D: upsample_nearest2d_backward symbolic input_size")
def issue_d():
    def fn(grad, ref):
        return torch.ops.aten.upsample_nearest2d_backward.default(
            grad,
            [grad.shape[-2], grad.shape[-1]],
            [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]],
        )

    grad = torch.randn(1, 3, 64, 64)
    ref = torch.randn(1, 3, 32, 32)
    torch._dynamo.maybe_mark_dynamic(ref, 2)
    torch._dynamo.maybe_mark_dynamic(ref, 3)
    print(torch.compile(fn, dynamic=True)(grad, ref).shape)


print("=" * 70)
print("SUMMARY (what the issue body must match)")
print("=" * 70)
for k, (kind, v) in results.items():
    print(f"{kind.upper():<8}{k}")
    if kind == "raised":
        print(f"        {v[:150]}")
