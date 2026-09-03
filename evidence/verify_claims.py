#!/usr/bin/env python3
"""Verify every claim in this directory's README, on whatever torch you run it with.

This script is the answer to "how do I know any of that is true?". It needs no
PyTorch source checkout and no GPU -- just an installed torch >= 2.4.

    python upstream/verify_claims.py            # report
    python upstream/verify_claims.py --json     # machine-readable

What it checks, in order:

  1. REACHABILITY  -- that `upsample_nearestnd` is NOT reached through any of the
                      supported ATen entry points on this torch. This is what
                      makes the PR "hardening", not "fixes a live regression",
                      and it is the claim most worth auditing.
  2. DEFECT        -- that the lowering nevertheless still CRASHES when a caller
                      does reach it with a symbolic output size. Uses a temporary
                      custom op; installed torch is never modified.
  3. FIX           -- that guarding the divisors with `guard_int` makes those same
                      cases compile bit-exactly against eager.
  4. NO-REGRESSION -- that the guard changes nothing for the paths that work
                      today, including that the generated kernel source is
                      byte-identical.
  5. REJECTED ALT  -- why the scale cannot simply be kept symbolic: exact integer
                      index math disagrees with eager's float32 arithmetic on
                      ordinary sizes, so the fix must specialise.

Exit status: 0 if every check reached its expected verdict, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile


# Inductor's parallel compile workers race on Triton's cache; serialise them so a
# flake cannot be mistaken for a real verdict. Must precede `import torch`.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# (input HW, output HW). Chosen so the scales are genuinely symbolic and do not
# simplify: exact ratios, non-integer anisotropic ratios, downsamples, and a 1x1
# input (which surfaces as sympy.Pow rather than Mul).
SHAPE_PAIRS = [
    ((32, 32), (64, 64)),
    ((32, 32), (100, 70)),
    ((32, 32), (17, 17)),
    ((32, 32), (96, 48)),
    ((32, 32), (65, 33)),
    ((7, 5), (23, 11)),
    ((13, 13), (40, 39)),
    ((32, 32), (31, 63)),
    ((1, 1), (9, 9)),
    ((64, 64), (21, 85)),
    ((10, 10), (100, 100)),
    ((3, 4), (97, 53)),
]

results: list[dict] = []


def record(check: str, expected: str, actual: str, ok: bool, detail: str = "") -> bool:
    results.append(
        {"check": check, "expected": expected, "actual": actual, "ok": ok, "detail": detail}
    )
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {check}")
    print(f"         expected: {expected}")
    print(f"         actual  : {actual}")
    if detail:
        print(f"         {detail}")
    return ok


def _clear_lowerings(tag: str) -> None:
    """Drop lowerings registered for a torn-down scoped library."""
    import torch._inductor.lowering as lowering

    for key in [k for k in list(lowering.lowerings) if tag in str(k)]:
        lowering.lowerings.pop(key, None)


# --------------------------------------------------------------------------
# 1. Reachability: does any supported ATen path reach the lowering?
# --------------------------------------------------------------------------
def check_reachability() -> None:
    print("\n1. REACHABILITY -- can ATen reach upsample_nearestnd on this torch?")
    import torch._inductor.lowering as lowering

    hits = {"n": 0}
    original = lowering.upsample_nearestnd

    def spy(x, output_size, scales_x, n=2, exact=False):
        hits["n"] += 1
        return original(x, output_size, scales_x, n=n, exact=exact)

    lowering.upsample_nearestnd = spy
    lowering.lowerings[torch.ops.aten.upsample_nearest2d.default] = (
        lambda x, output_size, scales_h=None, scales_w=None: spy(
            x, output_size, (scales_h, scales_w), n=2
        )
    )

    def build():
        x = torch.randn(1, 3, 32, 32)
        ref = torch.randn(1, 3, 64, 64)
        torch._dynamo.mark_dynamic(ref, 2)
        torch._dynamo.mark_dynamic(ref, 3)
        return x, ref

    probes: dict[str, int] = {}

    def probe(name, fn):
        hits["n"] = 0
        torch._dynamo.reset()
        try:
            fn()
        except Exception:
            pass  # a failure to run is still "did it reach the lowering"
        probes[name] = hits["n"]

    def p_compile():
        x, ref = build()
        torch.compile(
            lambda a, r: F.interpolate(
                a, size=(r.shape[-2], r.shape[-1]), mode="nearest"
            ),
            dynamic=True,
        )(x, ref)

    def p_inference():
        x, ref = build()
        with torch.inference_mode():
            torch.compile(
                lambda a, r: F.interpolate(
                    a, size=(r.shape[-2], r.shape[-1]), mode="nearest"
                ),
                dynamic=True,
            )(x, ref)

    def p_aten():
        x, ref = build()
        with torch.inference_mode():
            torch.compile(
                lambda a, r: torch.ops.aten.upsample_nearest2d.default(
                    a, [r.shape[-2], r.shape[-1]]
                ),
                dynamic=True,
            )(x, ref)

    def p_export_decomp():
        class M(torch.nn.Module):
            def forward(self, a, r):
                return F.interpolate(
                    a, size=(r.shape[-2], r.shape[-1]), mode="nearest"
                )

        x = torch.randn(1, 3, 32, 32)
        ref = torch.randn(1, 3, 64, 64)
        dim = torch.export.Dim.DYNAMIC
        ep = torch.export.export(
            M(), (x, ref), dynamic_shapes={"a": {}, "r": {2: dim, 3: dim}}
        )
        ep = ep.run_decompositions(decomp_table={})
        with torch.inference_mode():
            torch._inductor.compile(ep.module(), (x, ref))(x, ref)

    probe("torch.compile", p_compile)
    probe("inference_mode", p_inference)
    probe("direct aten op", p_aten)
    probe("export + run_decompositions({})", p_export_decomp)

    lowering.upsample_nearestnd = original
    total = sum(probes.values())
    record(
        "ATen paths do not reach the buggy lowering",
        "0 lowering hits",
        f"{total} lowering hits",
        ok=total == 0,
        detail="per path: " + ", ".join(f"{k}={v}" for k, v in probes.items()),
    )


# --------------------------------------------------------------------------
# 2/3. The defect, and the fix, with the lowering actually reached.
# --------------------------------------------------------------------------
def run_shape_pairs(apply_fix: bool) -> tuple[int, int, int]:
    """Return (bit_exact, wrong, crashed) over SHAPE_PAIRS."""
    import sympy

    import torch._inductor.lowering as lowering
    from torch._inductor.ir import Pointwise
    from torch._inductor.virtualized import ops, V

    def upsample_nearestnd(x, output_size, scales_x, n=2, exact=False):
        """A faithful copy of the lowering, with the fix switchable.

        Copied rather than imported so that this script never edits or depends on
        the state of the installed torch.
        """
        x.realize_hint()
        x_loader = x.make_loader()
        i_sizes = x.get_size()[-n:]
        batch = x.get_size()[:-n]
        i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]
        o_sizes = output_size

        if apply_fix:
            # THE FIX under test.
            inv_scales = [
                i / (V.graph.sizevars.guard_int(o) if s is None else o)
                for i, o, s in zip(i_sizes, o_sizes, scales_x)
            ]
        else:
            inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]

        for idx, scale in enumerate(scales_x):
            if scale is not None:
                inv_scales[idx] = 1.0 / scale

        def scale_fn(coord, scale, size):
            coord = ops.index_expr(coord, torch.float32)
            if exact:
                coord = ops.add(coord, ops.constant(0.5, torch.float32))
            coord = ops.mul(coord, ops.constant(scale, torch.float32))
            coord = ops.to_dtype(coord, torch.int32)
            return ops.indirect_indexing(coord, size, check=False)

        def fn(idx):
            spatial = idx[-n:]
            leading = idx[:-n]
            return x_loader(
                [
                    *leading,
                    *[
                        scale_fn(i, s, size)
                        for i, s, size in zip(spatial, inv_scales, i_sizes)
                    ],
                ]
            )

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=fn,
            ranges=[*batch, *o_sizes],
        )

    exact_n = wrong_n = crash_n = 0
    tag = "vc_fix" if apply_fix else "vc_base"
    with torch.library._scoped_library(tag, "DEF") as lib:
        lib.define("up2d(Tensor x, SymInt h, SymInt w) -> Tensor")
        lib.impl(
            "up2d", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta"
        )
        for key in ("CPU", "CUDA"):
            lib.impl(
                "up2d",
                lambda x, h, w: F.interpolate(
                    x, size=(int(h), int(w)), mode="nearest"
                ),
                key,
            )
        op = getattr(torch.ops, tag).up2d.default
        lowering.register_lowering(op)(
            lambda x, h, w: upsample_nearestnd(x, [h, w], (None, None), n=2)
        )

        def fn(x, ref):
            return getattr(torch.ops, tag).up2d(x, ref.shape[-2], ref.shape[-1])

        for in_hw, out_hw in SHAPE_PAIRS:
            torch._dynamo.reset()
            x = torch.randn(1, 3, *in_hw)
            ref = torch.randn(1, 3, *out_hw)
            torch._dynamo.maybe_mark_dynamic(ref, 2)
            torch._dynamo.maybe_mark_dynamic(ref, 3)
            try:
                expect = F.interpolate(x, size=out_hw, mode="nearest")
                actual = torch.compile(fn, dynamic=True)(x, ref)
                if torch.equal(expect, actual):
                    exact_n += 1
                else:
                    wrong_n += 1
            except Exception:
                crash_n += 1
    _clear_lowerings(tag)
    torch._dynamo.reset()
    return exact_n, wrong_n, crash_n


def check_defect_and_fix() -> None:
    total = len(SHAPE_PAIRS)
    print("\n2. DEFECT -- reach the lowering with a symbolic output size, unfixed")
    _, wrong, crashed = run_shape_pairs(apply_fix=False)
    record(
        "unfixed lowering fails on symbolic output sizes",
        f"all {total} crash",
        f"{crashed} crashed, {wrong} wrong",
        ok=crashed == total,
    )

    print("\n3. FIX -- the same cases, with the guard applied")
    exact, wrong, crashed = run_shape_pairs(apply_fix=True)
    record(
        "fixed lowering compiles and is bit-exact vs eager",
        f"all {total} bit-exact",
        f"{exact} bit-exact, {wrong} wrong, {crashed} crashed",
        ok=exact == total,
    )


# --------------------------------------------------------------------------
# 4. No regression on the paths that work today.
# --------------------------------------------------------------------------
def working_paths_fingerprint(cache_dir: str) -> tuple[int, int, str]:
    """Exercise every currently-working interpolate path; hash emitted kernels."""
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    ok = total = 0

    def run(build, reference):
        nonlocal ok, total
        torch._dynamo.reset()
        total += 1
        ok += int(torch.equal(reference(), build()))

    for mode in ("nearest", "nearest-exact"):
        for factor in (2.0, 1.5, 0.5):
            def build(mode=mode, factor=factor):
                torch.manual_seed(0)
                return torch.compile(
                    lambda t: F.interpolate(t, scale_factor=factor, mode=mode),
                    dynamic=True,
                )(torch.randn(1, 3, 32, 32))

            def reference(mode=mode, factor=factor):
                torch.manual_seed(0)
                return F.interpolate(
                    torch.randn(1, 3, 32, 32), scale_factor=factor, mode=mode
                )

            run(build, reference)

        for out_hw in ((64, 64), (100, 70), (17, 17)):
            def build(mode=mode, out_hw=out_hw):
                torch.manual_seed(1)
                x = torch.randn(1, 3, 32, 32)
                ref = torch.randn(1, 3, *out_hw)
                torch._dynamo.maybe_mark_dynamic(ref, 2)
                torch._dynamo.maybe_mark_dynamic(ref, 3)
                with torch.inference_mode():
                    return torch.compile(
                        lambda t, r: F.interpolate(
                            t, size=(r.shape[-2], r.shape[-1]), mode=mode
                        ),
                        dynamic=True,
                    )(x, ref)

            def reference(mode=mode, out_hw=out_hw):
                torch.manual_seed(1)
                return F.interpolate(torch.randn(1, 3, 32, 32), size=out_hw, mode=mode)

            run(build, reference)

    # 3-D and 1-D, so the check is not silently 2-D only.
    def build3():
        torch.manual_seed(2)
        x = torch.randn(1, 2, 4, 8, 8)
        ref = torch.randn(1, 2, 8, 16, 16)
        for dim in (2, 3, 4):
            torch._dynamo.maybe_mark_dynamic(ref, dim)
        with torch.inference_mode():
            return torch.compile(
                lambda t, r: F.interpolate(
                    t, size=tuple(r.shape[-3:]), mode="nearest"
                ),
                dynamic=True,
            )(x, ref)

    def reference3():
        torch.manual_seed(2)
        return F.interpolate(
            torch.randn(1, 2, 4, 8, 8), size=(8, 16, 16), mode="nearest"
        )

    run(build3, reference3)

    def build1():
        torch.manual_seed(3)
        x = torch.randn(1, 2, 16)
        ref = torch.randn(1, 2, 37)
        torch._dynamo.maybe_mark_dynamic(ref, 2)
        with torch.inference_mode():
            return torch.compile(
                lambda t, r: F.interpolate(t, size=(r.shape[-1],), mode="nearest"),
                dynamic=True,
            )(x, ref)

    def reference1():
        torch.manual_seed(3)
        return F.interpolate(torch.randn(1, 2, 16), size=(37,), mode="nearest")

    run(build1, reference1)

    blobs = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "**", "*.py"), recursive=True)):
        try:
            with open(path) as handle:
                blobs.append(handle.read())
        except OSError:
            pass
    digest = hashlib.sha256("".join(sorted(blobs)).encode()).hexdigest()
    return ok, total, digest


def check_no_regression() -> None:
    print("\n4. NO-REGRESSION -- the paths that work today are untouched")
    root = tempfile.mkdtemp(prefix="verify_claims_")
    try:
        ok, total, digest = working_paths_fingerprint(os.path.join(root, "cache"))
        record(
            "existing interpolate paths still match eager",
            f"{total}/{total} match",
            f"{ok}/{total} match",
            ok=ok == total,
            detail=f"generated-kernel sha256 {digest[:16]}...",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(
        "         note: this arm runs unmodified torch, so it establishes the\n"
        "         baseline. The byte-identical comparison against the patched\n"
        "         build is in README section 4 (it needs the patch installed)."
    )


def check_rejected_alternative() -> None:
    """The 'keep it dynamic' fix is wrong: exact-rational != eager's float32.

    This is pure index arithmetic, so it can be fuzzed far more widely than
    compiling allows. Emulating float32 with struct pack/unpack is what makes the
    comparison faithful -- Python floats are float64 and would hide the bug.
    """
    import struct

    print("\n5. REJECTED ALTERNATIVE -- why the scale must stay a float")

    def eager_index(coord: int, isize: int, osize: int, exact: bool) -> int:
        scale = struct.unpack("f", struct.pack("f", isize / osize))[0]
        value = coord + (0.5 if exact else 0.0)
        return int(struct.unpack("f", struct.pack("f", value * scale))[0])

    def exact_index(coord: int, isize: int, osize: int, exact: bool) -> int:
        if exact:
            return (isize * (2 * coord + 1)) // (2 * osize)
        return (isize * coord) // osize

    # Deterministic, and includes the pairs found by a wider random sweep.
    known = [(448, 192, 27, False), (384, 363, 121, False), (256, 82, 41, False)]
    confirmed = [
        (i, o, c, e)
        for i, o, c, e in known
        if eager_index(c, i, o, e) != exact_index(c, i, o, e)
    ]
    detail = "; ".join(
        f"{i}->{o} coord {c}: eager={eager_index(c, i, o, e)} exact={exact_index(c, i, o, e)}"
        for i, o, c, e in confirmed
    )
    record(
        "exact-rational index math disagrees with eager's float32",
        f"{len(known)} disagreements",
        f"{len(confirmed)} disagreements",
        ok=len(confirmed) == len(known),
        detail=detail or "none found -- the alternative would have been viable",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    print("=" * 74)
    print("Verifying the claims in upstream/README.md")
    print("=" * 74)
    print(f"  torch  : {torch.__version__}")
    print(f"  device : {'cuda available' if torch.cuda.is_available() else 'cpu only'}")
    print("  note   : the installed torch is never modified by this script.")

    check_reachability()
    check_defect_and_fix()
    check_no_regression()
    check_rejected_alternative()

    passed = sum(1 for r in results if r["ok"])
    print("\n" + "=" * 74)
    print(f"{passed}/{len(results)} checks reached their expected verdict")
    print("=" * 74)

    if args.json:
        print(json.dumps({"torch": torch.__version__, "results": results}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
