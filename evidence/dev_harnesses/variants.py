#!/usr/bin/env python3
"""Compare candidate fixes for the `upsample_nearestnd` symbolic-scale defect.

The lowering divides guarded input sizes by *unguarded* output sizes and hands
the quotient to `ops.constant`, which requires a concrete leaf value. With a
symbolic output size the quotient stays a `sympy.Mul` and compilation dies.

Four variants of the divisor line are compared:

  baseline  : `i / o`                      -- today's code, crashes when symbolic
  guard_int : `i / guard_int(o)`           -- concretise the divisor
  f32       : runtime `ops.truediv` in fp32 when `o` is symbolic
  f64       : runtime `ops.truediv` in fp64, narrowed to fp32, when symbolic

and judged on four axes:

  correctness : bit-exact vs eager on symbolic output sizes (`--mode correctness`)
  codegen     : byte-identical kernel source on the paths that work today,
                i.e. is the change a strict no-op? (`--mode codegen`)
  graphs      : how many graphs N distinct output sizes compile to (`--mode graphs`)
  constraint  : behaviour under a *hard* `mark_dynamic` on an output dim
                (`--mode constraint`)

Every arm must be run on CUDA as well as CPU: the fp32 variant is bit-exact on
CPU and silently WRONG on CUDA, because Triton lowers fp32 `/` to the
approximate `div.full`. That single-ULP difference flips a floored index.

    python variants.py --mode all --device cpu
    python variants.py --mode all --device cuda

Installed torch is never modified; the lowering is re-implemented here and
reached through a scoped custom op.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
import traceback

# Inductor's parallel compile workers race on Triton's cache; serialise them so
# a flake cannot be mistaken for a verdict. Must precede `import torch`.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import sympy  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch._inductor.ir import Pointwise  # noqa: E402
from torch._inductor.lowering import register_lowering  # noqa: E402
from torch._inductor.utils import run_and_get_code  # noqa: E402
from torch._inductor.virtualized import V, ops  # noqa: E402


VARIANTS = ("baseline", "guard_int", "f32", "f64")

# (input HW, output HW). Ratios that are exact, anisotropic non-integer,
# downsampling, and -- critically -- (448,448)->(192,192) and (384,384)->(363,363),
# the pairs where a one-ULP error in the scale flips a floored index.
CORRECTNESS_PAIRS = [
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
    ((448, 448), (192, 192)),
    ((384, 384), (363, 363)),
    ((256, 256), (82, 82)),
    ((10, 10), (100, 100)),
    ((3, 4), (97, 53)),
]

# Concrete cases that compile fine today. A fix must leave these byte-identical.
NOREGRESSION_CASES = [
    # (in_hw, out_hw, scales_x)
    ((32, 32), (64, 64), (None, None)),      # static size=
    ((32, 32), (64, 64), (2.0, 2.0)),        # scale_factor=2
    ((32, 32), (48, 48), (1.5, 1.5)),        # scale_factor=1.5
    ((32, 32), (16, 16), (0.5, 0.5)),        # downsample
    ((32, 32), (100, 70), (None, None)),     # anisotropic static
    ((32, 32), (17, 17), (None, None)),      # static downsample
    ((32, 32), (64, 32), (2.0, None)),       # mixed: one scale, one size
    ((13, 13), (40, 39), (None, None)),      # awkward static
]


def _symbolic(o) -> bool:
    """Is this output size a genuinely symbolic expression?

    Mirrors `lowering.py`'s own concreteness idiom (see `_full`, which does
    `isinstance(x, sympy.Expr) and not x.is_number`).
    """
    return isinstance(o, sympy.Expr) and not o.is_number


def build_lowering(variant: str, exact: bool, counter: dict):
    """Return a copy of `upsample_nearestnd` with the divisor line swapped out."""

    def lowering(x, output_size, scales_x, n=2):
        counter["calls"] += 1
        x.realize_hint()
        x_loader = x.make_loader()
        i_sizes = x.get_size()[-n:]
        batch = x.get_size()[:-n]
        i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]
        o_sizes = output_size

        # ---- the line under test -------------------------------------------
        inv_scales: list = []
        for i, o, s in zip(i_sizes, o_sizes, scales_x):
            if variant == "baseline" or s is not None or not _symbolic(o):
                # `s is not None` is overwritten below with 1.0/s anyway; and a
                # concrete `o` must keep today's folded constant so that the
                # generated kernel stays byte-identical.
                inv_scales.append(i / o)
            elif variant == "guard_int":
                inv_scales.append(i / V.graph.sizevars.guard_int(o))
            elif variant in ("f32", "f64"):
                # Defer the division into the kernel. Marker tuple; the actual
                # ops calls must happen inside `inner_fn` during codegen.
                inv_scales.append(("runtime", i, o))
            else:
                raise AssertionError(f"unknown variant {variant!r}")
        # --------------------------------------------------------------------

        for idx, scale in enumerate(scales_x):
            if scale is not None:
                inv_scales[idx] = 1.0 / scale

        def scale_fn(coord, scale, size):
            coord = ops.index_expr(coord, torch.float32)
            if exact:
                coord = ops.add(coord, ops.constant(0.5, torch.float32))
            if isinstance(scale, tuple):
                _, i, o = scale
                if variant == "f32":
                    sv = ops.truediv(
                        ops.constant(i, torch.float32),
                        ops.index_expr(o, torch.float32),
                    )
                else:  # f64: compute wide, then narrow
                    sv = ops.to_dtype(
                        ops.truediv(
                            ops.constant(i, torch.float64),
                            ops.index_expr(o, torch.float64),
                        ),
                        torch.float32,
                    )
                coord = ops.mul(coord, sv)
            else:
                coord = ops.mul(coord, ops.constant(scale, torch.float32))
            coord = ops.to_dtype(coord, torch.int32)
            return ops.indirect_indexing(coord, size, check=False)

        def fn(idx):
            coords = idx[-n:]
            b = idx[:-n]
            return x_loader(
                [
                    *b,
                    *[
                        scale_fn(c, s, size)
                        for c, s, size in zip(coords, inv_scales, i_sizes)
                    ],
                ]
            )

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=fn,
            ranges=[*batch, *o_sizes],
        )

    return lowering


def _clear_lowerings(tag: str) -> None:
    import torch._inductor.lowering as lowering_mod

    for key in [k for k in list(lowering_mod.lowerings) if tag in str(k)]:
        lowering_mod.lowerings.pop(key, None)


class Harness:
    """A scoped custom op whose lowering is the variant under test."""

    def __init__(self, tag: str, variant: str, exact: bool, scales_x):
        self.tag = tag
        self.variant = variant
        self.exact = exact
        self.scales_x = scales_x
        self.counter = {"calls": 0}

    def __enter__(self):
        self._lib = torch.library._scoped_library(self.tag, "FRAGMENT")
        lib = self._lib.__enter__()
        lib.define("ups(Tensor x, SymInt h, SymInt w) -> Tensor")

        exact, scales_x = self.exact, self.scales_x
        mode = "nearest-exact" if exact else "nearest"

        def meta(x, h, w):
            return x.new_empty((x.shape[0], x.shape[1], h, w))

        def ref(x, h, w):
            return F.interpolate(x, size=(int(h), int(w)), mode=mode)

        lib.impl("ups", meta, "Meta")
        lib.impl("ups", ref, "CPU")
        lib.impl("ups", ref, "CUDA")

        low = build_lowering(self.variant, exact, self.counter)
        op = getattr(torch.ops, self.tag).ups
        register_lowering(op)(lambda x, h, w: low(x, [h, w], scales_x, n=2))
        self.op = op
        self.eager = ref
        return self

    def __exit__(self, *exc):
        self._lib.__exit__(*exc)
        _clear_lowerings(self.tag)
        return False


_tags = itertools.count()


def _tag() -> str:
    return f"vdev{next(_tags)}"


# --------------------------------------------------------------------------
# correctness: symbolic output sizes, bit-exact vs eager
# --------------------------------------------------------------------------
def mode_correctness(device: str, variants) -> dict:
    print(f"\n=== correctness: bit-exact vs eager, symbolic sizes, {device} ===")
    out = {}
    for variant in variants:
        exact_n = wrong_n = crash_n = 0
        worst = []
        for exact in (False, True):
            for in_hw, out_hw in CORRECTNESS_PAIRS:
                with Harness(_tag(), variant, exact, (None, None)) as h:

                    def fn(x, sizes):
                        return h.op(x, sizes[0], sizes[1])

                    torch._dynamo.reset()
                    x = torch.randn(1, 3, *in_hw, device=device)
                    ref_t = torch.randn(1, 3, *out_hw, device=device)
                    torch._dynamo.maybe_mark_dynamic(ref_t, 2)
                    torch._dynamo.maybe_mark_dynamic(ref_t, 3)
                    expect = h.eager(x, *out_hw)
                    try:
                        got = torch.compile(fn, dynamic=True)(x, ref_t.shape[-2:])
                        if torch.equal(got, expect):
                            exact_n += 1
                        else:
                            wrong_n += 1
                            n_bad = int((got != expect).sum())
                            worst.append(
                                f"{in_hw}->{out_hw}"
                                f"{'/exact' if exact else ''}: "
                                f"{n_bad}/{expect.numel()} elems"
                            )
                    except Exception as e:  # noqa: BLE001
                        crash_n += 1
                        if len(worst) < 3:
                            worst.append(
                                f"{in_hw}->{out_hw}: "
                                f"{type(e).__name__}: {str(e).splitlines()[-1][:90]}"
                            )
        total = 2 * len(CORRECTNESS_PAIRS)
        out[variant] = (exact_n, wrong_n, crash_n, total)
        print(
            f"  {variant:10s} bit-exact {exact_n:3d}/{total}  "
            f"wrong {wrong_n:3d}  crashed {crash_n:3d}"
        )
        for w in worst[:4]:
            print(f"             ! {w}")
    return out


# --------------------------------------------------------------------------
# codegen: are the paths that work today byte-identical?
# --------------------------------------------------------------------------
def _normalise_code(blob: str, tag: str) -> str:
    """Strip per-compile identifiers so two compiles of the same math match.

    The generated source embeds the scoped-library tag, the cache-key hash, and
    kernel/buffer names that carry that hash. Without normalising these, a
    baseline-vs-baseline comparison is already unstable, so the whole
    no-regression check would be meaningless.
    """
    import re

    blob = blob.replace(tag, "TAG")
    # a global per-process compile counter, unrelated to the generated math
    blob = re.sub(r"# AOT ID: \[[^\]]*\]", "# AOT ID: [NORM]", blob)
    # kernel names: triton_poi_fused_..._0 / cpp_fused_..._0 carry a content hash
    blob = re.sub(r"\b(triton|cpp)_(\w*?)fused_\w+", r"\1_fused_NORM", blob)
    # cache dirs / file stems: /tmp/torchinductor_x/ab/cabcdef.py, 'cbAbCd'
    blob = re.sub(r"[0-9a-z]{2}/c[0-9a-z]{20,}", "HASHPATH", blob)
    blob = re.sub(r"\bc[0-9a-z]{20,}\b", "HASH", blob)
    blob = re.sub(r"async_compile\.triton\('[^']*'", "async_compile.triton('K'", blob)
    # torch version / device-property banners are irrelevant to the math
    blob = re.sub(r"# Topologically Sorted Source Nodes.*", "", blob)
    return blob


def _codegen_hash(variant, exact, in_hw, out_hw, scales_x, device):
    tag = _tag()
    with Harness(tag, variant, exact, scales_x) as h:

        def fn(x, hh, ww):
            return h.op(x, hh, ww)

        torch._dynamo.reset()
        x = torch.randn(1, 3, *in_hw, device=device)
        # dynamic=False -> concrete output sizes, i.e. today's working path.
        _, code = run_and_get_code(
            torch.compile(fn, dynamic=False), x, out_hw[0], out_hw[1]
        )
        blob = _normalise_code("\n".join(code), tag)
        return hashlib.sha256(blob.encode()).hexdigest(), blob


def mode_codegen(device: str, variants) -> dict:
    print(f"\n=== codegen: byte-identical on working paths, {device} ===")
    cases = [(e, *c) for c in NOREGRESSION_CASES for e in (False, True)]

    # Self-check first: baseline vs baseline must agree, or the normalisation is
    # incomplete and every later verdict is noise.
    k0 = cases[0]
    a, _ = _codegen_hash("baseline", k0[0], k0[1], k0[2], k0[3], device)
    b, _ = _codegen_hash("baseline", k0[0], k0[1], k0[2], k0[3], device)
    print(f"  [self-check] baseline vs baseline: {'STABLE' if a == b else 'UNSTABLE'}")
    if a != b:
        print("  ! normalisation incomplete -- codegen verdicts below are unreliable")

    base = {}
    for exact, in_hw, out_hw, scales_x in cases:
        base[(exact, in_hw, out_hw, scales_x)] = _codegen_hash(
            "baseline", exact, in_hw, out_hw, scales_x, device
        )[0]
    out = {}
    for variant in variants:
        if variant == "baseline":
            continue
        same = 0
        diff = []
        for key in base:
            exact, in_hw, out_hw, scales_x = key
            h, _ = _codegen_hash(variant, exact, in_hw, out_hw, scales_x, device)
            if h == base[key]:
                same += 1
            else:
                diff.append(
                    f"{in_hw}->{out_hw} scales={scales_x}"
                    f"{' exact' if exact else ''}"
                )
        out[variant] = (same, len(base))
        print(f"  {variant:10s} byte-identical {same:2d}/{len(base)}")
        for d in diff[:4]:
            print(f"             ! differs: {d}")
    return out


# --------------------------------------------------------------------------
# graphs: how many recompiles do N distinct output sizes cost?
# --------------------------------------------------------------------------
def mode_graphs(device: str, variants) -> dict:
    print(f"\n=== graphs: distinct output sizes -> compiled graphs, {device} ===")
    sizes = [(64, 64), (72, 72), (80, 80), (96, 96), (100, 100), (128, 128),
             (160, 160), (200, 200)]
    out = {}
    for variant in variants:
        if variant == "baseline":
            continue
        with Harness(_tag(), variant, False, (None, None)) as h:

            def fn(x, s):
                return h.op(x, s[0], s[1])

            torch._dynamo.reset()
            h.counter["calls"] = 0
            compiled = torch.compile(fn, dynamic=True)
            x = torch.randn(1, 3, 32, 32, device=device)
            ok = 0
            for out_hw in sizes:
                ref_t = torch.randn(1, 3, *out_hw, device=device)
                torch._dynamo.maybe_mark_dynamic(ref_t, 2)
                torch._dynamo.maybe_mark_dynamic(ref_t, 3)
                try:
                    compiled(x, ref_t.shape[-2:])
                    ok += 1
                except Exception:  # noqa: BLE001
                    pass
            out[variant] = (h.counter["calls"], ok, len(sizes))
            print(
                f"  {variant:10s} {h.counter['calls']:2d} lowering calls for "
                f"{len(sizes)} sizes ({ok} ran)"
            )
    return out


# --------------------------------------------------------------------------
# constraint: a HARD mark_dynamic on an output dim
# --------------------------------------------------------------------------
def mode_constraint(device: str, variants) -> dict:
    print(f"\n=== constraint: hard mark_dynamic on the output dim, {device} ===")
    sizes = [(64, 64), (72, 72), (80, 80), (96, 96), (100, 100),
             (128, 128), (160, 160), (200, 200), (48, 48), (56, 56)]
    out = {}
    for variant in variants:
        if variant == "baseline":
            continue
        ok = 0
        errs = []
        for out_hw in sizes:
            with Harness(_tag(), variant, False, (None, None)) as h:

                def fn(x, s):
                    return h.op(x, s[0], s[1])

                torch._dynamo.reset()
                x = torch.randn(1, 3, 32, 32, device=device)
                ref_t = torch.randn(1, 3, *out_hw, device=device)
                # HARD constraint: mark_dynamic, not maybe_mark_dynamic.
                torch._dynamo.mark_dynamic(ref_t, 2)
                torch._dynamo.mark_dynamic(ref_t, 3)
                expect = h.eager(x, *out_hw)
                try:
                    got = torch.compile(fn, dynamic=True)(x, ref_t.shape[-2:])
                    if torch.equal(got, expect):
                        ok += 1
                    else:
                        errs.append(f"{out_hw}: WRONG")
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{out_hw}: {type(e).__name__}")
        out[variant] = (ok, len(sizes))
        print(f"  {variant:10s} {ok:2d}/{len(sizes)} compiled and bit-exact")
        for e in errs[:3]:
            print(f"             ! {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "correctness", "codegen", "graphs", "constraint"],
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available", file=sys.stderr)
        return 3

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print("=" * 74)
    print(f"torch {torch.__version__}   device {args.device}")
    print(
        "cuda: "
        + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
    )
    print("=" * 74)

    modes = (
        ["correctness", "codegen", "graphs", "constraint"]
        if args.mode == "all"
        else [args.mode]
    )
    for m in modes:
        try:
            globals()[f"mode_{m}"](args.device, variants)
        except Exception:  # noqa: BLE001
            print(f"\n!!! mode {m} blew up:")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
