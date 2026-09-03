#!/usr/bin/env python
"""Why the same ratio can be right or wrong on ONE GPU (RESULTS_a100.md 19.3).

Issue A's row counts move around for reasons that have nothing to do with the
GPU, and two of them cost real time. Both are about the *emitted divide*:

  1. THE DIVISOR'S FORM DECIDES. Inductor emits `ks0 / 74` when the output size
     is a constant it can see, and `ks0 / ks1` when it cannot. In the first form
     the divide is often exact; in the second BOTH operands are runtime values,
     the hardware reciprocal is used, and it can land 1 ULP off -- changing the
     floored source index. Same ratio, same GPU, different answer.

     What decides which form you get is ordinary Python scoping: a lambda closing
     over a MODULE GLOBAL lets Dynamo fold the size to a literal; closing over a
     FUNCTION LOCAL keeps it symbolic. This script measures both on purpose.

  2. A PROBE MUST REPRODUCE THE EMITTED FORM. Constant-folding the divide inside
     a hand-written Triton kernel measures a path Inductor never generated, and
     reports the wrong ULP direction.

Ratios whose count survives BOTH forms are the ones to put in a bug report: they
do not depend on the reporter's SM, scoping, or compile order.

Usage:
    python issueA_runtime_divide.py --all
    python issueA_runtime_divide.py --ratio 41:82
"""
import argparse
import os
import struct
import subprocess
import sys

# Wrong on A100 and H100, with the conservative (literal-divisor) form.
HEADLINE = [(448, 192), (384, 363), (448, 368)]
# All four share the correctly-rounded scale 0x3f000000 and still disagree.
SAME_SCALE = [(37, 74), (41, 82), (74, 148), (101, 202)]

_G_SIZE = None  # module global, so the literal-divisor form is reachable


def bits(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _rows(t):
    import torch

    return t[0, 0, :, 0].to(torch.int64).tolist()


def _input(i):
    import torch

    return (torch.arange(i, device="cuda", dtype=torch.float32)
            .view(1, 1, i, 1).expand(1, 1, i, 4).contiguous())


def _count_wrong(fn, x):
    import torch

    eager, comp = _rows(fn(x)), _rows(torch.compile(fn, dynamic=True)(x))
    return [(d, a, b) for d, (a, b) in enumerate(zip(eager, comp)) if a != b]


def literal_divisor(i, o):
    """Output size reachable as a module global -> emits `ks0 / <literal>`."""
    global _G_SIZE
    import torch.nn.functional as F

    _G_SIZE = o
    return _count_wrong(lambda t: F.interpolate(t, size=(_G_SIZE, 4), mode="nearest"),
                        _input(i))


def symbolic_divisor(i, o):
    """Output size is a function local -> emits `ks0 / ks1`, both runtime."""
    import torch.nn.functional as F

    return _count_wrong(lambda t: F.interpolate(t, size=(o, 4), mode="nearest"),
                        _input(i))


def emitted_divide_error(i, o):
    """The divide in the form Inductor emits: numerator a runtime i64 arg."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def k(out, ks0, o_lit: tl.constexpr, BLOCK: tl.constexpr):
        xs = tl.arange(0, BLOCK)
        tl.store(out + xs, (ks0 / o_lit).to(tl.float32), mask=xs < 1)

    buf = torch.zeros(1, device="cuda", dtype=torch.float32)
    k[(1,)](buf, i, o, BLOCK=1)
    rt, exact = buf[0].item(), f32(i / o)
    if bits(rt) == bits(exact):
        return rt, exact, "none"
    return rt, exact, "-1 ULP (low)" if rt < exact else "+1 ULP (high)"


def one(i, o):
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/tic_issueA_{i}_{o}"
    import torch

    if not torch.cuda.is_available():
        print("no CUDA: this finding is CUDA-only")
        return 3

    lit = literal_divisor(i, o)
    sym = symbolic_divisor(i, o)
    rt, exact, err = emitted_divide_error(i, o)

    robust = "ROBUST" if lit else ("scope-dependent" if sym else "clean")
    print(f"{i:>4} -> {o:<4} | literal divisor {len(lit):>4}/{o:<4} "
          f"| symbolic divisor {len(sym):>4}/{o:<4} "
          f"| ks0/{o} = {bits(rt)} vs {bits(exact)} ({err}) | {robust}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", help="i:o — runs in this process")
    ap.add_argument("--all", action="store_true",
                    help="re-exec once per ratio, for clean isolation")
    a = ap.parse_args()

    if a.ratio:
        i, o = (int(v) for v in a.ratio.split(":"))
        return one(i, o)

    if a.all:
        print("One process per ratio. 'literal' vs 'symbolic' is the DIVISOR's form,")
        print("set by whether the output size is a module global or a closure local.\n")
        print("  headline ratios — wrong even with a literal divisor:")
        for i, o in HEADLINE:
            subprocess.run([sys.executable, __file__, "--ratio", f"{i}:{o}"])
        print("\n  all four share the correctly-rounded scale 0x3f000000:")
        for i, o in SAME_SCALE:
            subprocess.run([sys.executable, __file__, "--ratio", f"{i}:{o}"])
        print("\nRead: a ratio wrong under BOTH forms is safe to headline in a bug")
        print("report. One wrong only under 'symbolic' depends on how the caller")
        print("happened to write the size, and reads as non-reproducible.")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
