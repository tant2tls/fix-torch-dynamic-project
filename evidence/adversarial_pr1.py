#!/usr/bin/env python
"""Adversarial checks on PR1's forward fix, beyond the shipped tests.

Each of these is a way the patch could be wrong that the six tests would not
catch. Written to try to BREAK the fix, not to confirm it.

1. Mixed dims: one spatial dim symbolic, the other concrete. The `None` sentinel
   is per-dim, so a bug in the zip() would show up only here.
2. `scales_x` supplied for one dim and not the other, with a symbolic size --
   `1.0 / scale` must still win over the deferred path.
3. 1-D and 3-D with symbolic sizes (n != 2).
4. Downsampling as well as upsampling (i > o and i < o).
5. Ratios where the fp32 scale is exactly representable vs not.
6. Degenerate sizes: output 1, input 1, and o > i*2.
7. Non-contiguous / expanded input.
8. float16 and bfloat16 inputs (scale math stays fp32).
9. The 'both dims symbolic and EQUAL' case, where a shared symbol could be
   incorrectly reused across dims.
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import sympy
import torch
import torch.nn.functional as F
from torch._inductor.lowering import lowerings, register_lowering, upsample_nearestnd

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
fails = []
n_ok = 0


def check(label, device, build, ref_fn, exact=False, n=2):
    """build(lib_op) -> compiled callable; ref_fn() -> eager tensor"""
    global n_ok
    try:
        got, want = build()
        if got.shape != want.shape:
            fails.append((label, device, f"shape {got.shape} != {want.shape}"))
            print(f"  FAIL {label:<46} shape {got.shape} != {want.shape}")
            return
        if torch.equal(got, want):
            n_ok += 1
            print(f"  ok   {label:<46} bit-exact")
        else:
            d = (got != want).sum().item()
            fails.append((label, device, f"{d}/{want.numel()} differ"))
            print(f"  FAIL {label:<46} {d}/{want.numel()} differ")
    except Exception as e:
        fails.append((label, device, f"{type(e).__name__}: {str(e)[:70]}"))
        print(f"  RAISE {label:<45} {type(e).__name__}: {str(e)[:70]}")


def run_case(name, device, in_shape, out_sizes, scales, n, exact, dtype=torch.float32,
             dyn_dims=None, noncontig=False):
    """out_sizes: list of ints; dyn_dims: which spatial dims to mark dynamic."""
    opname = f"adv_{name}"
    with torch.library._scoped_library("adv_ops", "FRAGMENT") as lib:
        sig_syms = ", ".join(f"SymInt s{i}" for i in range(n))
        lib.define(f"{opname}(Tensor x, {sig_syms}) -> Tensor")

        def meta(x, *sz):
            return x.new_empty((*x.shape[: -n], *sz))

        def eager(x, *sz):
            mode = "nearest-exact" if exact else "nearest"
            return F.interpolate(x, size=tuple(int(s) for s in sz), mode=mode)

        lib.impl(opname, meta, "Meta")
        lib.impl(opname, eager, "CPU")
        lib.impl(opname, eager, "CUDA")
        op = getattr(torch.ops.adv_ops, opname)
        register_lowering(op)(
            lambda x, *sz: upsample_nearestnd(x, list(sz), scales, n=n, exact=exact)
        )

        torch._dynamo.reset()
        torch.manual_seed(0)
        x = torch.randn(*in_shape, device=device, dtype=dtype)
        if noncontig:
            x = x.transpose(-1, -2)
            in_sp = list(x.shape[-n:])
        # a tensor whose shape carries the symbolic output sizes
        ref = torch.randn(1, 1, *out_sizes, device=device)
        if dyn_dims is None:
            dyn_dims = range(n)
        for d in dyn_dims:
            torch._dynamo.maybe_mark_dynamic(ref, 2 + d)
        sizes = list(ref.shape[2:])

        def build():
            g = torch.compile(lambda t, s: op(t, *s), dynamic=True)
            return g(x, sizes), eager(x, *sizes)

        check(name, device, build, None, exact=exact, n=n)
        torch._dynamo.reset()
        lowerings.pop(op.default, None)
        lowerings.pop(op, None)


print(f"torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"gpu   {torch.cuda.get_device_name(0)}")

for device in DEVS:
    print(f"\n===== {device} =====")
    N = (None, None)
    # 1. both symbolic, plain
    run_case("both_sym_up", device, (1, 3, 32, 32), [64, 70], N, 2, False)
    run_case("both_sym_down", device, (1, 3, 64, 64), [32, 21], N, 2, False)
    # 2. mixed: only dim 0 dynamic (dim 1 stays a concrete int)
    run_case("mixed_dyn_dim0", device, (1, 3, 32, 32), [64, 70], N, 2, False, dyn_dims=[0])
    run_case("mixed_dyn_dim1", device, (1, 3, 32, 32), [64, 70], N, 2, False, dyn_dims=[1])
    # 3. explicit scale for one dim, symbolic size for the other
    run_case("scale_dim0_only", device, (1, 3, 32, 32), [64, 70], (2.0, None), 2, False)
    run_case("scale_dim1_only", device, (1, 3, 32, 32), [64, 96], (None, 3.0), 2, False)
    run_case("scale_both", device, (1, 3, 32, 32), [64, 64], (2.0, 2.0), 2, False)
    # 4. nearest-exact (the 0.5 add happens before the deferred multiply)
    run_case("exact_both_sym", device, (1, 3, 32, 32), [64, 70], N, 2, True)
    run_case("exact_ulp_448_192", device, (1, 3, 448, 448), [192, 192], N, 2, True)
    # 5. ULP-sensitive ratios
    run_case("ulp_448_192", device, (1, 3, 448, 448), [192, 192], N, 2, False)
    run_case("ulp_384_363", device, (1, 3, 384, 384), [363, 363], N, 2, False)
    run_case("ulp_37_74", device, (1, 3, 37, 37), [74, 74], N, 2, False)
    # 6. degenerate
    run_case("out_one", device, (1, 3, 32, 32), [1, 1], N, 2, False)
    run_case("in_one", device, (1, 3, 1, 1), [8, 8], N, 2, False)
    run_case("equal_sizes", device, (1, 3, 32, 32), [32, 32], N, 2, False)
    run_case("big_upscale", device, (1, 3, 8, 8), [64, 64], N, 2, False)
    # 7. n != 2
    run_case("1d_sym", device, (1, 3, 40), [90], (None,), 1, False)
    run_case("3d_sym", device, (1, 2, 8, 10, 12), [16, 15, 25], (None, None, None), 3, False)
    # 8. dtypes
    run_case("fp16", device, (1, 3, 32, 32), [64, 70], N, 2, False, dtype=torch.float16)
    run_case("bf16", device, (1, 3, 32, 32), [64, 70], N, 2, False, dtype=torch.bfloat16)
    # 9. non-contiguous input
    run_case("noncontig", device, (1, 3, 32, 40), [64, 70], N, 2, False, noncontig=True)

print(f"\n{'=' * 60}")
print(f"{n_ok} bit-exact, {len(fails)} problems")
for f in fails:
    print("   ", f)
