#!/usr/bin/env python
"""What does the deferred `ops.div_rn` cost?

Context: PR #164144 made inductor's `truediv` emit `div_rn` for eager parity, was
merged, then gated behind TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING after a ~28%
throughput regression on B200 (#164301). So "you added a div_rn" is a question a
reviewer will ask, and it needs a number, not an argument.

The claim to test: here the divide is loop-invariant -- one scalar division per
kernel launch, not one per element -- because the divisor is a kernel *argument*
(the symbolic output size), not a per-element value. So the cost should be
independent of tensor size and vanish against the memory traffic.

Measures, on the same A100:
  1. static  (scale folded to a constant at compile time)     <- baseline
  2. dynamic (deferred div_rn, this PR)                        <- the cost
  3. dynamic-but-guarded (guard_int, the alternative fix)      <- the comparison
and reports per-call latency and achieved bandwidth.
"""
import os
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F
from torch._inductor.lowering import lowerings, register_lowering, upsample_nearestnd

DEV = "cuda"
ITERS = 200
WARMUP = 20


def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6  # microseconds


print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
print(f"{ITERS} iters after {WARMUP} warmup, per-call microseconds\n")

SIZES = [
    (1, 3, 64, 64, 128, 128),
    (1, 3, 256, 256, 512, 512),
    (8, 64, 128, 128, 256, 256),
    (4, 128, 256, 256, 512, 512),
]

print(f"{'shape':<26}{'out':<12}{'static us':<12}{'dynamic us':<12}{'ratio':<9}{'GB/s dyn':<10}")
print("-" * 82)
for n, c, h, w, oh, ow in SIZES:
    x = torch.randn(n, c, h, w, device=DEV)

    # 1. static: scale is a compile-time constant
    torch._dynamo.reset()
    f_static = torch.compile(
        lambda t: F.interpolate(t, size=(oh, ow), mode="nearest"), dynamic=False
    )
    f_static(x)
    t_static = bench(f_static, x)

    # 2. dynamic: sizes symbolic -> the decomposition divides in-kernel
    torch._dynamo.reset()
    xd = x.clone()
    torch._dynamo.mark_dynamic(xd, 2)
    torch._dynamo.mark_dynamic(xd, 3)
    f_dyn = torch.compile(
        lambda t: F.interpolate(t, size=(oh, ow), mode="nearest"), dynamic=True
    )
    f_dyn(xd)
    t_dyn = bench(f_dyn, xd)

    nbytes = x.numel() * 4 + n * c * oh * ow * 4
    gbs = nbytes / (t_dyn * 1e-6) / 1e9
    shape = f"{n}x{c}x{h}x{w}"
    print(
        f"{shape:<26}{f'{oh}x{ow}':<12}{t_static:<12.1f}{t_dyn:<12.1f}"
        f"{t_dyn / t_static:<9.3f}{gbs:<10.0f}"
    )

print()
print("The dynamic column includes the deferred divide AND the general cost of")
print("symbolic-shape codegen (extra kernel args, no constant folding of strides).")
print("To separate them, the next block times the SAME dynamic kernel with the")
print("divide done by truediv instead of div_rn.")
print()

# 3. isolate div_rn from dynamic-shape overhead: same lowering, two divides.
import sympy
from torch._inductor.virtualized import ops


def make_lowering(use_div_rn):
    def low(x, h, w):
        x.realize_hint()
        x_loader = x.make_loader()
        i_sizes = [
            torch._inductor.virtualized.V.graph.sizevars.guard_int(s)
            for s in x.get_size()[-2:]
        ]
        batch = x.get_size()[:-2]
        o_sizes = [h, w]

        def scale_fn(idx, o_size, size):
            f = ops.index_expr(idx, torch.float32)
            num = ops.constant(size, torch.float32)
            den = ops.index_expr(o_size, torch.float32)
            inv = ops.div_rn(num, den) if use_div_rn else ops.truediv(num, den)
            f = ops.mul(f, inv)
            f = ops.to_dtype(f, torch.int32)
            return ops.indirect_indexing(f, size, check=False)

        def fn(index):
            sp = index[-2:]
            b = index[:-2]
            return x_loader(
                [*b, *[scale_fn(i, o, s) for i, o, s in zip(sp, o_sizes, i_sizes)]]
            )

        from torch._inductor.ir import Pointwise

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=fn,
            ranges=[*batch, h, w],
        )

    return low


print(f"{'shape':<26}{'div_rn us':<12}{'truediv us':<12}{'div_rn/truediv':<16}")
print("-" * 66)
for n, c, h, w, oh, ow in SIZES:
    times = {}
    for use_rn in (True, False):
        nm = f"bmark_{int(use_rn)}_{n}_{c}_{h}_{w}"
        with torch.library._scoped_library("perf_ops", "FRAGMENT") as lib:
            lib.define(f"{nm}(Tensor x, SymInt h, SymInt w) -> Tensor")
            lib.impl(
                nm, lambda t, a, b: t.new_empty((t.shape[0], t.shape[1], a, b)), "Meta"
            )
            lib.impl(
                nm,
                lambda t, a, b: F.interpolate(t, size=(int(a), int(b)), mode="nearest"),
                "CUDA",
            )
            op = getattr(torch.ops.perf_ops, nm)
            register_lowering(op)(make_lowering(use_rn))
            torch._dynamo.reset()
            x = torch.randn(n, c, h, w, device=DEV)
            ref = torch.randn(n, c, oh, ow, device=DEV)
            torch._dynamo.mark_dynamic(ref, 2)
            torch._dynamo.mark_dynamic(ref, 3)
            g = torch.compile(lambda t, s: op(t, s[0], s[1]), dynamic=True)
            g(x, ref.shape[-2:])
            times[use_rn] = bench(g, x, ref.shape[-2:])
            torch._dynamo.reset()
            lowerings.pop(op.default, None)
            lowerings.pop(op, None)
    shape = f"{n}x{c}x{h}x{w}"
    print(
        f"{shape:<26}{times[True]:<12.1f}{times[False]:<12.1f}"
        f"{times[True] / times[False]:<16.4f}"
    )
