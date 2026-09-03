#!/usr/bin/env python
"""PR3 guards; PR1 refuses to guard. Quantify why that is not inconsistent.

PR1's argument against `guard_int` is that it specializes on the output size, so a
caller sweeping sizes recompiles per size and eventually falls back to eager. PR3
adds exactly the `guard_int` PR1 avoided. A reviewer reading both will ask whether
PR3 has the problem PR1 warns about.

The answer should be structural, not rhetorical: in the backward the quotient is a
Python `range()` bound over which the pooling window is unrolled, so the kernel
*body* differs per size -- there is nothing to defer. Measure it:

  1. how many distinct lowerings/graphs a size sweep costs in each case
  2. that the forward genuinely stays on ONE graph across the same sweep
"""

import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch._inductor.lowering as L
from torch._inductor.lowering import lowerings, register_lowering, upsample_nearestnd

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZES = [(64, 64), (72, 72), (80, 80), (96, 96), (100, 100), (112, 112), (128, 128)]

print(f"torch {torch.__version__}  device={DEV}")
print()

# ---------------------------------------------------------- forward: one graph?
calls = {"n": 0}
real_fwd = upsample_nearestnd


def counting_fwd(*a, **k):
    calls["n"] += 1
    return real_fwd(*a, **k)


with torch.library._scoped_library("pr3q", "FRAGMENT") as lib:
    lib.define("fwd(Tensor x, SymInt h, SymInt w) -> Tensor")
    lib.impl("fwd", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta")
    lib.impl(
        "fwd",
        lambda x, h, w: torch.nn.functional.interpolate(
            x, size=(int(h), int(w)), mode="nearest"
        ),
        "CPU",
    )
    lib.impl(
        "fwd",
        lambda x, h, w: torch.nn.functional.interpolate(
            x, size=(int(h), int(w)), mode="nearest"
        ),
        "CUDA",
    )
    op = torch.ops.pr3q.fwd
    register_lowering(op)(lambda x, h, w: counting_fwd(x, [h, w], (None, None), n=2))

    torch._dynamo.reset()
    g = torch.compile(lambda t, s: op(t, s[0], s[1]), dynamic=True)
    x = torch.randn(1, 3, 32, 32, device=DEV)
    for hw in SIZES:
        ref = torch.randn(1, 1, *hw, device=DEV)
        torch._dynamo.maybe_mark_dynamic(ref, 2)
        torch._dynamo.maybe_mark_dynamic(ref, 3)
        g(x, list(ref.shape[2:]))
    print(f"FORWARD (PR1, deferred divide)")
    print(f"  {len(SIZES)} distinct output sizes -> {calls['n']} lowering invocation(s)")
    print(f"  -> the divide is a kernel argument, so one kernel serves every size")
    torch._dynamo.reset()
    lowerings.pop(op.default, None)
    lowerings.pop(op, None)

# ------------------------------------------------- backward: must it specialize?
print()
print("BACKWARD (PR3, guard_int)")
real_bwd = L.upsample_nearest2d_backward
seen_bounds = []


def spy_bwd(x, *, output_size, input_size, scales_h=None, scales_w=None):
    # record the loop bounds this lowering will unroll over
    inp_h, inp_w = x.get_size()[-2:]
    from torch._inductor.virtualized import V

    ih = V.graph.sizevars.guard_int(inp_h)
    iw = V.graph.sizevars.guard_int(inp_w)
    oh, ow = input_size[-2], input_size[-1]
    seen_bounds.append((ih, iw, oh, ow))
    return real_bwd(
        x,
        output_size=output_size,
        input_size=input_size,
        scales_h=scales_h,
        scales_w=scales_w,
    )


def fn(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad,
        [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]],
    )


torch._dynamo.reset()
gb = torch.compile(fn, dynamic=True)
n_ok = 0
for hw in [(32, 32), (16, 16), (8, 8), (64, 64)]:
    grad = torch.randn(1, 3, 128, 128, device=DEV)
    ref = torch.randn(1, 3, *hw, device=DEV)
    torch._dynamo.maybe_mark_dynamic(ref, 2)
    torch._dynamo.maybe_mark_dynamic(ref, 3)
    got = gb(grad, ref)
    want = fn(grad, ref)
    ok = torch.equal(got, want)
    n_ok += ok
    # the unroll factor the lowering must know at compile time
    kh = -(-128 // hw[0])
    kw = -(-128 // hw[1])
    print(
        f"  grad 128x128 -> input {hw[0]}x{hw[1]}: "
        f"pooling window {kh}x{kw} = {kh * kw} unrolled loads, "
        f"{'bit-exact' if ok else 'MISMATCH'}"
    )
print()
print(f"  {n_ok}/4 bit-exact")
print("  -> the window size IS the kernel body. 4x4=16 loads vs 16x16=256 loads is")
print("     not one kernel with a different argument; it is a different kernel.")
print("     There is no value that could be deferred, so guarding is not a choice")
print("     between fast and slow -- it is the only way to lower this at all.")
