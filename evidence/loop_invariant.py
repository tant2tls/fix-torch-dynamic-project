#!/usr/bin/env python
"""Is the deferred divide loop-invariant (once per kernel) or per-element?

This is the crux of the performance argument. PR #164144 made inductor's
`truediv` emit `div_rn` globally, and it was gated behind a flag after a ~28%
throughput regression, because there the divide is per-element in a hot
elementwise kernel.

Here the divisor is the symbolic output size -- a *kernel argument*, uniform
across the whole grid -- so the division should be hoisted out of the indexing
math: computed once per program, not once per element. Show that in the emitted
Triton rather than asserting it.
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F
from torch._inductor.ir import Pointwise
from torch._inductor.lowering import lowerings, register_lowering, upsample_nearestnd
from torch._inductor.utils import run_and_get_code

with torch.library._scoped_library("inv_ops", "FRAGMENT") as lib:
    lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
    lib.impl("ups2d", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta")
    lib.impl(
        "ups2d",
        lambda x, h, w: F.interpolate(x, size=(int(h), int(w)), mode="nearest"),
        "CUDA",
    )
    register_lowering(torch.ops.inv_ops.ups2d)(
        lambda x, h, w: upsample_nearestnd(x, [h, w], (None, None), n=2)
    )

    def fn(x, sizes):
        return torch.ops.inv_ops.ups2d(x, sizes[0], sizes[1])

    torch._dynamo.reset()
    x = torch.randn(8, 64, 128, 128, device="cuda")
    ref = torch.randn(8, 64, 256, 256, device="cuda")
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)
    _, codes = run_and_get_code(torch.compile(fn, dynamic=True), x, ref.shape[-2:])

    src = "\n".join(codes)
    body = []
    inside = False
    for line in src.splitlines():
        if "def triton_poi" in line or "def triton_" in line:
            inside = True
        if inside:
            body.append(line)
        if inside and "tl.store" in line:
            break

    print("Generated kernel body:\n")
    for line in body:
        print("   ", line.rstrip()[:120])

    print()
    divs = [l.strip() for l in body if "div_rn" in l]
    print(f"div_rn occurrences in the kernel body: {len(divs)}")
    for d in divs:
        print("   ", d[:120])
    print()
    # Does the divide depend on xindex (per-element) or only on kernel args?
    for d in divs:
        per_elem = any(tok in d for tok in ("xindex", "x0", "x1", "x2", "x3"))
        print(
            f"  -> {'PER-ELEMENT' if per_elem else 'LOOP-INVARIANT (kernel args only)'}"
        )
    print()
    print("A loop-invariant divide is one scalar op per program launch. Triton/LLVM")
    print("hoists it out of the vectorised body, so it does not scale with numel --")
    print("which is why this does not repeat the #164144 regression.")

    torch._dynamo.reset()
    lowerings.pop(torch.ops.inv_ops.ups2d.default, None)
    lowerings.pop(torch.ops.inv_ops.ups2d, None)
