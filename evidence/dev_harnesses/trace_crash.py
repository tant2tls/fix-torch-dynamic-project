"""Reproduce the upsample_nearestnd symbolic-scale crash and print the FULL traceback.

Purpose: establish exactly what a developer sees today, so we can judge how much
a better diagnostic actually buys. Run under any torch >= 2.4.
"""
import os, traceback
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch
from torch._inductor.lowering import register_lowering, upsample_nearestnd

DEV = os.environ.get("DEV", "cpu")

with torch.library._scoped_library("trace_dev", "FRAGMENT") as lib:
    lib.define("ups(Tensor x, SymInt h, SymInt w) -> Tensor")

    def meta(x, h, w):
        return x.new_empty((x.shape[0], x.shape[1], h, w))

    def ref(x, h, w):
        return torch.nn.functional.interpolate(x, size=(int(h), int(w)), mode="nearest")

    lib._register_fake("ups", meta)
    lib.impl("ups", ref, "CPU")
    lib.impl("ups", ref, "CUDA")

    register_lowering(torch.ops.trace_dev.ups)(
        lambda x, h, w: upsample_nearestnd(x, [h, w], (None, None), n=2)
    )

    def fn(x, sizes):
        return torch.ops.trace_dev.ups(x, sizes[0], sizes[1])

    x = torch.randn(1, 3, 32, 32, device=DEV)
    ref_t = torch.randn(1, 3, 64, 64, device=DEV)
    torch._dynamo.maybe_mark_dynamic(ref_t, 2)
    torch._dynamo.maybe_mark_dynamic(ref_t, 3)
    try:
        torch.compile(fn, dynamic=True)(x, ref_t.shape[-2:])
        print("NO CRASH -- lowering did not fail")
    except Exception:
        traceback.print_exc()
