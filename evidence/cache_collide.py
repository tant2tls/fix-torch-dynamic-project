#!/usr/bin/env python
"""Is the suite failure a FX-graph-cache collision on a reused op name?

The test registers a lowering for the SAME op name `test_ups_ops::ups2d` twice --
once with exact=False, once with exact=True. The op schema/name is identical, so
if Inductor's cache key does not see the `exact` flag (it lives in the lowering,
not the graph), the second iteration can reuse the first's kernel and silently
produce nearest results where nearest-exact was asked for.

Run A: same op name for both exact values  (mirrors the test)
Run B: distinct op name per exact value     (mirrors iso_fwd.py, which passed)
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from torch._inductor.lowering import lowerings, register_lowering, upsample_nearestnd

IN_HW, OUT_HW = (32, 32), (100, 70)


def run(same_name: bool, cache: bool):
    torch._inductor.config.fx_graph_cache = cache
    results = {}
    for exact in (False, True):
        mode = "nearest-exact" if exact else "nearest"
        opname = "ups2d" if same_name else f"ups2d_{int(exact)}"
        with torch.library._scoped_library("collide_ops", "FRAGMENT") as lib:
            lib.define(f"{opname}(Tensor x, SymInt h, SymInt w) -> Tensor")

            def ref(x, h, w, mode=mode):
                return torch.nn.functional.interpolate(
                    x, size=(int(h), int(w)), mode=mode
                )

            lib.impl(
                opname,
                lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)),
                "Meta",
            )
            lib.impl(opname, ref, "CPU")
            lib.impl(opname, ref, "CUDA")
            op = getattr(torch.ops.collide_ops, opname)
            register_lowering(op)(
                lambda x, h, w, exact=exact: upsample_nearestnd(
                    x, [h, w], (None, None), n=2, exact=exact
                )
            )

            def fn(x, sizes, op=op):
                return op(x, sizes[0], sizes[1])

            torch._dynamo.reset()
            torch.manual_seed(0)
            x = torch.randn(1, 3, *IN_HW, device="cuda")
            ref_t = torch.randn(1, 3, *OUT_HW, device="cuda")
            torch._dynamo.maybe_mark_dynamic(ref_t, 2)
            torch._dynamo.maybe_mark_dynamic(ref_t, 3)
            got = torch.compile(fn, dynamic=True)(x, ref_t.shape[-2:])
            want = ref(x, *OUT_HW)
            n = (got != want).sum().item()
            results[mode] = (n, want.numel(), got.clone())
            torch._dynamo.reset()
            lowerings.pop(op.default, None)
            lowerings.pop(op, None)
    return results


for label, same, cache in [
    ("A same-op-name, cache ON ", True, True),
    ("B distinct-names, cache ON", False, True),
    ("C same-op-name, cache OFF", True, False),
]:
    r = run(same, cache)
    line = []
    for mode, (n, tot, _) in r.items():
        line.append(f"{mode}: {'ok' if n == 0 else f'MISMATCH {n}/{tot}'}")
    # did the two modes produce byte-identical output? (they must NOT)
    a = r["nearest"][2]
    b = r["nearest-exact"][2]
    same_out = torch.equal(a, b)
    print(f"{label} | {'  '.join(line)} | outputs identical across modes: {same_out}")
print()
print("If A mismatches and B does not, the test's reused op name is the bug,")
print("not the lowering. 'outputs identical across modes: True' proves reuse.")
