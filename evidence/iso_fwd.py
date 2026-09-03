#!/usr/bin/env python
"""Isolate the A100 failure in test_upsample_nearestnd_symbolic_output_size.

Reports, per (mode, device, in_hw, out_hw), whether the compiled result matches
eager bit-exactly -- so we learn whether this is device-specific, shape-specific,
or mode-specific rather than guessing.
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from torch._inductor.lowering import lowerings, register_lowering, upsample_nearestnd

SHAPES = [
    ((32, 32), (64, 64)),
    ((32, 32), (100, 70)),
    ((32, 32), (17, 17)),
    ((448, 448), (192, 192)),
    ((384, 384), (363, 363)),
]

print(f"torch {torch.__version__}  dev={torch.cuda.get_device_name(0)}")
print(f"{'mode':<14}{'dev':<6}{'in':<12}{'out':<12}{'result'}")
print("-" * 62)

fails = []
for exact in (False, True):
    mode = "nearest-exact" if exact else "nearest"
    for device in ["cpu", "cuda"]:
        for in_hw, out_hw in SHAPES:
            name = f"ups_{int(exact)}_{device}_{in_hw[0]}_{out_hw[0]}_{out_hw[1]}"
            with torch.library._scoped_library("iso_ops", "FRAGMENT") as lib:
                lib.define(f"{name}(Tensor x, SymInt h, SymInt w) -> Tensor")

                def ref(x, h, w, mode=mode):
                    return torch.nn.functional.interpolate(
                        x, size=(int(h), int(w)), mode=mode
                    )

                lib.impl(
                    name,
                    lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)),
                    "Meta",
                )
                lib.impl(name, ref, "CPU")
                lib.impl(name, ref, "CUDA")
                op = getattr(torch.ops.iso_ops, name)
                register_lowering(op)(
                    lambda x, h, w, exact=exact: upsample_nearestnd(
                        x, [h, w], (None, None), n=2, exact=exact
                    )
                )

                def fn(x, sizes, op=op):
                    return op(x, sizes[0], sizes[1])

                torch._dynamo.reset()
                x = torch.randn(1, 3, *in_hw, device=device)
                ref_t = torch.randn(1, 3, *out_hw, device=device)
                torch._dynamo.maybe_mark_dynamic(ref_t, 2)
                torch._dynamo.maybe_mark_dynamic(ref_t, 3)
                try:
                    got = torch.compile(fn, dynamic=True)(x, ref_t.shape[-2:])
                    want = ref(x, *out_hw)
                    if torch.equal(got, want):
                        res = "ok (bit-exact)"
                    else:
                        n = (got != want).sum().item()
                        res = f"MISMATCH {n}/{want.numel()}"
                        fails.append((mode, device, in_hw, out_hw, n))
                except Exception as e:
                    res = f"RAISED {type(e).__name__}: {str(e)[:60]}"
                    fails.append((mode, device, in_hw, out_hw, res))
                print(f"{mode:<14}{device:<6}{str(in_hw):<12}{str(out_hw):<12}{res}")
                torch._dynamo.reset()
                lowerings.pop(op.default, None)
                lowerings.pop(op, None)

print("-" * 62)
print(f"{len(fails)} failing configs")
for f in fails:
    print("  ", f)
