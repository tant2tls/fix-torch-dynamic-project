#!/usr/bin/env python3
"""Run PR #1's two tests, extracted verbatim from the patch, against installed torch.

The tests ship inside the patch as additions to
`test/inductor/test_custom_lowering.py`. PyTorch's own test files are not
executed here (they need a source build and were out of scope for this session),
so the bodies are mirrored 1:1 and driven by unittest directly.

    patch applied  -> 2 pass
    patch reverted -> 2 fail   (this is what makes them regression tests)

    python upstream/dev/run_pr1_tests.py

Exit 0 if the verdicts match the detected patch state, 1 otherwise.
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch  # noqa: E402
from torch._inductor.lowering import (  # noqa: E402
    lowerings,
    register_lowering,
    upsample_nearestnd,
)


HAS_GPU = torch.cuda.is_available()
# Dispatch keys for `Library.impl` are capitalised ("CUDA"); the device string
# passed to torch.randn is lowercase. Conflating them raises
# "RuntimeError: could not parse dispatch key: cuda".
GPU_KEY = "CUDA"
GPU_DEVICE = "cuda"


def patch_state() -> str:
    return "patched" if "div_rn" in inspect.getsource(upsample_nearestnd) else "unpatched"


class TestUpsampleSymbolic(unittest.TestCase):
    """Bodies mirrored verbatim from the patch's additions."""

    def test_upsample_nearestnd_symbolic_output_size(self):
        for exact in (False, True):
            mode = "nearest-exact" if exact else "nearest"
            with torch.library._scoped_library("test_ups_ops", "FRAGMENT") as lib:
                lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")

                def ref(x, h, w, mode=mode):
                    return torch.nn.functional.interpolate(
                        x, size=(int(h), int(w)), mode=mode
                    )

                lib.impl(
                    "ups2d",
                    lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)),
                    "Meta",
                )
                lib.impl("ups2d", ref, "CPU")
                if HAS_GPU:
                    lib.impl("ups2d", ref, GPU_KEY)

                register_lowering(torch.ops.test_ups_ops.ups2d)(
                    lambda x, h, w: upsample_nearestnd(
                        x, [h, w], (None, None), n=2, exact=exact
                    )
                )

                def fn(x, sizes):
                    return torch.ops.test_ups_ops.ups2d(x, sizes[0], sizes[1])

                for device in ["cpu"] + ([GPU_DEVICE] if HAS_GPU else []):
                    for in_hw, out_hw in [
                        ((32, 32), (64, 64)),
                        ((32, 32), (100, 70)),
                        ((32, 32), (17, 17)),
                        ((448, 448), (192, 192)),
                        ((384, 384), (363, 363)),
                    ]:
                        torch._dynamo.reset()
                        x = torch.randn(1, 3, *in_hw, device=device)
                        ref_t = torch.randn(1, 3, *out_hw, device=device)
                        torch._dynamo.maybe_mark_dynamic(ref_t, 2)
                        torch._dynamo.maybe_mark_dynamic(ref_t, 3)
                        self.assertEqual(
                            ref(x, *out_hw).tolist(),
                            torch.compile(fn, dynamic=True)(
                                x, ref_t.shape[-2:]
                            ).tolist(),
                        )

                torch._dynamo.reset()
                lowerings.pop(torch.ops.test_ups_ops.ups2d.default, None)
                lowerings.pop(torch.ops.test_ups_ops.ups2d, None)

    def test_upsample_nearestnd_symbolic_output_size_one_graph(self):
        calls = 0
        real = upsample_nearestnd

        def counting(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real(*args, **kwargs)

        with torch.library._scoped_library("test_ups_ops2", "FRAGMENT") as lib:
            lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")

            def ref(x, h, w):
                return torch.nn.functional.interpolate(
                    x, size=(int(h), int(w)), mode="nearest"
                )

            lib.impl(
                "ups2d",
                lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)),
                "Meta",
            )
            lib.impl("ups2d", ref, "CPU")

            register_lowering(torch.ops.test_ups_ops2.ups2d)(
                lambda x, h, w: counting(x, [h, w], (None, None), n=2)
            )

            def fn(x, sizes):
                return torch.ops.test_ups_ops2.ups2d(x, sizes[0], sizes[1])

            torch._dynamo.reset()
            compiled = torch.compile(fn, dynamic=True)
            x = torch.randn(1, 3, 32, 32)
            for out_hw in [(64, 64), (72, 72), (80, 80), (96, 96), (100, 100)]:
                ref_t = torch.randn(1, 3, *out_hw)
                torch._dynamo.maybe_mark_dynamic(ref_t, 2)
                torch._dynamo.maybe_mark_dynamic(ref_t, 3)
                self.assertEqual(
                    ref(x, *out_hw).tolist(),
                    compiled(x, ref_t.shape[-2:]).tolist(),
                )
            self.assertEqual(calls, 1)

            torch._dynamo.reset()
            lowerings.pop(torch.ops.test_ups_ops2.ups2d.default, None)
            lowerings.pop(torch.ops.test_ups_ops2.ups2d, None)


def main() -> int:
    state = patch_state()
    print("=" * 70)
    print(f"torch {torch.__version__}  gpu={HAS_GPU}  lowering: {state}")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestUpsampleSymbolic)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    n_fail = len(result.failures) + len(result.errors)
    n_pass = result.testsRun - n_fail

    print("-" * 70)
    if state == "patched":
        ok = n_fail == 0
        print(f"expected on patched torch: 2 pass / 0 fail   got {n_pass}/{n_fail}")
    else:
        ok = n_pass == 0
        print(f"expected on unpatched torch: 0 pass / 2 fail  got {n_pass}/{n_fail}")
        print("(a regression test that passes unpatched would pin nothing)")
    print("VERDICT:", "as expected" if ok else "NOT as expected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
