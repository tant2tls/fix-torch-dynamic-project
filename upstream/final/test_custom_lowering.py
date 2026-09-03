# Owner(s): ["module: inductor"]

from functools import partial
from unittest import skipIf

import sympy

import torch
from torch._inductor import config
from torch._inductor.ir import Pointwise
from torch._inductor.lowering import (
    lowerings,
    make_fallback,
    make_pointwise,
    register_lowering,
    upsample_nearestnd,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import ops
from torch.testing._internal.common_utils import skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


# These tests check issues for lowerings that aren't in the main pytorch repo
class TestCustomLowering(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_inductor_ops = torch.library.Library(  # noqa: SCOPED_LIBRARY
            "test_inductor_ops", "DEF"
        )
        cls.device_list = ["Meta", "CUDA", "XPU"]
        for device in cls.device_list:
            setattr(
                cls,
                "impl_" + device.lower(),
                torch.library.Library(  # noqa: SCOPED_LIBRARY
                    "test_inductor_ops", "IMPL", device
                ),
            )
        cls._register_jagged_to_padded_dense()
        cls._register_asm_op()

    @classmethod
    def tearDown(cls):
        super().tearDownClass()

    @classmethod
    def _register_jagged_to_padded_dense(cls):
        # Approximation of fbgemm.jagged_to_padded_dense_forward
        cls.test_inductor_ops.define(
            "jagged_to_padded_dense(Tensor input, Tensor offsets, SymInt max_seq_len, Scalar pad_value) -> Tensor"
        )

        def j2pd_meta(inp, offsets, max_seq_len, pad_value):
            return torch.empty(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                device=inp.device,
                dtype=inp.dtype,
            )

        def j2pd_gpu(inp, offsets, max_seq_len, pad_value):
            res = torch.full(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                pad_value,
                device=inp.device,
                dtype=inp.dtype,
            )
            for b in range(offsets.shape[0] - 1):
                for r in range(offsets[b + 1] - offsets[b]):
                    res[b][r] = inp[offsets[b] + r]
            return res

        def j2pd_lowering(inp, offsets, max_seq_len, pad_value):
            offsets_loader = offsets.make_loader()
            inp_loader = inp.make_loader()
            jagged_len = inp.get_size()[0]
            offsets_dtype = offsets.get_dtype()

            def inner_fn(index):
                batch_idx, seq_idx, emb_idx = index

                begin_idx = ops.indirect_indexing(
                    offsets_loader([batch_idx]),
                    jagged_len + 1,
                )
                end_idx = offsets_loader([batch_idx + 1])
                jagged_idx = begin_idx + seq_idx

                return ops.masked(
                    ops.lt(
                        ops.index_expr(jagged_idx, offsets_dtype),
                        end_idx,
                    ),
                    lambda: inp_loader([jagged_idx, emb_idx]),
                    pad_value,
                )

            return Pointwise.create(
                device=inp.get_device(),
                dtype=inp.get_dtype(),
                inner_fn=inner_fn,
                ranges=[offsets.get_size()[0] - 1, max_seq_len, inp.get_size()[1]],
            )

        register_lowering(
            torch.ops.test_inductor_ops.jagged_to_padded_dense, type_promotion_kind=None
        )(j2pd_lowering)

        cls.impl_meta.impl("jagged_to_padded_dense", j2pd_meta)
        cls.impl_cuda.impl("jagged_to_padded_dense", j2pd_gpu)
        cls.impl_xpu.impl("jagged_to_padded_dense", j2pd_gpu)

    @classmethod
    def _register_asm_op(cls):
        # Approximation of fbgemm.jagged_to_padded_dense_forward
        cls.test_inductor_ops.define("tanh_approx(Tensor input) -> Tensor")

        def tanh_approx_meta(inp):
            return torch.tanh(inp)

        cls.impl_meta.impl("tanh_approx", tanh_approx_meta)

        def tanh_approx_lowering(inp):
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(
            torch.ops.test_inductor_ops.tanh_approx, type_promotion_kind=None
        )(tanh_approx_lowering)

        cls.test_inductor_ops.define("add_custom(Tensor a, Tensor b) -> Tensor")

        def add_custom(a, b):
            return a + b

        cls.impl_meta.impl("add_custom", add_custom)

        def add_custom_lowering(a, b):
            if torch.version.hip:
                # ROCm GCN assembly
                fn = partial(
                    ops.inline_asm_elementwise,
                    asm="v_add_f32 $0, $1, $2",
                    constraints="=v, v, v",
                )
            else:
                fn = partial(ops.inline_asm_elementwise, asm="add.f32 $0, $1, $2;")
            return make_pointwise(fn)(a, b)

        register_lowering(
            torch.ops.test_inductor_ops.add_custom, type_promotion_kind=None
        )(add_custom_lowering)

    def test_register_lowering_custom_dict(self):
        custom_lowering_dict = {}

        from torch._inductor.lowering import register_lowering

        @torch.library.custom_op("helion_test::foo", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x

        @register_lowering(
            torch.ops.helion_test.foo, lowering_dict=custom_lowering_dict
        )
        def foo_lowering(x):
            return x

        if torch.ops.helion_test.foo not in custom_lowering_dict:
            raise AssertionError
        if torch.ops.helion_test.foo in torch._inductor.lowering.lowerings:
            raise AssertionError

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_jagged_to_padded_dense_sanity_cuda(self):
        def fn(inp, offsets, max_seq_len):
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        inp = torch.rand((9, 96), device=GPU_TYPE)
        offsets = torch.tensor([0, 2, 5, 9], dtype=torch.int32, device=GPU_TYPE)
        max_seq_len = 4

        res = fn(inp, offsets, max_seq_len)
        self.assertEqual(inp[0], res[0][0])
        self.assertEqual(inp[1], res[0][1])
        self.assertEqual(inp[2], res[1][0])
        self.assertEqual(inp[3], res[1][1])
        self.assertEqual(inp[5], res[2][0])
        self.assertEqual(inp[8], res[2][3])

        fn_opt = torch.compile(fn)

        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_jagged_to_padded_dense_zero_size(self):
        # Previously, the masking was being completely stripped for the
        # masked load of the input value. That would lead to an IMA
        # because cuda was trying to read index 0 of a zero-size tensor.
        def fn(inp, offsets, max_seq_len):
            inp = torch.bmm(inp, torch.ones((1, 96, 1), device=GPU_TYPE)).view((0, 1))
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        inp = torch.rand((1, 0, 96), device=GPU_TYPE)
        offsets = torch.zeros(1025, device=GPU_TYPE, dtype=torch.int32)
        max_seq_len = 20

        fn_opt = torch.compile(fn)

        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_tanh_approx(self):
        def fn(inp):
            return torch.ops.test_inductor_ops.tanh_approx(inp)

        inp = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        a = torch.tanh(inp)
        b = fn_opt(inp)
        self.assertEqual(a, b)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_reused_inline_asm_realized(self):
        def fn(inp):
            y = torch.ops.test_inductor_ops.tanh_approx(inp)
            return y.sum(dim=0), y.sum(dim=1)

        inp = torch.randn(32, 64, device=GPU_TYPE)
        expected = (torch.tanh(inp).sum(dim=0), torch.tanh(inp).sum(dim=1))
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), inp)

        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual("\n".join(code).count("tanh.approx.f32"), 1)

    @requires_gpu()
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_multi_inp_asm(self):
        def fn(a, b):
            return torch.ops.test_inductor_ops.add_custom(a, b)

        a = torch.randn(32, device=GPU_TYPE)
        b = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        out1 = a + b
        out2 = fn_opt(a, b)
        self.assertEqual(out1, out2)

    @config.patch(joint_graph_constant_folding=False)
    def test_constant_creation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor(1)

        make_fallback(torch.ops.aten.lift_fresh_copy.default)
        self.assertTrue(
            torch.allclose(torch.compile(M())(torch.ones(3)), torch.ones(3) + 1)
        )

    def test_upsample_nearestnd_symbolic_output_size(self):
        # `upsample_nearestnd` divides the (guarded) input sizes by the output
        # sizes and passes the quotient to `ops.constant`, which requires a
        # concrete value. With a symbolic output size and no explicit scale the
        # quotient stayed symbolic and lowering raised
        # "NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>".
        #
        # The ATen upsample ops are decomposed (arange/mul/_unsafe_index) before
        # Inductor lowers them, so they never reach this lowering; drive it
        # directly, as the tests above do.
        for exact in (False, True):
            mode = "nearest-exact" if exact else "nearest"
            # `exact` is captured by the lowering, not visible in the graph, so
            # the FX graph cache would key both iterations the same and the
            # second would reuse the first's kernel. Vary the op name.
            opname = f"ups2d_exact{int(exact)}"
            with torch.library._scoped_library("test_ups_ops", "FRAGMENT") as lib:
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
                if HAS_GPU:
                    # Dispatch keys are capitalised; GPU_TYPE is the lowercase
                    # device string used for tensor construction below.
                    lib.impl(opname, ref, GPU_TYPE.upper())

                op = getattr(torch.ops.test_ups_ops, opname)
                register_lowering(op)(
                    lambda x, h, w: upsample_nearestnd(
                        x, [h, w], (None, None), n=2, exact=exact
                    )
                )

                def fn(x, sizes, op=op):
                    return op(x, sizes[0], sizes[1])

                for device in ["cpu"] + ([GPU_TYPE] if HAS_GPU else []):
                    # 448 -> 192 and 384 -> 363 are ratios where a scale that is
                    # one ulp off changes the floored index, so they pin the
                    # division to a correctly-rounded one (ops.div_rn).
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
                            ref(x, *out_hw),
                            torch.compile(fn, dynamic=True)(x, ref_t.shape[-2:]),
                            atol=0,
                            rtol=0,
                        )

                torch._dynamo.reset()
                lowerings.pop(op.default, None)
                lowerings.pop(op, None)

    def test_upsample_nearestnd_concrete_sympy_output_size(self):
        # A concrete `sympy` output size (e.g. `sympy.Integer`, from a size a
        # guard already specialized) is a `sympy.Expr` but is not deferred, so
        # `scale_fn` must still fold it. Reading it as a deferred output size
        # divides by an already-inverted scale, which does not crash -- it
        # silently indexes far outside the input -- so pin it exactly.
        for i, size_type in enumerate((int, sympy.Integer)):
            # Both iterations build the same graph, so they must not share an op
            # name or the second would hit the first's cached kernel and pass
            # without ever lowering the sympy.Integer sizes.
            opname = f"ups2d_t{i}"
            with torch.library._scoped_library("test_ups_ops3", "FRAGMENT") as lib:
                lib.define(f"{opname}(Tensor x) -> Tensor")

                def ref(x):
                    return torch.nn.functional.interpolate(
                        x, size=(64, 64), mode="nearest"
                    )

                lib.impl(
                    opname,
                    lambda x: x.new_empty((x.shape[0], x.shape[1], 64, 64)),
                    "Meta",
                )
                lib.impl(opname, ref, "CPU")
                if HAS_GPU:
                    lib.impl(opname, ref, GPU_TYPE.upper())

                op = getattr(torch.ops.test_ups_ops3, opname)
                register_lowering(op)(
                    lambda x: upsample_nearestnd(
                        x, [size_type(64), size_type(64)], (None, None), n=2
                    )
                )

                for device in ["cpu"] + ([GPU_TYPE] if HAS_GPU else []):
                    torch._dynamo.reset()
                    x = torch.randn(1, 3, 32, 32, device=device)
                    self.assertEqual(
                        ref(x),
                        torch.compile(op)(x),
                        atol=0,
                        rtol=0,
                    )

                torch._dynamo.reset()
                lowerings.pop(op.default, None)
                lowerings.pop(op, None)

    def test_upsample_nearestnd_symbolic_output_size_one_graph(self):
        # The division is deferred to the kernel rather than guarded, so a
        # caller sweeping output sizes stays on one graph. Guarding the divisor
        # instead would add a specialization per size and, past
        # torch._dynamo.config.recompile_limit, fall back to eager.
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
                    ref(x, *out_hw), compiled(x, ref_t.shape[-2:]), atol=0, rtol=0
                )
            self.assertEqual(calls, 1)

            torch._dynamo.reset()
            lowerings.pop(torch.ops.test_ups_ops2.ups2d.default, None)
            lowerings.pop(torch.ops.test_ups_ops2.ups2d, None)

    def test_ops_constant_rejects_symbolic_value(self):
        # ops.constant takes a concrete leaf value; a symbolic one cannot be
        # codegened and previously failed much further downstream, with nothing
        # naming the op or the fix.
        s = sympy.Symbol("s0", integer=True, positive=True)
        with self.assertRaisesRegex(TypeError, "requires a concrete value"):
            ops.constant(32 * s, torch.float32)
        # The message should point at the alternative, not just the symptom.
        with self.assertRaisesRegex(TypeError, "ops.index_expr"):
            ops.constant(32 * s, torch.float32)

    def test_ops_constant_accepts_concrete_values(self):
        # sympy numbers are sympy.Expr but are concrete, so they must pass.
        def fn(x):
            return (
                torch.full_like(x, 3.5)
                + torch.add(x, x, alpha=2.5)
                + torch.addcmul(x, x, x, value=0.7)
            )

        x = torch.randn(32, 32)
        self.assertEqual(torch.compile(fn, dynamic=True)(x), fn(x))


    def test_upsample_nearest2d_backward_symbolic_input_size(self):
        # `upsample_nearest2d_backward` guards the grad's spatial sizes but not
        # `input_size`, then uses the latter as a divisor for `ceildiv`, whose
        # result becomes a `range()` bound in `_adaptive_pooling_fn`. With a
        # symbolic `input_size` that raised
        # "TypeError: 'FloorDiv' object cannot be interpreted as an integer".
        def fn(grad, ref):
            return torch.ops.aten.upsample_nearest2d_backward.default(
                grad,
                [grad.shape[-2], grad.shape[-1]],
                [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]],
            )

        for device in ["cpu"] + ([GPU_TYPE] if HAS_GPU else []):
            # Both the exactly-divisible path (which routes to avg_pool2d) and
            # the general adaptive-pooling path.
            for grad_hw, ref_hw in [
                ((64, 64), (32, 32)),
                ((63, 63), (32, 32)),
                ((96, 96), (32, 32)),
                ((100, 70), (50, 35)),
            ]:
                torch._dynamo.reset()
                grad = torch.randn(1, 3, *grad_hw, device=device)
                ref = torch.randn(1, 3, *ref_hw, device=device)
                torch._dynamo.maybe_mark_dynamic(ref, 2)
                torch._dynamo.maybe_mark_dynamic(ref, 3)
                self.assertEqual(
                    fn(grad, ref),
                    torch.compile(fn, dynamic=True)(grad, ref),
                    atol=0,
                    rtol=0,
                )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
