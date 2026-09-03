# Issue to file first (pt2-bug-report template)

PyTorch requires that a PR map to an issue labeled **actionable**:

> "Only PRs that address issues labeled actionable will be considered for review."
> — [The Ultimate Guide to PyTorch Contributions](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)

So **file this first**, wait for a maintainer to label it, then open PR #1.
Use the **torch.compile** template: <https://github.com/pytorch/pytorch/issues/new?template=pt2-bug-report.yml>

Per [AI_POLICY.md](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md), review
everything below yourself before posting, and keep any AI-generated text you quote inside
a code block with your own commentary around it. Post in your own words.

---

## Title

```
[inductor] upsample_nearestnd fails to lower with a symbolic output_size
```

## 🐛 Describe the bug

`torch/_inductor/lowering.py::upsample_nearestnd` passes `V.graph.sizevars.guard_int`
over its **input** sizes but not its **output** sizes, then divides one by the other and
hands the quotient to `ops.constant`, which requires a concrete value:

```python
i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]   # concrete
o_sizes = output_size                                        # not concrete
inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
...
x = ops.mul(x, ops.constant(scale, torch.float32))
```

If this lowering is reached with a symbolic output size and no explicit `scales_x` entry
(i.e. the caller passed `size=` rather than `scale_factor=`), `scale` stays a `sympy.Mul`
and lowering fails.

**Reachability, stated up front:** the ATen `upsample_nearest*` ops do **not** reach this
lowering today — their decompositions are registered on `CompositeImplicitAutograd`, so
they fire before Inductor sees the op. I checked `torch.compile`, `torch.inference_mode`,
a direct `aten.upsample_nearest2d.default` call, `torch.export`,
`export(...).run_decompositions({})` and AOTInductor: zero hits. So this is **not** a
user-visible regression through `F.interpolate`; it is reachable from a custom lowering or
an out-of-tree backend, where it crashes. Filing it because the asymmetry looks
unintentional and the fix is small — happy to be told it is not worth changing.

Repro (drives the lowering directly, the way `test/inductor/test_custom_lowering.py`
already does for lowerings that ATen cannot reach):

```python
import torch
import torch.nn.functional as F
from torch._inductor.lowering import register_lowering, upsample_nearestnd

with torch.library._scoped_library("demo", "FRAGMENT") as lib:
    lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
    ref = lambda x, h, w: F.interpolate(x, size=(int(h), int(w)), mode="nearest")
    lib.impl("ups2d", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta")
    lib.impl("ups2d", ref, "CPU")
    register_lowering(torch.ops.demo.ups2d)(
        lambda x, h, w: upsample_nearestnd(x, [h, w], (None, None), n=2)
    )

    x = torch.randn(1, 3, 32, 32)
    sizes = torch.randn(1, 3, 64, 64)
    torch._dynamo.maybe_mark_dynamic(sizes, 2)
    torch._dynamo.maybe_mark_dynamic(sizes, 3)
    torch.compile(
        lambda a, s: torch.ops.demo.ups2d(a, s[0], s[1]), dynamic=True
    )(x, sizes.shape[-2:])
```

Two things worth noting for whoever picks this up:

1. Guarding the divisor (`guard_int(o)`) fixes the crash but specializes on the output
   size — one recompile per distinct size, and past `recompile_limit` the frame drops back
   to eager. Deferring the division to the kernel keeps a single graph.
2. The runtime division needs `ops.div_rn`, not `ops.truediv`. Eager computes this scale
   as a float32 division (`compute_scales_value` in `ATen/native/UpSample.h`), and
   Triton's default fp32 divide is an approximate reciprocal that can land one ulp off and
   change the floored index — the same hazard `_floor_div_floating` already documents.
   Sweeping `(i, o)` over `1..512`, an approximate reciprocal disagrees with a
   correctly-rounded divide on the floored index for **11695** pairs.

I have a patch for this and can send it if the issue is accepted.

## Error logs

```
torch._inductor.exc.InductorError: NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
```

The informative frame is roughly a dozen frames above the raised error:

```
  File "torch/_inductor/lowering.py", line 5360, in scale_fn
    x = ops.mul(x, ops.constant(scale, torch.float32))
  ...
  File "torch/fx/proxy.py", line 483, in create_arg
    raise NotImplementedError(f"argument of type: {type(a)}")
```

## Versions

Reproduced on `2.13.0` (CPU and CUDA builds). Fill in your own `collect_env.py` output
when filing — the template requires it:

```
python -m torch.utils.collect_env
```
