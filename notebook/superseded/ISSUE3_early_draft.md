# Second issue to file (pt2-bug-report template)

For **PR #3**. Same process as `ISSUE.md`: file it, wait for the **actionable** label, then
open the PR. Link it to the first issue as related — same defect class, different lowering.
Rewrite in your own words per `AI_POLICY.md`.

Template: <https://github.com/pytorch/pytorch/issues/new?template=pt2-bug-report.yml>

---

## Title

```
[inductor] upsample_nearest2d_backward fails to lower with a symbolic input_size
```

## 🐛 Describe the bug

`torch/_inductor/lowering.py::upsample_nearest2d_backward` guards the grad tensor's spatial
sizes but not `input_size`:

```python
inp_h = V.graph.sizevars.guard_int(inp_h)
inp_w = V.graph.sizevars.guard_int(inp_w)
*_batch, out_h, out_w = input_size          # not guarded
```

`out_h` / `out_w` are then used as divisors — first in a Python-level `%`, then in `ceildiv`,
whose result becomes a `range()` bound in `_adaptive_pooling_fn`:

```python
if inp_h % out_h == 0 and inp_w % out_w == 0:
    ...
h_kernel_max = ceildiv(inp_h, out_h)
...
for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
```

With a symbolic `input_size`, `ceildiv(64, s28)` produces `((s28 + 63)//s28)` — a sympy
`FloorDiv` — and lowering fails.

This is the same guarded-input / unguarded-output asymmetry as [the forward
issue](#NNNNN), but with a different symptom and a different correct fix: here the quotient
is a loop bound consumed at lowering time, so it genuinely has to be concrete — guarding is
the fix, not deferring.

**Reachability, up front:** autograd does not reach this lowering today — the backward is
decomposed before Inductor sees it. I checked `F.interpolate` and `nn.Upsample` backward with
`size=` and `scale_factor=`, static and dynamic, and with the input's spatial dims marked
dynamic: zero lowering hits. So this is not a user-visible regression through the public API;
it is reachable by calling the aten op directly, where it fails. Filing because the asymmetry
looks unintentional and the fix is two lines — happy to be told it is not worth changing.

Repro (CPU is enough):

```python
import torch

def fn(grad, ref):
    return torch.ops.aten.upsample_nearest2d_backward.default(
        grad,
        [grad.shape[-2], grad.shape[-1]],
        [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]],
    )

grad = torch.randn(1, 3, 64, 64)
ref = torch.randn(1, 3, 32, 32)
torch._dynamo.maybe_mark_dynamic(ref, 2)
torch._dynamo.maybe_mark_dynamic(ref, 3)
torch.compile(fn, dynamic=True)(grad, ref)
```

Succeeds with `dynamic=False`; fails with `dynamic=True`. Reproduces on both the
exactly-divisible path (64 → 32, which routes to `avg_pool2d`) and the general
adaptive-pooling path (63 → 32).

I have a two-line patch plus a regression test and can send it if this is accepted.

## Error logs

```
torch._inductor.exc.InductorError: TypeError: 'FloorDiv' object cannot be interpreted as an integer
```

The relevant frames:

```
  File "torch/_inductor/lowering.py", line 6288, in fn      # upsample_nearest2d_backward
  File "torch/_inductor/lowering.py", line 5898, in fn      # _adaptive_pooling_fn
    for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
```

## Versions

Reproduced on `2.13.0` (CPU and CUDA 13.0 builds). Paste your own:

```
python -m torch.utils.collect_env
```
