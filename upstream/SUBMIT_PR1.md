# PR 1 — `[inductor] Fix symbolic output_size in upsample_nearestnd`

| | |
|---|---|
| **Commit** | `cee885e016` (branch `prseries`, on `b1716d913a`) |
| **Files** | `torch/_inductor/lowering.py` (+30/−4), `test/inductor/test_custom_lowering.py` (+181/−1) |
| **Blocked on** | Issue B marked **`actionable`** |
| **Send order** | **second** — after PR 3, before PR 2 |
| **Verified on** | 1× A100-SXM4-80GB, driver 575.57.08, torch 2.13.0+cu130, triton 3.7.1; CPU on 2.13.0+cpu |

---

## 1. The defect

`upsample_nearestnd` guards its input sizes and not its output sizes, divides one by the
other, and hands the quotient to `ops.constant`, which requires a concrete value:

```python
i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]   # guarded
o_sizes = output_size                                        # NOT guarded
inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
...
x = ops.mul(x, ops.constant(scale, torch.float32))           # scale may be sympy.Mul
```

Reached with a symbolic output size and no explicit `scales_x` entry — i.e. the caller
passed `size=`, not `scale_factor=` — the quotient stays a `sympy.Mul` and lowering aborts:

```
NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
```

## 2. The fix, and the alternative that was rejected

`None` marks the dims whose divisor is symbolic; `scale_fn` divides by `o_size` in the
kernel for those.

The obvious alternative is `V.graph.sizevars.guard_int(o)` — one line, no new op. It was
rejected on a measurement, not taste. Same lowering, same sweep of 7 distinct output
sizes, only the divisor handling differing:

| approach | lowering invocations for 7 output sizes |
|---|---|
| deferred divide (this PR) | **1** |
| `guard_int(o)` | **7** |

Past `torch._dynamo.config.recompile_limit` that drops the caller to eager, which is the
opposite of what someone compiling a resize-heavy model wants. **Put this table in the PR
body** — it converts "guarding would specialize" from an assertion into a comparison, and
it is the single most likely thing a reviewer challenges.

## 3. Why `ops.div_rn` and not `ops.truediv` — and why that is not #164144 again

This is the highest-risk part of the PR. Be explicit about the precedent before a reviewer
raises it.

[PR #164144](https://github.com/pytorch/pytorch/pull/164144) made inductor's `truediv`
emit `div_rn` globally for eager parity. It merged, was reverted several times, and was
finally gated behind `TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING` by
[#165566](https://github.com/pytorch/pytorch/pull/165566) after a throughput regression on
B200 ([#164301](https://github.com/pytorch/pytorch/issues/164301): ~6511 → ~4692 GB/s).

**Correctness — why `div_rn` is required.** Eager computes the scale as a float32 divide
(`compute_scales_value`, `ATen/native/UpSample.h`) then floors `dst_index * scale`.
Triton's `/` lowers to an approximate reciprocal, which can land one ulp low and change the
floor. Modelling `UpSample.h` exactly and sweeping 9 ratios (3305 output coordinates):

| divide | coordinates disagreeing with eager |
|---|---|
| `I / O` (approximate) | **45** |
| `tl.math.div_rn(I, O)` | **0** |

**Cost — why the regression does not apply.** The divisor here is the symbolic output
size: a *kernel argument*, uniform across the grid. Confirmed in the emitted Triton —

```python
tmp0 = tl.full([1], 128.0, tl.float32)
tmp1 = (ks0).to(tl.float32)
tmp2 = triton.language.div_rn(tmp0, tmp1)   # ks0/ks1 only, never xindex
tmp4 = tmp3 * tmp2
```

so it is one scalar op per program launch, hoisted out of the vectorised body.

⚠️ **The perf table below was re-measured 2026-08-28 and CORRECTED.** The old
numbers (0.965 / 0.957 / 0.983 / 0.984 — i.e. `div_rn` *faster*) came from
`perf_divrn.py`, which times each variant **once, in a fixed order**, with no clock
control. This box idles at 210 MHz against a 1410 MHz max, so whichever variant ran
first absorbed the clock ramp. Re-running that same script in a later session gave
1.009–1.212 — the opposite direction. Use `evidence/perf_divrn_controlled.py`
(CUDA-event timing, 15 A/B/A/B alternations, clock stabilization, and an **A-vs-A
control first** to prove the harness is unbiased).

Timed on A100, same lowering, only the divide differing — median of 15 alternations,
`[min, max]`:

| shape | control (rn vs rn) | `div_rn`/`truediv` | resolvable? |
|---|---|---|---|
| 1×3×64×64 → 128×128 | 0.979 [0.925, 1.207] | 0.984 [0.914, 1.102] | no — launch-bound |
| 1×3×256×256 → 512×512 | 0.980 [0.850, 1.160] | 0.984 [0.872, 1.193] | no — launch-bound |
| 8×64×128×128 → 256×256 | 1.000 [0.998, 1.045] | **1.017** [1.016, 1.030] | yes |
| 4×128×256×256 → 512×512 | 1.000 [0.9995, 1.0003] | **1.018** [1.017, 1.019] | yes |

At the two large shapes both the control and the A/B interval are tight and the A/B
interval **excludes 1.0**, so the cost is real and consistent: **≈1.8%**. At the two
small shapes the control itself swings ±10%, so nothing about the divide is
resolvable there. #164144's regression came from a *per-element* divide in a
memory-bound elementwise kernel — a different regime, and that structural argument
is what carries the PR.

> ⚠️ **Claim limit — state it in the PR.** Measured on A100 (SM 8.0) at four shapes.
> It says nothing about B200. Write **"the divide is loop-invariant here; measured
> ≈2% at large shapes on A100"** — never "div_rn is free" and never "not slower."
> A reviewer who re-runs it and sees 1.02 when the PR claimed 0.96 will distrust
> everything else in the diff; ≈2% for eager-exact indices is an easy trade to
> defend on its own terms.

**Supporting precedent — upstream already decided this for our exact op.** Cite
[PR #95698](https://github.com/pytorch/pytorch/pull/95698) (merged Mar 2023, nkaretnikov,
approved ezyang/jgong5): it changed the CPP expression printer to emit `(1.0/45.0)*ks2`
instead of `(1/45)*ks2` **because symbolic upsample scale factors were collapsing to zero
under C++ integer semantics**. ezyang's review was *"Nope. Make cpp implement div
correctly."* That establishes the principle this PR relies on — a symbolic-shape-derived
upsample scale is a real fraction and must be evaluated as one, not folded into integer
arithmetic — and it is the reason the rejected `guard_int` alternative in §2 is the wrong
shape of fix rather than merely a slower one.

**And the counter-question it invites, so answer it first.**
[Issue #185806](https://github.com/pytorch/pytorch/issues/185806) (`high priority`,
`module: correctness (silent)`) is the other side of #95698: there a float reciprocal was
used to recover an *integer* dimension, and inexact `1.0/1496.0` silently corrupted dynamic
batch sizes. A reviewer who knows it will ask why we are adding a float divide on a
symbolic size. The answer is that the two cases differ in what the quotient *is*: #185806
recovers an integer that must be exact, whereas this scale is a genuine fraction, and eager
itself computes it as an fp32 divide (`static_cast<float>(input_size) / output_size` in
`UpSample.h`). Matching eager therefore **requires** the float path — `div_rn` is what
makes it bit-exact rather than approximately right. Evidence: `../evidence/aten_ref_probe.py`
(45 → 0 disagreeing coordinates). See `../evidence/RESULTS_a100.md` §17.2.

## 4. Reachability — disclose this up front

**Not reachable through the ATen upsample ops.** Measured **0 lowering hits** across all
eight entry points tried: `F.interpolate` with `size=` (static and dynamic) and with
`scale_factor=`, `nn.Upsample`, raw `aten.upsample_nearest2d.default`, raw
`.vec`, under `inference_mode`, and `torch.export`.

The mechanism is the **decomposition table**, not just a dispatch key — be precise here,
because a reviewer who knows this code will check. Counted empirically on 2.13.0 (2026-08-28):
across `upsample_nearest{1,2,3}d` and `_upsample_nearest_exact{1,2,3}d` there are **19
overloads**, of which **18 are in `torch._decomp.decomposition_table`** (the exception is
`upsample_nearest2d.vec_out`) and only **6 are `CompositeImplicitAutograd`** — the `.vec`
overload of each of the six ops. The `.default` and `.out` overloads carry an Autograd
kernel. What makes the lowering unreachable is the decomposition table, not the dispatch
key: Inductor applies those decompositions (`arange/mul/_unsafe_index`) before lowering.

So do **not** write "all 12 overloads are CompositeImplicitAutograd" (wrong on both numbers)
and do **not** write "the decompositions are registered on `CompositeImplicitAutograd`."
Say "`upsample_nearest{1,2,3}d` are in `torch._decomp.decomposition_table`, so the
decompositions fire before lowering — verified 0 lowering hits."

Frame it as **defensive hardening of a live, exported lowering that a custom lowering or
out-of-tree backend can reach** — and that older versions did reach (on torch 2.3.1 this
aborts a real SDXL/VAE decoder compile under dynamic shapes). Do **not** call it a
user-visible regression. Overclaiming reachability is the fastest way to lose the review.

## 5. Evidence

- **3 new tests**, CPU and GPU. Each fails with only this fix reverted:
  `test_upsample_nearestnd_symbolic_output_size` and `..._one_graph` → FAIL; the
  `concrete_sympy` test and the other PRs' tests stay green.
- **Strict no-op for existing callers:** 63 real `F.interpolate`/`nn.Upsample`
  configurations (`size=` and `scale_factor=`, 1-D/2-D/3-D, static and dynamic, forward and
  autograd backward) — compiled results bit-identical with and without the patch, **0/63
  differ**.
- **Adversarial sweep: 42/42 bit-exact** — mixed symbolic/concrete dims, explicit
  `scales_x` on one dim only, nearest-exact at ULP-sensitive ratios, 1-D/3-D,
  float16/bfloat16, non-contiguous input, degenerate sizes (output 1, input 1, equal), and
  downsampling. Coverage verified by instrumenting the predicate: the deferred branch
  actually ran (`defer_per_dim=(True, True)`) rather than silently taking the concrete path.
- **Full dynamic-shapes suite:** 2621 tests, 1 failure, **pre-existing and unrelated**
  (`test_unbacked_reduction_cpu`, an inverted-xfail that fails identically with all patches
  reverted).

## 6. Two things a reviewer will probe

**"A concrete `sympy.Integer` output size is also a `sympy.Expr` — what happens?"** It must
be folded, not deferred, which is why the predicate is `isinstance(o, sympy.Expr) and not
o.is_number` rather than `isinstance` alone. An earlier revision of this patch stored the
output size *in* `inv_scales` in place of the scale, which forced that predicate to be
repeated at two sites; when the two drifted, a `sympy.Integer` was read as a deferred size
and divided by an already-inverted scale — producing silent out-of-range indices, not a
crash. `inv_scales` now holds `float | None`, one sentinel tested in one place.
`test_upsample_nearestnd_concrete_sympy_output_size` pins it.

**"Does `div_rn` exist on every backend?"** Yes — `codegen/common.py` defines a default
`div_rn` that falls through to `ops.truediv`, so CPU/Halide/MPS are unaffected. Only the
Triton backend overrides it.

## 7. Pre-push

- [ ] Issue B filed and labelled **`actionable`**; CLA signed
- [ ] Rebased onto current `origin/main`, tests re-run after the rebase
- [ ] `lintrunner -a` clean
- [ ] `python ../tools/state.py set pr1=off` → the two forward tests FAIL; back to `on` → pass
- [ ] Reviewers list **empty**
- [ ] PR body states the reachability limit and the A100-only perf caveat
