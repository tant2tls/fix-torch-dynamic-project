# PR #1 body — local review draft

**Branch:** `inductor-upsample-symbolic-output-size`
**Patch:** `0001-inductor-Fix-symbolic-output_size-in-upsample_neares.patch`
**Files:** `torch/_inductor/lowering.py` (+30 −4), `test/inductor/test_custom_lowering.py` (+181 −1)

Open this only after the issue is labeled **actionable**. Replace `#NNNNN` with the issue
number. Keep the body roughly this length — PyTorch's AI policy warns explicitly against
overly verbose contributions.

---

Fixes #NNNNN

### The problem

`upsample_nearestnd` guards its input sizes but not its output sizes, then divides one by
the other and passes the quotient to `ops.constant`, which requires a concrete value:

```python
i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]   # concrete
o_sizes = output_size                                        # not concrete
inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
...
x = ops.mul(x, ops.constant(scale, torch.float32))
```

With a symbolic output size and no explicit `scales_x` entry (`size=` rather than
`scale_factor=`), the quotient stays a `sympy.Mul` and lowering fails with
`NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>`.

### The fix

Keep `i / o` when `o` is concrete; when it is symbolic, defer the division to the kernel.

The runtime division uses `ops.div_rn` rather than `ops.truediv`, for the same reason
`_floor_div_floating` already uses `_div_rn`: Triton's default fp32 division is an
approximate reciprocal that can land one ulp below the true quotient and change the floored
index. Eager computes this scale as a float32 division (`compute_scales_value`,
`ATen/native/UpSample.h`), so `div_rn` reproduces it exactly. Across nine measured ratios
and 3305 output coordinates, the approximate divide disagrees with eager on **45**
coordinates; `div_rn` disagrees on **0**. The generated kernel contains
`triton.language.div_rn`, and the 42-case adversarial CPU/CUDA sweep is bit-exact.

Keeping `i / o` for concrete `o` makes this a no-op for existing callers. In a
same-device comparison with all three candidates ON versus OFF, all **63/63 compiled
result hashes are identical**.
Raw generated-code hashes are intentionally not used for this claim because an A-vs-A
control changed 50/63 of them even with the patch state held constant.

`inv_scales` holds `float | None`, where `None` means "deferred", and `scale_fn` takes the
output size as its own parameter. It deliberately does *not* store the output size *in*
`inv_scales`: that would make one list hold two different kinds of value and force the same
"is this symbolic" predicate at two sites. `isinstance(o, sympy.Expr)` alone is not a correct
predicate — `sympy.Integer(64)` is a `sympy.Expr` but concrete, so it must still be folded —
and when the two sites disagreed on that, a folded scale was read back as an output size,
dividing by an already-inverted scale and indexing far outside the input instead of crashing.
One sentinel, tested in one place, removes that failure mode.
`test_upsample_nearestnd_concrete_sympy_output_size` pins the case for both `int` and
`sympy.Integer` sizes.

### Why defer, rather than guard the divisor

Guarding (`V.graph.sizevars.guard_int(o)`, mirroring the line above) also fixes the crash and
produces a *faster* kernel — the folded constant lets Triton strength-reduce the index
arithmetic: 1.59 ms vs 12.7 ms for one 8×32×512×512 → 2048² call on an H100. Attributing that
gap: a plain `ops.truediv` in place of `div_rn` measures 12.70 ms, so the ~8× gap is the
dynamic index math, not the rounding mode. (A controlled A/B on A100 puts `div_rn` at ≈2%
above `truediv` for the same lowering — real, but two orders of magnitude too small to
explain the 1.59-vs-12.7 gap.)

But guarding specializes on the output size, so a caller sweeping resolutions recompiles per
size and, past `torch._dynamo.config.recompile_limit`, stops compiling at all. Over 12
distinct output sizes × 5 iterations, end to end:

| | total | graphs |
|---|---|---|
| guard the divisor | 7.09 s | 8, then eager fallback |
| defer the division | **0.81 s** | **1** |

The crossover for a *single fixed* output size is ≈213 iterations; beyond that the folded
constant wins. I chose the deferred form because it cannot fall off the compiled path, and a
caller who wants the fast kernel can guard the size itself. Happy to flip it if you would
rather optimize the fixed-shape case — the tradeoff is the substantive question here and I
did not want to bury it.

**No existing caller pays either way:** the deferred path only fires when the output size is
symbolic, which today means the compile crashes. Nothing that works now changes.

**Scope, to be precise:** this removes only the *output*-size specialization. `i_sizes` is
still guarded above (pre-existing), so a caller varying its **input** size still recompiles
per size and still hits `recompile_limit`. Measured: **1** lowering call for 20 varying output
sizes, **8** for 20 varying input sizes.

### Reachability — the caveat, up front

The ATen `upsample_nearest*` ops do not reach this lowering: `upsample_nearest{1,2,3}d` are
present in `torch._decomp.decomposition_table`, and Inductor applies those decompositions
(`arange`/`mul`/`_unsafe_index`) before lowering, so this function is not reached. Zero
hits across `torch.compile`, `inference_mode`, a direct `aten.upsample_nearest2d.default`
call, `torch.export`, `run_decompositions({})` and AOTInductor.

So this is **hardening a lowering ATen cannot currently reach**, not a fix for a user-visible
regression. A test driven through `F.interpolate` would pass with or without the change and
pin nothing, which is why the tests register a custom op and call `upsample_nearestnd`
directly — the technique the neighbouring tests in `test_custom_lowering.py` already use. If
you would rather not carry the change given the reachability, that is a fair call.

### Testing

Both tests fail before this change and pass after:

```bash
python test/inductor/test_custom_lowering.py -k upsample_nearestnd
```

- `test_upsample_nearestnd_symbolic_output_size` — 5 shape pairs × `nearest` /
  `nearest-exact`, CPU and GPU, `atol=rtol=0` vs eager. Includes `448 -> 192` and
  `384 -> 363`, where a one-ulp scale error changes the result.
- `test_upsample_nearestnd_concrete_sympy_output_size` — a concrete `sympy.Integer` size stays
  on the folded path. Passes on unpatched `main` too; it guards the new branch, not the bug.
- `test_upsample_nearestnd_symbolic_output_size_one_graph` — asserts 5 distinct output sizes
  compile to a single lowering call, so a future change back to guarding fails here.

Also checked on an H100 (torch 2.13, CUDA 13.0): unaffected by `cpp_wrapper`,
`triton.cudagraphs`, `max_autotune`, and both settings of
`eager_numerics.division_rounding`; bit-exact for fp16 / bf16 / fp64; 1-D and 3-D `nearest`
work through the same lowering. `F.interpolate` paths unchanged (`scale_factor=` 2.0/1.5/0.5,
static `size=`, 1d/2d/3d, both modes).

Wider sweep: **276 (input, output) shape pairs × `nearest`/`nearest-exact` × CPU/CUDA =
1104/1104 bit-exact**, `atol=rtol=0`, spanning ratios from 512→3 to 7→1000 and including the
ULP-hostile ones. The current six-test guard matrix and the 42-case adversarial sweep were
re-run on an RTX PRO 6000 Blackwell Server Edition on 2026-09-04.

Two details behind the `div_rn` choice, in case they come up:
`nearest_neighbor_compute_source_index` takes `floor` of a non-negative product, so eager's
`floorf` and the lowering's `to_dtype(int32)` truncation agree; and eager's
`min(..., input_size - 1)` clamp never binds for a correctly-rounded scale (checked all
262144 `(i, o)` pairs in 1..512, both modes — 0 cases), which is what makes
`indirect_indexing(..., check=False)` safe here.

---

**Do not add reviewers.** No entry in `.github/merge_rules.yaml` covers
`torch/_inductor/lowering.py`, and `CONTRIBUTING.md` says to leave Reviewers empty so triage
can add the module label and assign. Label `module: inductor` if you can.

---

### ⚠️ AI-assistance disclosure — required by `AI_POLICY.md`, edit to match the truth

`AI_POLICY.md`: *"AI-generated content in comments, issues, or PRs must be clearly disclosed
and contained."* Keep whichever sentence is accurate, delete the rest, and **do not submit
without one of them.** Adjust to reflect what you actually did — an inaccurate disclosure is
worse than none.

> Disclosure: I used an AI assistant while investigating this and while drafting parts of
> this description. The diagnosis, the choice of `div_rn` over `truediv`, the
> defer-vs-guard tradeoff, and the measurements above are mine; I have read the change and
> can answer for every line of it.

Then delete this whole section from the pasted body — it is instructions to you, not text
for the PR.
