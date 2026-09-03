# Experiment report: designing the upstream fix

This is the working record behind `upstream/` — every measurement, every wrong
turn, and the reasoning that selected the final patch. `README.md` (this
directory's parent) tells the *story* of the original 2.3.1 bug;
`upstream/README.md` is the *pull request*. This file is the **lab notebook**:
it exists so that any number in either document can be traced to the run that
produced it.

Everything below was measured, not reasoned about. Where a claim was made from
reading code and later contradicted by measurement, both are kept.

> **§9 supersedes parts of §3–§5.** A later pass through the PyTorch source found
> that the codebase already has the primitive this fix needed (`ops.div_rn`) and
> that its contribution policy requires a different PR shape. The fix shrank from
> +43/−4 to **+24/−2** and split into two PRs. §3–§5 are kept because the
> variant selection is still the evidence for *why* a runtime divide beats a
> guard — but the shipped patch is the one described in §9.

> **Path note (2026-08-28 consolidation).** The harnesses this file cites moved from
> `upstream/dev/` to **`../evidence/dev_harnesses/`**; their portable, documented
> successors are in **`../evidence/`**. Paths below have been updated.

**Provenance of every number, since the container changes between sessions:**

| measured | where | what it covers |
|---|---|---|
| session A | `tan-2gpu-chip-0-3`, **2× H100 80GB**, torch 2.13.0+**cu130** | all CUDA numbers: the fp32-divide wrong-pixel result, graph counts, byte-identical codegen |
| session B | `hu-tanngo-lv`, **no GPU**, torch 2.13.0+**cpu** | §9: the `div_rn` redesign, the 11695-pair ULP characterization, the 25-lowering sweep, both PRs' tests in both directions |

Nothing from session A was re-run in session B — **the Triton `div_rn` path has
not been executed on a GPU.** §9.6 says what remains to verify before pushing.

---

## 0. Environment

Two environments matter, and conflating them has cost time before.

| | original bug | upstream PR work |
|---|---|---|
| torch | 2.3.1+cu121 | **2.13.0+cu130** (A) / 2.13.0+cpu (B) |
| python | 3.11.15 | 3.12.14 |
| built from | internal venv | `uv venv`, PyPI wheel |
| GPU | H100 (4×, earlier session) | **H100 80GB HBM3 (2×)** (A) / **none** (B) |
| driver | 575.57.08 | 575.57.08 (A) / n/a (B) |
| host | RunAI container | `tan-2gpu-chip-0-3` (A) / `hu-tanngo-lv` (B) |
| CPU quota | 198 (cgroup) | 198 (cgroup) |

**Correction to a prior session's note.** An earlier note recorded that this
box's driver was CUDA 12.9 and therefore *"cu13 wheels fall back to CPU
silently."* That is false on this container:

```
$ nvidia-smi | head -3      # CUDA Version: 13.0
$ /tmp/cu130/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.13.0+cu130 True
```

The container spec is not stable across sessions, so the driver must be
re-probed rather than inherited. Every GPU number in this report was produced
with `torch 2.13.0+cu130` on a real H100, confirmed by
`torch.cuda.get_device_name(0)`.

The upstream checkout used for the patch is `pytorch/pytorch` @ `497054b`.

---

## 1. What the PR must accomplish

The 2.3.1 bug (README §2–4) was fixed upstream at the **dispatch** layer: from
2.4, `upsample_nearest*` decompositions carry
`py_impl(DispatchKey.CompositeImplicitAutograd)`, so they fire even under
`inference_mode` and the buggy lowering is never reached from ATen.

But the **caller-side defect was never repaired**. On current `main`:

```python
i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]   # guarded
o_sizes = output_size                                        # NOT guarded
inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
...
x = ops.mul(x, ops.constant(scale, torch.float32))           # wants a leaf
```

So the PR has two jobs, and the second is the one that makes it worth a
reviewer's time:

1. **Fix the lowering** so a symbolic output size compiles.
2. **Generalize the failure mode** so the *next* lowering that violates the
   same contract says so immediately, instead of failing fifteen frames away.

---

## 2. The baseline: what a developer sees today

`../evidence/dev_harnesses/trace_crash.py` reproduces the crash on 2.13.0 and prints the full
traceback. Measured length: **18 frames**. The last three:

```
File ".../torch/_inductor/ops_handler.py", line 1078, in _default
    return getattr(self._inner, name)(*args, **kwargs)
File ".../torch/fx/proxy.py", line 875, in __call__
File ".../torch/fx/proxy.py", line 483, in create_arg
    raise NotImplementedError(f"argument of type: {type(a)}")
torch._inductor.exc.InductorError: NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
```

The informative frame — `lowering.py:5042, x = ops.mul(x, ops.constant(scale, ...))`
— is **twelve frames above** the raised error. The message names a sympy class
and nothing else: not the op whose contract was violated, not the value, not the
remedy. This measurement is the entire justification for change #2.

---

## 3. Selecting the fix: four variants, four axes

The naive fix is to guard the divisor (`guard_int(o)`), mirroring the line above
it. That was the previous session's patch. To test whether it was the *best*
fix, `../evidence/dev_harnesses/variants.py` implements four variants of the divisor line and
scores them on four axes. All four variants are re-implementations of the
lowering reached through a scoped custom op, so the installed torch is never
modified during selection.

| variant | divisor line |
|---|---|
| `baseline` | `i / o` — today's code |
| `guard_int` | `i / V.graph.sizevars.guard_int(o)` |
| `f32` | runtime `ops.truediv` in float32 when `o` is symbolic |
| `f64` | runtime `ops.truediv` in float64, narrowed to float32 |

### 3.1 Axis 1 — correctness (bit-exact vs eager, symbolic sizes)

15 shape pairs × {`nearest`, `nearest-exact`} = 30 cases. `atol=rtol=0`.

```
$ python ../evidence/dev_harnesses/variants.py --mode correctness --device cpu
  baseline   bit-exact   0/30  wrong   0  crashed  30
  guard_int  bit-exact  30/30  wrong   0  crashed   0
  f32        bit-exact  30/30  wrong   0  crashed   0
  f64        bit-exact  30/30  wrong   0  crashed   0

$ python ../evidence/dev_harnesses/variants.py --mode correctness --device cuda
  baseline   bit-exact   0/30  wrong   0  crashed  30
  guard_int  bit-exact  30/30  wrong   0  crashed   0
  f32        bit-exact  25/30  wrong   5  crashed   0
             ! (448, 448)->(192, 192): 7917/110592 elems
             ! (384, 384)->(363, 363): 4344/395307 elems
             ! (10, 10)->(100, 100): 5157/30000 elems
             ! (32, 32)->(65, 33)/exact: 195/6435 elems
  f64        bit-exact  30/30  wrong   0  crashed   0
```

**This is the most important result in the report.** The float32 variant is
perfectly green on CPU and **silently produces wrong pixels on an H100** — no
crash, no warning, just shifted samples. Had the fix been validated on CPU only,
a numerics bug would have shipped.

**Why.** Eager computes the scale as a float32 division
(`compute_scales_value`, `ATen/native/UpSample.h:261`):

```cpp
return (scale.has_value() && scale.value() > 0.)
    ? static_cast<scalar_t>(1.0 / scale.value())
    : (static_cast<scalar_t>(input_size) / output_size);
```

so float32 is the right *value* to target. But Triton lowers float32 `/` to the
approximate `div.full` unless `config.eager_numerics.division_rounding` is set,
and that flag is **off by default**. Being one ULP high is enough: for
448→192 the scale is `0x40155556` instead of the correctly-rounded `0x40155555`,
which pushes `floor(27 * scale)` from 62 to 63 and samples the wrong pixel.
Computing in float64 and narrowing is correctly rounded regardless of how the
backend lowers float32 division.

A PR must not depend on a non-default config flag, so `f64` it is.

### 3.2 Axis 2 — is the change a no-op for existing callers?

The paths that compile today must generate **byte-identical** kernels. 8
concrete cases × {`nearest`, `nearest-exact`} = 16, comparing the sha256 of the
generated source against `baseline`.

**A harness bug first, because it nearly produced a false result.** The initial
run reported `0/16` for *all three* variants, including cases they cannot touch.
A baseline-vs-baseline self-check showed the hash was unstable, and the diff was
a single line:

```
-# AOT ID: ['0_inference']
+# AOT ID: ['1_inference']
```

— a global per-process compile counter. After normalising that (plus the scoped
library tag), the self-check is `STABLE` and the real result appears:

```
$ python ../evidence/dev_harnesses/variants.py --mode codegen --device cuda
  [self-check] baseline vs baseline: STABLE
  guard_int  byte-identical 16/16
  f32        byte-identical 16/16
  f64        byte-identical 16/16
```

All three are strict no-ops. The self-check is now part of the harness and
prints a warning if it ever regresses — a comparison harness that cannot
reproduce its own baseline should not be trusted to judge a variant.

What preserves this for `f64`: when `o` is already concrete, the code keeps
today's folded `i / o` constant. An *unconditional* runtime divide would emit
`32.0/64.0` instead of the folded `0.5` and would not be a no-op.

### 3.3 Axis 3 — recompilation cost

```
$ python ../evidence/dev_harnesses/variants.py --mode graphs --device cuda
  guard_int   8 lowering calls for 8 sizes (8 ran)
  f32         1 lowering calls for 8 sizes (8 ran)
  f64         1 lowering calls for 8 sizes (8 ran)
```

`guard_int` compiles one graph per distinct output size. This was known and
disclosed in the previous PR draft as an accepted cost. Extending the sweep to
20 sizes shows it is worse than a cost:

```
$ DEV=cuda python /tmp/fallback_proof.py      # inlined below, see §6
W0825 10:06:08 torch/_dynamo/convert_frame.py:1994] [0/8] torch._dynamo hit config.recompile_limit (8)
  guard_int  lowering calls (=graphs)   8 for 20 sizes, wrong=0
  f64        lowering calls (=graphs)   1 for 20 sizes, wrong=0
```

Past Dynamo's `recompile_limit` (default 8) the frame stops being recompiled and
**falls back to eager**. So `guard_int` does not merely add specialisations — for
a caller with more than 8 distinct output sizes it silently removes
compilation. The recompile reason names the guard line directly:

```
last reason: 0/7: s[0] == 200  # (variants.py:125 in lowering)
```

This is the same shape-variety cliff documented in `../fix.md` §4c, except here
the *fix itself* would cause it. That makes `guard_int` not just dominated but
actively harmful for the workload the bug came from — a diffusion decoder
serving many resolutions.

### 3.4 Axis 4 — hard `mark_dynamic`

A prior session recorded that `guard_int` turns a hard `mark_dynamic` on the
output dim into a `ConstraintViolationError` (9/10 failing). **That did not
reproduce.** Measured on 2.13.0+cu130:

```
$ python ../evidence/dev_harnesses/variants.py --mode constraint --device cuda
  guard_int  10/10 compiled and bit-exact
  f32        10/10 compiled and bit-exact
  f64        10/10 compiled and bit-exact
```

and with a single reused compiled callable, `guard_int` is also 10/10 correct —
it just hits the recompile limit (§3.3). The earlier claim is **withdrawn**: the
case against `guard_int` rests on the recompile cliff, which is solidly
measured, not on a constraint violation. Reporting this correction matters more
than the tidier story would have.

### 3.5 Verdict

| axis | `guard_int` | `f32` | `f64` |
|---|---|---|---|
| bit-exact, symbolic, CPU | 30/30 | 30/30 | **30/30** |
| bit-exact, symbolic, **CUDA** | 30/30 | **25/30 — 5 WRONG** | **30/30** |
| byte-identical on existing paths | 16/16 | 16/16 | **16/16** |
| graphs for 8 output sizes | 8 | 1 | **1** |
| graphs for 20 output sizes | 8, then eager | 1 | **1** |
| depends on a non-default flag | no | **yes** | no |

`f64` wins on every axis. It is the patch.

---

## 4. The generalized diagnostic

The fix above repairs one lowering. The traceback in §2 is what makes the *class*
of bug expensive, so the PR also enforces the contract.

**Placement.** `OpsWrapper._default` in `torch/_inductor/virtualized.py` — the
funnel every `ops.*` call from a lowering passes through. Checking here catches
all callers, costs one `isinstance` on a path that already builds two lists, and
needs no change to any backend handler.

**What the audit found.** Grepping `ops.constant(` on `main` gives 44 call sites.
Most pass literals. Filtering to computed values and reading each one, **four
sites already defend this exact contract by hand**:

| site | code |
|---|---|
| `lowering.py:690` (`add`/`sub` with `alpha`) | `if isinstance(alpha, sympy.Basic): ops.index_expr(...) else: ops.constant(...)` |
| `lowering.py:4407` (`_full`) | `elif isinstance(value, sympy.Basic): ops.index_expr(...)` |
| `lowering.py:8586` (`addcmul`) | `# Use index_expr for sympy expressions (e.g., from .item()), constant otherwise` |
| `lowering.py:8657` (`addcdiv`) | same comment, same guard |

That comment — *"Use index_expr for sympy expressions … constant otherwise"* — is
the invariant, written out longhand, four times, with no mechanism enforcing it.
`upsample_nearestnd` is the site that forgot. This is the strongest available
argument that the check belongs upstream: it is not a new rule, it is an existing
convention that the codebase currently trusts every author to remember.

**The message.** Names the expression, its free symbols, and *both* remedies
(guard the inputs, or use `ops.index_expr`):

```
ops.constant() requires a concrete value, but got the symbolic expression
2*s0 (free symbols: ['s0']).
A lowering probably derived this from a size that was never made concrete.
Either guard the inputs to it (e.g. V.graph.sizevars.guard_int(...)), which
specialises on that size, or -- if the value should stay dynamic -- compute it
in the kernel and pass it via ops.index_expr(), whose contract accepts
symbolic expressions.
```

**False positives.** `sympy.Integer(3)` and `sympy.Float(0.5)` are `sympy.Expr`
but *are* numbers, and legitimately reach `ops.constant`. The check is
`isinstance(value, sympy.Expr) and not value.is_number`, and the test asserts
`0.5, 1, True, sympy.Integer(3), sympy.Float(0.5)` all pass, positionally and by
keyword.

---

## 5. Validating the final patch

Unlike §3, these runs patch the **real installed torch** (`patch -p1` into
site-packages), so they exercise genuine code paths rather than a
re-implementation.

### 5.1 The defect path and ATen paths

```
$ DEV=cuda python /tmp/test_real_patch.py
torch 2.13.0+cu130  device cuda
A. defect path: bit-exact 24/24  wrong 0  crash 0
B. ATen interpolate: 12 match, 0 mismatch
C. diagnostic: fired, actionable=True
C2. no false positives on concrete values

$ DEV=cpu  ...
A. defect path: bit-exact 24/24  wrong 0  crash 0
B. ATen interpolate: 12 match, 0 mismatch
```

### 5.2 The tests, in both directions

This is what makes them regression tests rather than tests.

> **Note (session B):** the runs below used `../evidence/dev_harnesses/run_pr_tests.py`, which drove
> session A's three-test single-PR layout. After the split (§9.4) it was replaced
> by **`../evidence/dev_harnesses/run_pr1_tests.py`** for PR #1's two tests; PR #2's two tests are
> covered in §9.5. The transcripts are kept as the record of session A.

```
$ python ../evidence/dev_harnesses/run_pr_tests.py --device cuda     # patch applied
torch 2.13.0+cu130   device cuda   patch: patched
  [PASS] test_ops_constant_rejects_symbolic_value
  [PASS] test_upsample_nearestnd_symbolic_output_size
  [PASS] test_upsample_nearestnd_symbolic_output_size_single_graph
3 passed, 0 failed        VERDICT: as expected

$ patch -p1 -R < pr.patch && python ../evidence/dev_harnesses/run_pr_tests.py --device cuda
torch 2.13.0+cu130   device cuda   patch: unpatched
  [FAIL] test_ops_constant_rejects_symbolic_value
         AssertionError: OpsWrapper._check_constant is missing
  [FAIL] test_upsample_nearestnd_symbolic_output_size
         InductorError: NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
  [FAIL] test_upsample_nearestnd_symbolic_output_size_single_graph
         InductorError: NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
0 passed, 3 failed        VERDICT: as expected
```

`run_pr_tests.py` detects the patch state from the installed source and asserts
the *expected* verdict for that state, so it cannot silently pass in the wrong
direction. Both devices, both directions: green.

### 5.3 Blast radius

The diagnostic runs on every `ops.constant` call during lowering, so the risk is
breaking an unrelated lowering. A 20-lowering sweep — avg/max/adaptive pooling,
clamp, constant pad, `masked_fill`, softmax, gelu, layernorm, `where`, scalar
div, pow, cumsum, nearest **and bilinear** upsample, hardtanh, logsumexp, norm,
conv2d, tril — run both static and `dynamic=True`:

```
broad lowering sweep: 19 ok, 1 bad, of 20
dynamic sweep:        19 ok, 1 bad, of 20
```

The one failure was **my test's bug**, not the patch's: the conv2d case called
`torch.randn` for the weight inside the lambda, so eager and compiled saw
different weights. Hoisting the weight:

```
conv2d dynamic=False: allclose=True
conv2d dynamic=True:  allclose=True
```

So the honest number is **20/20 static and 20/20 dynamic**. Recording it this way
rather than quietly fixing the harness is the point — an unexplained 19/20 in a
PR invites exactly the suspicion it deserves.

### 5.4 Regression baseline unchanged

`upstream/verify_claims.py`, which re-derives every claim independently, on
2.13.0+cu130 with a real H100:

```
1. REACHABILITY  [PASS] 0 lowering hits (torch.compile=0, inference_mode=0,
                        direct aten op=0, export + run_decompositions({})=0)
2. DEFECT        [PASS] 12 crashed, 0 wrong
3. FIX           [PASS] 12 bit-exact
4. NO-REGRESSION [PASS] 14/14 match
5. REJECTED ALT  [PASS] 3 disagreements (448->192 coord 27: eager=62 exact=63; ...)
5/5 checks reached their expected verdict
```

Check 5 is worth keeping in view: it refutes the *other* tempting fix — keeping
the scale symbolic and emitting exact integer index math. That disagrees with
eager's float32 arithmetic on ordinary sizes (2102 disagreements fuzzing
i,o ∈ 1..256). The final patch matches eager because it does the same float32
division eager does, just correctly rounded.

---

## 6. Reproducing this report

```bash
# environment (the driver on this container serves cu130; re-probe yours)
uv venv /tmp/cu130 --python 3.12
uv pip install --python /tmp/cu130/bin/python torch numpy \
    --index-url https://download.pytorch.org/whl/cu130

# variant selection (never modifies installed torch)
/tmp/cu130/bin/python ../evidence/dev_harnesses/variants.py --mode all --device cuda
/tmp/cu130/bin/python ../evidence/dev_harnesses/variants.py --mode all --device cpu

# the baseline traceback
DEV=cuda /tmp/cu130/bin/python ../evidence/dev_harnesses/trace_crash.py

# the tests, both directions (needs the patch applied/reverted around it)
/tmp/cu130/bin/python ../evidence/dev_harnesses/run_pr1_tests.py

# the fix as installed, end to end
/tmp/cu130/bin/python ../evidence/dev_harnesses/check_installed_fix.py --device cuda

# independent re-derivation of every PR claim
/tmp/cu130/bin/python upstream/verify_claims.py
```

`variants.py`, `trace_crash.py`, `run_pr1_tests.py` and `check_installed_fix.py`
are committed here. Session A's `run_pr_tests.py` was replaced by
`run_pr1_tests.py` when the PR split (§9.4). Throwaway scripts quoted in §5
(`/tmp/test_real_patch.py`, `/tmp/broad_check.py`) are not committed — `/tmp` is
not durable and the container was in fact wiped between sessions, so re-derive
rather than trust them. Session B's equivalents live in `../evidence/dev_harnesses/`.

**Two harness gotchas that each cost time:**

- Leaking raw `torch.library.Library` objects across many variants **segfaults**
  the interpreter at teardown. Use `torch.library._scoped_library`, and pop the
  registered lowerings out of `lowerings` afterwards or the next variant
  inherits them.
- `TORCHINDUCTOR_COMPILE_THREADS=1` must be set **before `import torch`**.
  Without it, ~1 run in 18 fails on a Triton cache race
  (`FileNotFoundError: ...cubin.tmp.pid_*`) and a flake reads as a verdict.

---

## 7. What this PR claims, and what it does not

**Claims.** A latent type-contract violation in `upsample_nearestnd` is fixed
without specialising on output size and without changing generated code for any
path that compiles today; the contract itself is now enforced at the boundary
with an actionable message; three tests pin all of it and fail without it.

**Does not claim.** That any ATen entry point reaches this lowering today. It
does not — verified across seven entry points, zero hits. The honest framing is
**defensive hardening of a reachable-but-not-ATen-reached lowering**, and the
reachability table is in the PR body rather than hidden, because it is the
legitimate basis for a maintainer to decline.

**For CV / SoP / email.** Until it merges, the accurate sentence is *"submitted a
fix and regression tests to PyTorch for a latent type-contract violation in an
Inductor lowering, plus a diagnostic that generalizes the failure mode."* Never
"fixed a bug in PyTorch" — `repro/check_upstream.py` collapses that in one
command.

---

## 8. Open threads

- **Not yet sent.** Needs a GitHub account + CLA. See §9.6 for the current send
  order — **the issue must be filed and labeled `actionable` before any PR**.
- **`lintrunner -a` has not been run** — no PyTorch build on this box. Expect
  formatting-only edits.
- **The upstream test files were not executed** against the patch: running the
  cloned repo's test files was not authorised in this session. The test bodies
  were validated verbatim from `../evidence/dev_harnesses/run_pr1_tests.py` instead, and the
  file parses (`ast.parse`) with all imports resolving. Run
  `python test/inductor/test_custom_lowering.py -k upsample_nearestnd` in the
  fork before pushing.
- **Rebase before sending.** Upstream HEAD moved `20e11d6` → `497054b` →
  `b1716d9` → `0303631` across three sessions. Both patches were last checked
  against **`0303631`**.

---

## 9. Session B: reading the source, and what it changed

Session A selected a fix by measuring four variants of my own devising. Session B
did the thing that should have come first: **read how the codebase already solves
this**, and read the contribution policy. Both changed the deliverable.

### 9.1 `ops.div_rn` already exists, and exists for exactly this

Session A's conclusion was that a fp32 runtime divide is wrong on CUDA (Triton's
approximate `div.full`) so the divide must be done in fp64 and narrowed. That
conclusion was correct about the *hazard* and wrong about the *remedy*: Inductor
has a dedicated op for it.

```
torch/_inductor/ops_handler.py:719
    def div_rn(self, x0: T, x1: T) -> T:
        """Division with round-to-nearest rounding mode.  Used for matching
        eager CUDA semantics where division uses IEEE round-to-nearest."""

torch/_inductor/codegen/triton.py:1517
    def div_rn(x, y):
        """Always uses triton.language.div_rn for float32 inputs to match eager
        CUDA behavior."""              # <- unconditional, no config flag

torch/_inductor/codegen/common.py:1027
    def div_rn(x, y):
        # for backends that don't override this, just use regular div
        return ops.truediv(x, y)       # <- CPU is correctly rounded anyway
```

Full coverage: an `OpsHandler` contract, a Triton implementation, a CPU/common
fallback, and a dtype rule in `dtype_propagation.py:310`. It also has four
existing callers, and one of them documents *precisely* the bug I measured on the
H100:

```
torch/_inductor/lowering.py:7388  (in _floor_div_floating)
    # Use div_rn (IEEE round-to-nearest) instead of truediv here because
    # Triton's default division uses an approximate reciprocal, which can
    # produce a result slightly below the true quotient and cause floor()
    # to round down by one.
    return floordiv(a, b) if both_integer else floor(_div_rn(a, b))
```

Divide, then floor, one ulp low, off-by-one index — the same failure, already
diagnosed and already solved upstream. `upsample_nearestnd` does divide-then-floor
too and never got the same treatment.

**Consequence:** the fp64→narrow construction is replaced by `ops.div_rn`. It is
the idiomatic call, needs no explanation of a novel technique in review, and
inherits any future backend work on rounding modes.

### 9.2 Are the two numerically the same? (measured, CPU, no GPU needed)

`../evidence/dev_harnesses/../ulp_proof.py` reasoning, run in session B over `i, o ∈ 1..512`:

```
(i,o) pairs where an approx reciprocal flips a floored index : 11695
f32-divide vs f64-divide-then-narrow, disagreements          : 0
```

Two results worth separating:

- **11695 pairs** is the generalized version of session A's single H100
  observation. It is computed in numpy on CPU, so it is reproducible anywhere and
  is the number quoted in the PR. Examples: `3→135` at index 45 (1 vs 0),
  `3→291` at index 97 (0 vs 1).
- **0 disagreements** between a direct fp32 divide and fp64-then-narrow means
  session A's fix and session B's fix compute the identical value. `div_rn` is
  strictly simpler at equal numerics — a clean domination, not a tradeoff.

Session A's CUDA result stands as the *demonstration* that this matters on real
hardware (5/30 cases wrong, 448→192 mismatching 7917/110592 elements), and is
labelled as session A throughout.

### 9.3 The fix got smaller

|  | session A | session B |
|---|---|---|
| `lowering.py` | +43 / −4 | **+24 / −2** |
| hunks | 2 (incl. rewriting `fn`) | 2 (`fn` untouched) |
| new concepts | fp64 divide + narrow, parallel `runtime_divisors` list | `ops.div_rn`, existing precedent |

The parallel-list bookkeeping is gone: `inv_scales` now holds *either* a concrete
float *or* the symbolic `o`, and `scale_fn` branches on `isinstance(scale,
sympy.Expr)`. `fn` is byte-identical to upstream, so the diff is two tight hunks.

Verified on the real installed lowering (`../evidence/dev_harnesses/check_installed_fix.py`,
CPU):

```
A. symbolic output size : 24/24 bit-exact, 0 wrong, 0 crashed
B. 20 distinct output sizes: 1 graph(s), 20/20 bit-exact
C. ATen interpolate paths: 12 match, 0 mismatch
VERDICT: PASS
```

And the generated CPU kernel confirms `div_rn` degrades to a plain divide there,
as `common.py` promises — `auto tmp3 = tmp0 / tmp2;` for `32.0 / ks1`.

### 9.4 The policy forced a split

`CONTRIBUTING.md` and `AI_POLICY.md` (both read in session B) impose three things
session A's single big PR violated:

1. *"Only PRs that address issues labeled actionable will be considered for
   review"* — so an **issue comes first**, and waits for a maintainer's label.
2. *"If the comments, issues or PRs you send are low quality or consistently
   overly verbose compared to what is expected, your contributions will not be
   accepted anymore."* — session A's PR body was ~230 lines. Cut to ~60.
3. *"AI-generated code can sometimes be overly complex, indirect, inconsistent…
   Before submitting, simplify where possible and make sure the core change is
   easy for maintainers to review."* — the fp64 construction was exactly that.

Also: *"AI-generated content in comments, issues, or PRs must be clearly disclosed
and contained"*, and *"We do not accept contributions created by fully autonomous
agents."* So the issue and PR text must be posted by a human who has read the
code and can defend it — `ISSUE.md` / `PR1_BODY.md` / `PR2_BODY.md` are drafts to
review and rewrite in your own voice, not paste blindly.

The upsample fix and the `ops.constant` contract check are **two concerns**, so
they are now two PRs:

| | PR #1 | PR #2 |
|---|---|---|
| change | `lowering.py` +24 −2 | `virtualized.py` +23 |
| tests | +122 −1 | +25 |
| claim | fixes the reported crash | enforces an existing convention |
| risk | low — measured no-op elsewhere | debatable; may be declined |

PR #2 stacks on PR #1 (it extends the same test file), which is why
`git apply --check` of PR #2 against pristine `main` fails and against
`main`+PR #1 succeeds. That is intended, not a broken patch.

### 9.5 PR #2 was reimplemented too

Session A put the check inside `OpsWrapper._default`, i.e. on the path *every*
`ops.*` call takes. Session B moved it to an explicit `OpsWrapper.constant`, so
only `ops.constant` pays. Two things had to be verified, because
`DefaultHandler._init_cls()` metaprograms method names onto the class and could
have clobbered the override:

```
1. OpsWrapper.constant is our override: True      # survives _init_cls()
2. symbolic value raises TypeError: True
3. no false positives on concrete values (incl. sympy numbers)
```

Blast radius, **diagnostic alone**, 25 lowerings including the four that
legitimately pass computed values (`full_like`, `add(alpha=)`, `addcmul`,
`addcdiv`) plus `floor_divide`:

```
static sweep : 25 ok, 0 bad, of 25
dynamic sweep: 25 ok, 0 bad, of 25
```

Cost, measured: **124 ns** per `ops.constant` call (~1.2 ms per 10k calls). PR #2's
body raises the gate-it-behind-config question explicitly rather than pretending
the cost is zero.

**A crash that was not ours.** The sweep initially core-dumped. Isolating each
case showed `conv2d` was the culprit — and that it dumps core identically with
both patches reverted, and even in **pure eager** with no `torch.compile` at all:

```
$ python -c "import torch,torch.nn.functional as F; F.conv2d(torch.randn(2,3,16,16), torch.randn(4,3,3,3))"
Segmentation fault (core dumped)
```

A broken CPU convolution path in this container, unrelated to any change here.
It is excluded from the sweep and recorded here so the exclusion is not mistaken
for cherry-picking. (Session A's equivalent sweep reported 19/20 with conv2d
"failing" for a *different* reason — a random weight inside the lambda — which is
its own lesson about trusting a harness.)

### 9.6 What must still be done before pushing

Superseded by **§10.6** — session C closed the GPU verification and added a third
PR. Keep reading.

Do not claim a merged fix until it merges. Accurate now: *"filed an issue and
submitted a fix + regression tests to PyTorch for a latent type-contract violation
in an Inductor lowering."*

---

## 10. Session C: GPU verification, a performance surprise, and a third bug

Container: **`tan-2gpu-chip-0-5`, 2× H100 80GB HBM3, driver 575.57.08, CUDA 13.0**,
198-CPU quota, `torch 2.13.0+cu130`. Everything on NFS (`prwork/`) survived the
reschedule; `/tmp` from session A was long gone, which is why the harnesses now
live in `../evidence/dev_harnesses/`.

First, two container facts worth recording because they contradict earlier notes:

- **cu130 wheels see the H100.** `torch.cuda.is_available()` is True,
  `torch.version.cuda == 13.0`. Third session in a row where this held, against an
  old note claiming cu13 falls back to CPU.
- **`conv2d` does *not* segfault here.** Session B recorded eager `F.conv2d`
  core-dumping on `hu-tanngo-lv`; on this node it runs fine. That confirms the
  session-B diagnosis (per-container fault, not a torch bug) and vindicates
  excluding it rather than chasing it.

### 10.1 The open verification, closed

`../evidence/dev_harnesses/triton_verify.py`, on the H100:

```
torch 2.13.0+cu130 | gpu: NVIDIA H100 80GB HBM3
1. generated Triton contains 'div_rn': True
      tmp2 = triton.language.div_rn(tmp0, tmp1)
      tmp11 = triton.language.div_rn(tmp0, tmp10)
2. GPU bit-exact vs eager: 30/30  wrong=0 crashed=0
```

The 30 cases are 15 shape pairs × {`nearest`, `nearest-exact`}, `atol=rtol=0`, and
include the pairs where session A's plain-fp32 variant was **wrong** (448→192,
384→363, 512→300). So the entire numerics argument now rests on a GPU measurement
rather than an inference from the source.

Full end-to-end on GPU (`../evidence/dev_harnesses/check_installed_fix.py --device cuda`):

```
A. symbolic output size : 24/24 bit-exact, 0 wrong, 0 crashed
B. 20 distinct output sizes: 1 graph(s), 20/20 bit-exact
C. ATen interpolate paths: 12 match, 0 mismatch
VERDICT: PASS
```

And the strict-no-op claim, this time against the **shipped patch** rather than a
re-implementation — hashing normalised `run_and_get_code` output with the patch
applied and reverted: **12/12 byte-identical**.

### 10.2 The performance surprise, and what it actually was

Benchmarking one 8×32×512×512 → 2048² call on the H100
(`../evidence/dev_harnesses/perf_attrib.py`, 200 iters after warmup):

```
  eager F.interpolate        2.701 ms
  guard_int                  1.587 ms
  div_rn                    12.723 ms
  truediv                   12.696 ms
```

**`div_rn` is 8× slower than guarding.** That is a real cost and it had to be
chased, not explained away. Two things resolve it:

1. **It is not the rounding mode.** Swapping `div_rn` for a plain (approximate)
   `truediv` gives 12.696 ms — a 0.2% difference. The `div_rn` calls are two
   loop-invariant scalar ops; they are free.
2. **It is the dynamic shape.** The generated kernels say it plainly:

   ```
   guard_int:  x1 = ((xindex // 2048) % 2048)     # constant divisors
               xmask = tl.full([XBLOCK], True, tl.int1)
   div_rn:     x1 = ((xindex // ks1) % ks0)       # runtime integer division
               xmask = xindex < xnumel            # real bounds check
               + tl.where negative-index wrapping
   ```

   Folded sizes let Triton strength-reduce the index decomposition and drop the
   mask. Keeping the output shape symbolic is what costs 8×, and that is inherent
   to *not specializing* — any fix that avoids the recompile pays it.

**The tradeoff, measured end to end** (`../evidence/dev_harnesses/tradeoff.py`, 12 distinct output sizes
× 5 iterations, including compile time):

| | total | graphs |
|---|---|---|
| guard the divisor | 7.09 s | 8, then **eager fallback** |
| defer the division | **0.81 s** | **1** |

At a single fixed output size the ordering reverses once compilation amortizes:
crossover ≈ **213 iterations**. Below it, deferring wins; above it, the folded
constant wins.

**Why this does not change the design, but does change the PR body.** The deferred
path only fires when the output size is symbolic — which today means the compile
*crashes*. Verified: on every ATen `interpolate` path the lowering is not reached
at all (0 hits), so no existing caller can regress into the slow kernel. The fix
converts "does not compile" into "compiles, 8× slower than a specialized kernel
would be," and a caller who wants the fast kernel can guard the size itself. PR #1
now states the 1.59-vs-12.7 number, the attribution, the crossover, and offers to
flip the choice. Hiding a measured 8× and letting a reviewer find it would be the
fastest way to lose the review.

### 10.3 A third bug: the same asymmetry in the backward lowering

Auditing sibling lowerings for the same defect class turned up
`upsample_nearest2d_backward` (`lowering.py:6626`), the **seventh** upsample
lowering and one I had not examined:

```python
inp_h = V.graph.sizevars.guard_int(inp_h)   # guarded
inp_w = V.graph.sizevars.guard_int(inp_w)
*_batch, out_h, out_w = input_size          # NOT guarded
if inp_h % out_h == 0 and ...               # eager Python %
h_kernel_max = ceildiv(inp_h, out_h)        # -> sympy FloorDiv
```

`h_kernel_max` becomes a `range()` bound in `_adaptive_pooling_fn`. Measured on
the H100 (`../evidence/dev_harnesses/probe_backward_bug.py`):

```
  dynamic=None : OK bit-exact=True
  dynamic=False: OK bit-exact=True
  dynamic=True : InductorError: TypeError: 'FloorDiv' object cannot be interpreted as an integer
```

with the traced values confirming the cause:

```
  inp = (64, 64)                       # guarded, concrete
  out = (s28, s28)                     # unguarded, symbolic
  kernel_max_h = ((s28 + 63)//s28)     # FloorDiv -> range() rejects it
```

**Same defect class, opposite correct fix.** Here the quotient is a Python loop
bound consumed at lowering time, so there is nothing to defer it to — guarding is
the only option. That asymmetry with PR #1 is called out in the patch comment and
the PR body so the two do not read as inconsistent.

Verified (`../evidence/dev_harnesses/backward_fix_test.py`, H100): **baseline 0/6 bit-exact, 6/6 crashed
→ fixed 6/6 bit-exact, 0 crashed**, covering both the exactly-divisible path (which
routes to `avg_pool2d`) and the general adaptive-pooling path. No-regression:
**20/20** autograd gradient comparisons through `F.interpolate` remain
bit-identical, and the static direct backward op is unchanged.

**Reachability, same story:** autograd does not reach it — 0 lowering hits across
`F.interpolate` and `nn.Upsample` backward, `size=` and `scale_factor=`, static and
dynamic, with input spatial dims marked dynamic. Latent, not user-visible.

### 10.4 Sibling candidates that turned out clean

Worth recording the negatives, since "I checked and it's fine" is part of the audit:

| site | verdict |
|---|---|
| `adaptive_avg_pool2d` (`lowering.py:6399`) | exact integer `FloorDiv` on indices — no float scale, correct by construction |
| `fractional_max_pool` (`6534`) | divides then `trunc`s, but computes in **float64** — already immune to the Triton fp32 issue |
| `avg_pool2d` backward (`7000`, `7204`) | `truediv` not followed by a floor/index — no index hazard |
| `upsample_bilinear` / `bicubic` | no `upsample_nearestnd`-style unguarded-divisor path; backwards are `make_fallback` |

So the class is real but small: two live instances, both now fixed.

### 10.5 Adversarial checks the PR now cites

`../evidence/dev_harnesses/stress_configs.py` on the H100, all bit-exact:

- **configs:** default, `cpp_wrapper=True`, `triton.cudagraphs=True`,
  `max_autotune=True`, `eager_numerics.division_rounding` **both** True and False
  (proving no dependence on that flag), `fallback_random=True`.
- **dtypes:** fp16, bf16, fp64 bit-exact. `int32` fails in *eager*
  (`"upsample_nearest2d_out_frame" not implemented for 'Int'`), so it is not our path.
- **ranks:** 1-D and 3-D `nearest` through the same lowering, symbolic, bit-exact.

### 10.6 Send order (current)

Three PRs, two issues, in this order:

1. **File `ISSUE.md`** → wait for `actionable` → **PR #1** (`0001-*.patch`, the
   forward fix). Repro verified to raise the stated error on stock torch.
2. **File `ISSUE3.md`** (link as related) → wait for `actionable` → **PR #3**
   (`0003-*.patch`, the backward fix, +7 −1). Its code is independent of PR #1;
   only the test file overlaps.
3. **After PR #1 lands**, raise the `ops.constant` contract check on the issue
   thread; if welcomed, send **PR #2** (`0002-*.patch`).

For each: sign the CLA, fork, **rebase** (HEAD moved `20e11d6` → `497054b` →
`b1716d9` → `0303631` → `7337468` across four sessions), `git am`, run the test,
**revert only the code hunk and re-run to watch it fail**, `lintrunner -a`, push.
Leave Reviewers empty; label `module: inductor`.

Still not done here: **`lintrunner -a`** (no PyTorch build on this box) and
executing PyTorch's own test files (not authorised; bodies validated verbatim via
`../evidence/dev_harnesses/run_pr1_tests.py` and the `t3.py` equivalent, and the file `ast.parse`s clean
with all imports resolving).

---

## 11. Session D: reviewing the series as a PyTorch reviewer would — one blocker found

Hardware re-probed: `tan-2gpu-chip-0-6`, **2× H100 80GB**, driver 575.57.08, CUDA 13.0,
**198-CPU cgroup quota** (`nproc` says 256 — the host count, not the budget). torch
2.13.0+cu130, python 3.12.14.

The goal this session was adversarial: read the three patches the way an Inductor reviewer
would, and try to break them on both CPU and CUDA. One **blocker-class defect** surfaced in
PR #1, plus two lint-level issues. All are fixed; the patches are regenerated.

### 11.1 BLOCKER — PR #1 type-punned `inv_scales`, and the two sites disagreed

The patch makes `inv_scales[i]` hold **one of two different things**:

- `i / o` — an *inverse scale* (a float), when `o` is concrete, or
- `o` itself — a *deferred output size*, when `o` is symbolic.

`scale_fn` then has to recover which. The patch chose the predicates independently:

| site | predicate |
|---|---|
| `inv_scales` (chooses) | `isinstance(o, sympy.Expr) and not o.is_number` |
| `scale_fn` (recovers) | `isinstance(scale, sympy.Expr)` ← **missing `is_number`** |

The two disagree on exactly one class of value: a **concrete `sympy` number**.
`sympy.Integer(64)` is a `sympy.Expr` whose `is_number` is `True`, so site 1 folds it to
`i / o = 0.5`, and site 2 then misreads that `0.5` as an output size and computes
`div_rn(i_size, 0.5)` = **64.0** instead of 0.5 — the inverse scale inverted and scaled by
`i²`.

It does **not** crash. `indirect_indexing(..., check=False)` suppresses the bounds check, so
the kernel reads far outside the input:

```
448 -> 192, index 191:  correct floor(191 * 448/192) = 445   (input dim 448)
                        emitted floor(191 * 448)     = 85568  (~191x past the buffer)
```

**Measured, all on this box (`/tmp/prove_bug.py`, `/tmp/sweep.py`):**

| state | concrete-`sympy` size, bit-exact vs eager |
|---|---|
| stock `main` | **8/8 CPU, 8/8 CUDA** — correct today |
| PR #1 as drafted | **0/8 CPU, 0/8 CUDA** — e.g. 12285/12288 elements wrong at 32→64; 110589/110592 at 448→192 |
| PR #1 + one-clause fix | **8/8 CPU, 8/8 CUDA** |

So it is a **silent wrong-numerics regression introduced by the patch** — the worst failure
class for an Inductor review, and worse than the crash the PR set out to fix. At the larger
ratios the out-of-range reads also **segfaulted the interpreter** (exit 139), which is how it
was first noticed: a sweep died mid-run rather than reporting.

**Reachability is the same as the bug the PR fixes** — the direct-lowering callers the PR
itself introduces and tests. ATen never passes a `sympy` number here (probed: `dynamic=False`
gives `int`; guard-specialized, algebraically-cancelling, and `torch.export` paths all
resolve to `int` before the lowering). But the PR's own stated audience is direct callers of
`upsample_nearestnd`, so the defect sits squarely inside the surface the PR adds.

**The fix — align the predicates** (`torch/_inductor/lowering.py`):

```python
if isinstance(scale, sympy.Expr) and not scale.is_number:
```

with a comment stating the coupling, so the next editor does not re-split them. Also added
`test_upsample_nearestnd_concrete_sympy_output_size`, which covers `int` and `sympy.Integer`
and has the right signature: **PASS on stock, FAIL on the patch as drafted, PASS with the
fix.**

**Note PR #2 does not catch this** — verified by running both together. The bad value is
`0.5`, a concrete float, so the `ops.constant` contract check correctly accepts it. The two
PRs are genuinely independent; neither substitutes for the other.

### 11.2 Lint — two `ruff` E303 violations in the test file

`git am` of PRs #2 and #3 left **two blank lines inside the class body** before
`test_upsample_nearest2d_backward_symbolic_input_size` and three before the module-level
`if __name__`. `lintrunner` would have flagged both. Fixed; the file `ast.parse`s clean, ends
with a newline, and a blank-run scan reports 0 issues.

### 11.3 Re-validation of the corrected series (all three PRs installed together)

| check | result |
|---|---|
| symbolic output size — the PR's target (8 pairs × 2 modes × CPU/CUDA) | **32/32 bit-exact** |
| concrete `sympy.Integer` / `int` sizes — the regression above | **32/32 bit-exact** |
| ATen `F.interpolate` no-regression (2 modes × `size=`/`scale_factor=` × static/dynamic × CPU/CUDA) | **16/16 bit-exact** |
| **byte-identical generated code** on existing ATen paths, stock vs fixed | **16/16** (strict no-op) |
| generated Triton contains `triton.language.div_rn` | **yes**, H100 |
| GPU shape×mode matrix (`../evidence/dev_harnesses/triton_verify.py`) | **30/30 bit-exact** |
| non-default configs (`cpp_wrapper`, cudagraphs, `max_autotune`, `division_rounding` both ways, `fallback_random`) | all bit-exact |
| dtypes fp16 / bf16 / fp64, and 1-D / 3-D `nearest` | all bit-exact |
| PR #1's own two tests (`../evidence/dev_harnesses/run_pr1_tests.py`) | 2 pass / 0 fail, as expected |
| PR #2 contract check: rejects `32*s0`, accepts `int`/`float`/`sympy.Integer`/`sympy.Float` | as specified |
| PR #3 direct op (4 pairs × CPU/CUDA) | **8/8 bit-exact** |
| PR #3 autograd no-regression through `F.interpolate` (2 modes × static/dynamic × `size=`/`scale_factor=` × CPU/CUDA) | **16/16 bit-identical** |

The codegen-hash comparison uses a normalizer (AOT ID, addresses, `/tmp` paths, cache hashes)
and was self-checked stock-vs-stock first — without that, the per-process `# AOT ID` counter
makes every variant look different.

### 11.4 Regenerated patches

Rebuilt as three self-consistent commits on `main` @ **`b1716d9`**, each still one concern.
`git apply --check` clean in sequence against pristine base.

| | PR #1 | PR #2 | PR #3 |
|---|---|---|---|
| code | `lowering.py` **+29 −3** | `virtualized.py` **+23** | `lowering.py` **+7** |
| test | **+170 −1** | +23 | +35 |

Note PR #1's `import sympy` in the test file now belongs to PR #1 (it was PR #2's before);
applying #2 or #3 alone onto pristine `main` will therefore need that line. Apply in order.

### 11.5 Still not done here

`lintrunner -a` (no PyTorch build on this box) and executing PyTorch's own test files. The
verification above runs the test **bodies** against installed torch via `../evidence/dev_harnesses/run_pr1_tests.py`
and standalone harnesses, with `InductorTestCase` as the base class — plain
`unittest.TestCase` silently lacks `assertEqual(atol=...)` and reports a spurious error.

---

## 12. Session E: second adversarial pass — PR #2 hardened, three claims stress-tested

Same box as §11 (`tan-2gpu-chip-0-6`, 2× H100 80GB, driver 575.57.08, CUDA 13.0, 198-CPU
quota, torch 2.13.0+cu130). This pass attacked the two PRs that got least scrutiny (#2, #3)
and tried to falsify PR #1's supporting claims. **No new blocker.** One real design flaw in
PR #2, two scope limits that needed disclosing, and four claims now measured rather than
argued.

### 12.1 PR #2 was the only op in Inductor that did not unwrap its argument

`OpsWrapper._default` unwraps every argument (`OpsWrapper._unwrap`) before dispatching. The
drafted `OpsWrapper.constant` bypassed `_default` entirely — `return
OpsWrapper._wrap(_ops.constant(value, dtype))` — so it became **the only op that skips
`_unwrap`**. Demonstrated with a recording handler: `ops.constant(OpsValue(x))` delivered an
`OpsValue` to the handler, where `ops.add(OpsValue, OpsValue)` correctly delivered raw values.

Not reachable today — all **112** `ops.constant` call sites pass plain Python scalars (surveyed
by grep) — but it is exactly the kind of gratuitous inconsistency an Inductor reviewer asks
about, and it would have been a latent trap for the next caller.

**Fixed by unwrapping first, then delegating:**

```python
def constant(self, value: Any, dtype: torch.dtype) -> Any:
    value = OpsWrapper._unwrap(value)
    if isinstance(value, sympy.Expr) and not value.is_number:
        raise TypeError(...)
    return self._default("constant", (value, dtype), {})
```

**Both orderings are load-bearing**, and the commit message now says so: checking *before*
`_unwrap` would reject an `OpsValue` wrapping a concrete number; not delegating would keep the
divergence. Also dropped `@staticmethod` (now needs `self`).

### 12.2 PR #2's cost, measured instead of hand-waved

The drafted body admitted "I have not profiled a large-model compile." Now measured:

| | |
|---|---|
| `ops.constant` calls in one compile of a constant-heavy graph | **160** |
| added predicate | ~109 ns/call |
| added `_unwrap` | ~270 ns/call |
| **total added per compile** | **~61 µs = 0.0017% of compile time** |
| generated code, stock vs patched, **24 constant-using ops × CPU/CUDA × static/dynamic** | **96/96 byte-identical**, 0 compile failures either side |

The 96-case sweep is the stronger claim: it covers `full_like`, `add(alpha=)`, `addcmul`,
`addcdiv`, `clamp`, `leaky_relu`, `gelu`, `elu`, `hardtanh`, `softplus`, `avg_pool2d`,
`max_pool2d`, `adaptive_avg_pool2d`, `softmax`, `logsumexp`, `var_mean`, `cumsum`, `where`,
`pow`, `erf`, chained sigmoid/tanh, nearest and **bilinear** interpolate, and `layer_norm`.

Also re-ran a 15-case legit-caller sweep including unbacked symints from `.item()`: no
behaviour change. One case (`index_put` with a bad broadcast in the harness) fails — **verified
pre-existing by removing PR #2 and re-running**, so not PR #2's.

### 12.3 PR #1's `div_rn` justification — two gaps closed by reading ATen

Read `aten/src/ATen/native/UpSample.h` directly rather than trusting the earlier note:

- `compute_scales_value<float>` really is `static_cast<float>(input_size) / output_size`
  (line 261), and `nearest_idx` / `nearest_exact_idx` both instantiate it at **`float`**
  (lines 357, 367). The PR's claim is exact.
- **Rounding-mode gap:** eager uses `floorf`, the lowering uses `to_dtype(int32)`
  (truncation). These differ for negatives — but `dst_index * scale` is always non-negative
  here, so they agree. Worth stating rather than leaving for a reviewer to spot.
- **Clamp gap:** eager applies `min(..., input_size - 1)`; the lowering passes
  `check=False` to `indirect_indexing`, so it has no clamp. Swept **all 262144 `(i, o)` pairs
  in 1..512 × both modes: the clamp never binds** for a correctly-rounded scale (0 cases). So
  omitting it is safe — but only *because* the scale is correctly rounded, which is a second,
  independent argument for `div_rn` over an approximate reciprocal.
- Confirmed `float32(i)/float32(o)` and `float64(i)/float64(o)` narrowed to fp32 are
  **bit-identical for all 262144 pairs**, which retires the earlier fp64-narrow design as
  strictly unnecessary rather than merely more verbose.
- Eager's two `nearest_idx` fast paths (`o == i` → identity, `o == 2i` → `idx >> 1`) that the
  lowering does not replicate: checked across i ∈ 1..2048 — **0 disagreements** with the float
  path. No hidden divergence.

### 12.4 Two scope limits, now disclosed in the commit messages

Both were true of the drafted PRs and unstated — the kind of thing that reads as overclaiming
when a reviewer finds it:

- **PR #1 only removes the *output*-size specialization.** `i_sizes` is still guarded
  (pre-existing), so a caller varying its *input* size still hits `recompile_limit`. Measured
  on both devices: **1** lowering call for 20 varying output sizes, **8** for 20 varying input
  sizes. The earlier fuzz tripping `recompile_limit` was this, not a regression — separated by
  holding one dim fixed at a time.
- **PR #3 guards the locals but leaves `ranges=list(input_size)` symbolic.** Traced: `ranges`
  receives `['1', 's71', 's28', 's13']` while `out_sizes` are guarded ints. Safe because the
  guard makes symbol == value within a compilation — verified by reusing one compiled callable
  across six differing `input_size` values on CPU and CUDA: each recompiles, all bit-exact.

### 12.5 PR #1's one-graph claim, pushed past the cliff

The shipped test uses 5 output sizes, but `recompile_limit` defaults to **8** — so the test
cannot actually demonstrate what the PR body claims. Pushed to **20 distinct output sizes**:
**1 lowering call, 20/20 bit-exact, on CPU and CUDA.** The claim holds; the test just
under-demonstrates it. Left the test at 5 (keeping CI cheap) since §12.4's measurement is now
in the commit message.

### 12.6 Re-validation of the final series

| check | result |
|---|---|
| **wide fuzz**: 276 shape pairs × 2 modes × CPU/CUDA | **1104/1104 bit-exact** (`atol=rtol=0`) |
| symbolic output size (the PR target) | 32/32 bit-exact |
| concrete `sympy.Integer` / `int` sizes (§11's blocker) | 32/32 bit-exact |
| ATen `F.interpolate` no-regression | 16/16 bit-exact |
| byte-identical codegen, upsample paths | **16/16** |
| byte-identical codegen, 24 constant-using ops | **96/96** |
| GPU: Triton contains `div_rn`, shape×mode matrix | yes; **30/30** |
| PR #3 direct op / autograd no-regression | 8/8 · 16/16 |
| 20 output sizes → one graph, CPU and CUDA | 1 call, 20/20 |

### 12.7 Final patch series

Rebuilt on `main` @ **`b1716d9`**, `git apply --check` clean in sequence, resulting tree
verified identical to the branch.

| | PR #1 | PR #2 | PR #3 |
|---|---|---|---|
| code | `lowering.py` **+29 −3** | `virtualized.py` **+23 −1** | `lowering.py` **+7** |
| test | **+170 −1** | +23 | +35 |

Still not done here: `lintrunner -a` (no PyTorch build) and executing PyTorch's own test
files. A clean A/B of *total compile wall time* for PR #2 was attempted and abandoned —
compiles on this box ran past a 2-minute budget under contention, so the cost is bounded by
call-count × per-call cost (§12.2) instead of measured end-to-end. That substitution is
disclosed here rather than presented as a wall-clock result.
