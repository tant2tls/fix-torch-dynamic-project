# A100 verification results — 2026-08-26

Hardware: 1x NVIDIA A100-SXM4-80GB, driver 575.57.08, 198-CPU cgroup quota
(host `nproc` reports 256 — not the budget). torch 2.13.0+cu130 (release wheel,
git cf30153c), triton 3.7.1, python 3.12.14. CPU checks: torch 2.13.0+cpu.

State is toggled by `a100/state.py`, which does exact-string replacement in both
directions and refuses to run against a file matching neither form.

## 1. New tests: 6/6 pass on A100 (CPU + CUDA)

## 2. Guard matrix — every test fails with only its own fix reverted

| state | forward | concrete-sympy | one-graph | const-reject | const-accept | backward |
|---|---|---|---|---|---|---|
| all ON  | OK | OK | OK | OK | OK | OK |
| PR1 off | **FAIL** | OK | **FAIL** | – | – | OK |
| PR2 off | OK | – | – | **FAIL** | OK | – |
| PR3 off | OK | – | – | – | – | **FAIL** |

## 3. Strict no-op: 63/63 real ATen upsample cases bit-identical ON vs OFF

`a100/noop_digest.py` over F.interpolate / nn.Upsample, size= and scale_factor=,
1d/2d/3d, static and dynamic, forward and real-autograd backward, plus
adaptive_avg_pool2d. **Compiled-result hashes differ in 0/63 cases.**

⚠️ The generated-Triton-source hash is NOT usable as evidence: it differs in
50/63 cases between two runs of the *identical* state (it embeds cache paths /
kernel names). Control run: `ctrl_OFF_a.json` vs `ctrl_OFF_b.json` — 0 result
differences, 50 code differences. Only the result hashes are load-bearing.

## 4. Test-harness bug found and fixed (would have failed CI)

`test_upsample_nearestnd_symbolic_output_size` registered the same op name
`test_ups_ops::ups2d` for both `exact=False` and `exact=True`. `exact` is captured
in the lowering closure, invisible to the FX graph cache key, so the second
iteration reused the first's kernel: outputs byte-identical across two modes that
must differ, 7392/21000 elements wrong (35.2%).

Proof (`a100/cache_collide.py`):

| variant | nearest | nearest-exact | outputs identical across modes |
|---|---|---|---|
| same op name, cache ON | ok | **MISMATCH 7392/21000** | **True** |
| distinct op names, cache ON | ok | ok | False |
| same op name, cache OFF | ok | ok | False |

`test_upsample_nearestnd_concrete_sympy_output_size` had the same defect and it
was worse: both iterations build the same graph, so the collision was invisible
and the test passed **without ever lowering the sympy.Integer sizes** — a false
pass on exactly the path that caused the earlier wrong-numerics blocker.

Fixed by making the op name unique per iteration.

## 5. PR1 refactored to remove a type-punned list

`inv_scales` previously held *either* an inverse scale *or* an output size, so the
predicate `isinstance(o, sympy.Expr) and not o.is_number` had to be duplicated at
two sites and kept in sync; drift between those two sites is what caused the
earlier wrong-numerics blocker. Now it holds `float | None`, `None` meaning
"deferred", and `scale_fn` takes `o_size` as its own argument. One predicate, one
site. Guard matrix and 20/20 shape sweep re-verified after the refactor.

## 6. `ops.div_rn` choice independently validated against the exact ATen model

`a100/aten_ref_probe.py` models `UpSample.h` exactly (float32 scale, float32
multiply, floor, clamp) and compares Triton's two divides over 9 ratios / 3305
output coordinates:

| divide | coordinates disagreeing with eager |
|---|---|
| `I / O` (approximate, what inductor emits today) | **45** |
| `tl.math.div_rn(I, O)` | **0** |

## 7. Pre-existing bugs found, NOT caused by these PRs (verified identical with all three reverted)

- **Wrong pixels** from plain `F.interpolate(mode="nearest")` under `dynamic=True`
  on **CUDA only** (CPU exact, static exact). Off-by-one in the source index:
  448->192 7/192 rows, 384->363 2/363, 37->74 36/74, nearest-exact 384->363 2/363.
  Cause is in the *decomposition*, not the lowering these PRs touch: it emits
  `(ks0 / 74).to(tl.float32)` — an approximate on-device divide — where static
  folds the scale to a literal. → **Issue A**, the strongest finding.
- `_adaptive_avg_pool2d` under `dynamic=True`: "cannot determine truth value of
  Relational". 3/3 cases, identical with PRs on and off. → Issue E (hold).
- `test_reused_inline_asm_realized` fails with all PRs reverted — pre-existing,
  unrelated.

## 8. Regression suites, both states, identical results

| suite | PRs OFF | PRs ON |
|---|---|---|
| `inductor/test_indexing.py` | 56 OK | 56 OK |
| `inductor/test_optimize_indexing.py` | 15 OK | 15 OK |
| `inductor/test_codegen_triton.py` | 16 OK (2 skip) | 16 OK (2 skip) |

Note: the suites must be run from a checkout matching the *wheel's* commit
(cf30153c, release/2.13), extracted to /tmp/t213. The `prwork/pytorch` checkout is
`main` @ b1716d9 and its tests import symbols the 2.13 wheel does not export
(`assert_size_stride_grouped`), which fails at import, not at test time.

`test_torchinductor_dynamic_shapes.py` (the full suite, thousands of GPU
compilations) was still running at hand-off with **0 FAIL / 0 ERROR** across
346KB of output. Let it finish and record the final line before sending.

## 9. Issue repros verified against genuinely stock torch

Run with all three patches reverted (`a100/verify_issue_repros.py`):

| Issue | stock behaviour | matches issues.md |
|---|---|---|
| B `upsample_nearestnd` | `NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>` | yes |
| C `ops.constant` | accepts the symbolic value silently (no error — the bug) | yes |
| D backward | `TypeError: 'FloorDiv' object cannot be interpreted as an integer` | yes |

⚠️ The CPU env had been left in a mixed state (`pr1 = INCONSISTENT`) from earlier
work, which made B appear to pass. Restored from `.orig` before verifying. Always
run `state.py status` before drawing a conclusion from either env.

## 10. Performance: the `div_rn` objection answered (A100)

Context: PR #164144 made inductor's `truediv` emit `div_rn` globally for eager
parity; it merged, was reverted repeatedly, and was finally gated behind
`TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING` (#165566) after a throughput regression
on B200 (#164301: ~6511 -> ~4692 GB/s). So "you added a div_rn" is the first
question a reviewer will ask.

**Isolated comparison — same lowering, same dynamic shapes, only the divide
differs** (`a100/perf_divrn.py`, 200 iters after 20 warmup):

| shape | div_rn us | truediv us | div_rn/truediv |
|---|---|---|---|
| 1x3x64x64 -> 128x128 | 93.5 | 96.8 | 0.965 |
| 1x3x256x256 -> 512x512 | 94.6 | 98.9 | 0.957 |
| 8x64x128x128 -> 256x256 | 706.1 | 718.0 | 0.983 |
| 4x128x256x256 -> 512x512 | 2917.5 | 2963.9 | 0.984 |

`div_rn` is **not slower** here — all four ratios are below 1.0, i.e. within
run-to-run noise and marginally faster.

**Why, structurally** (`a100/loop_invariant.py`, from the emitted Triton):

```python
tmp0  = tl.full([1], 128.0, tl.float32)
tmp1  = (ks0).to(tl.float32)
tmp2  = triton.language.div_rn(tmp0, tmp1)   # <- kernel args only
tmp3  = (x1).to(tl.float32)
tmp4  = tmp3 * tmp2
```

Both `div_rn` calls depend only on kernel arguments (`ks0`, `ks1`) and a literal —
never on `xindex`/`x0`/`x1`. The divisor is the symbolic output size, uniform
across the grid, so the divide is **loop-invariant**: one scalar op per program
launch, hoisted out of the vectorised body. #164144's regression came from a
*per-element* divide in a memory-bound elementwise kernel. Different regime.

This is also the honest limit of the claim: measured on A100 (SM 8.0) at four
shapes. It does not prove anything about B200, and the PR should say so.

## 11. Adversarial sweep of PR1 — 42/42 bit-exact, 0 problems

`a100/adversarial_pr1.py` — 21 configurations x {cpu, cuda}, written to try to break
the fix rather than confirm it. All bit-exact (atol=rtol=0) vs eager:

- **mixed symbolic/concrete dims** — only dim 0 dynamic; only dim 1 dynamic. The
  `None` sentinel is per-dim, so a `zip()` misalignment would only show up here.
- **explicit `scales_x` for one dim, symbolic size for the other** — the `1.0/scale`
  override must beat the deferred path. Verified it does: instrumenting the predicate
  shows `scale_dim0` reporting `explicit_scale_per_dim=(True, False)`.
- **nearest-exact** — the `ops.add(x, 0.5)` happens *before* the deferred multiply,
  incl. at the ULP-sensitive 448->192.
- **ULP-sensitive ratios** 448->192, 384->363, 37->74.
- **degenerate** — output size 1, input size 1, equal sizes, 8x upscale.
- **n != 2** — 1-D and 3-D with symbolic sizes.
- **dtypes** — float16, bfloat16 (scale math stays fp32).
- **non-contiguous** (transposed) input.
- **downsampling as well as upsampling.**

Coverage was verified, not assumed: instrumenting the predicate confirms
`defer_per_dim=(True, True)` for these cases, i.e. the `None` + `ops.div_rn` branch
actually ran rather than silently taking the concrete path.

## 12. The full dynamic-shapes suite: status at hand-off

`test_torchinductor_dynamic_shapes.py` is the suite that matters most for PR2 (a global
`ops.constant` contract change), because it exercises `ops.constant` across hundreds of
lowerings under dynamic shapes. It takes many hours on one A100.

**Three attempts, and the first two are NOT usable — recorded so they are not mistaken
for evidence:**

1. First run: died at *import* with
   `ImportError: cannot import name 'assert_size_stride_grouped'` — the `prwork/pytorch`
   checkout is `main` @ b1716d9 but the installed wheel is release/2.13 @ cf30153c. It
   still exited `rc=0`, i.e. **a fake pass**. Fix: run the wheel-matched tests from
   `/tmp/t213` (`git archive cf30153c test/inductor`).
2. Second run: invalidated because `state.py` was toggled for other checks *while it
   ran*, so it was compiling against a mixture of states. Never toggle during a long run.
3. Third run: two instances were accidentally launched and both wrote the same log file
   (interleaved output). Quarantined as `INVALID_two_writers.log`.

**Fourth run — the valid one:** `a100/logs/final_dynshapes_ON.log`, single instance
(verified: exactly one matching pid), against a **pinned copy of the venv** at
`/tmp/envpinned` (distinct inodes from `envcu130`, so `state.py` cannot reach it), all
three PRs ON, fresh cache dirs. **Still running at hand-off with 0 FAIL / 0 ERROR.**

**Before sending PR2, record its final `Ran N tests ... OK` line, and run the same suite
once with the patches OFF for comparison.** Until then the honest statement is "the
targeted suites pass in both states; the full dynamic-shapes suite was clean as far as it
ran," not "the full suite passes."

## 13. Systems-review pass: the questions a reviewer asks that the diff cannot answer

Four measurements added because each PR made a structural claim it had not measured.

### 13.1 PR1 — the rejected alternative, measured head to head

PR1 argues against `guard_int` on the output size. Same lowering, same sweep of 7
distinct output sizes, only the divisor handling differing
(`a100/pr3_why_guard.py` + the defer-vs-guard sweep):

| approach | lowering invocations for 7 sizes |
|---|---|
| deferred divide (shipped) | **1** |
| `V.graph.sizevars.guard_int(o)` | **7** |

The claim is now a comparison rather than an assertion.

### 13.2 PR2 — blast radius and cost of enforcing a global contract

`OpsWrapper.constant` governs **111 static call sites**. Instrumenting a 24-case
compile workload (pointwise, reductions, norms, softmax, pooling, upsample,
cat/cumsum/cast, matmul, static and dynamic) — `a100/pr2_blast_radius.py`:

| population | count | share |
|---|---|---|
| total `ops.constant` calls | 1157 | 100% |
| plain (non-sympy) values | 1144 | 98.88% |
| sympy, **concrete** | 13 | 1.12% |
| sympy, **symbolic** | 0 | 0% |

All 13 sympy values are `sympy.Integer` from `get_constant_or_index_expr` in
`var_mean_welford_`, reached via `layer_norm` (7) and `mean`/`var` (6). **An
`isinstance`-only check would have rejected all 13** — so `is_number` is
load-bearing, not defensive.

Cost: **1.23 s vs 1.25 s** (0.98×) for the whole workload with the check vs
without. It runs once per IR node at lowering time and never appears in the
generated kernel.

### 13.3 PR2 — the real justification: six hand-guards using TWO disagreeing predicates

The earlier framing ("several lowerings already dispatch on it by hand") understated
the case and cited the weaker example. There are **six** hand-guard sites in
`lowering.py`, and they do not agree:

- `isinstance(value, sympy.Basic)` — `_add_with_alpha_fma` (687), `tensor` (4261),
  `_full` (4409), `addcmul` (8577), `addcdiv` (8648)
- `isinstance(x, sympy.Expr) and not x.is_number` — `get_constant_or_index_expr` in
  `var_mean_welford_` (7576)

They diverge on exactly the sympy-but-concrete values:

| value | `sympy.Basic` | `Expr and not is_number` |
|---|---|---|
| `sympy.Integer(5)` | True | False |
| `sympy.Float(2.5)` | True | False |
| `sympy.Rational(1,2)` | True | False |

Both work today (index_expr accepts concrete values too), but only one expresses the
contract, and a new call site has a five-in-six chance of copying the wrong one.
PR2 adopts `var_mean`'s predicate. This reframes PR2 from "add a check" to "codify
the correct one of two predicates already in tree."

### 13.4 PR3 — why guarding is not the thing PR1 warns against, and a tolerance limit

Structural: the quotient is the `range()` bound the pooling window unrolls over, so
window size *is* the kernel body — 2×2 = 4 unrolled loads, 8×8 = 64. Different
kernels, not one kernel with a different argument. Nothing to defer.

⚠️ **Found while measuring this, and it changes what PR3 may claim:**
`upsample_nearest2d_backward` is **not** bit-exact vs eager for every ratio. grad
64×64 → input 8×8 differs on 155/192 elements, max|d| 3.8e-06.

Triaged before trusting anything (`a100/bwd_mismatch_triage.py`):

- **Identical with a static `input_size`** → not caused by PR3.
- **float64 drops it to max|d| ~7e-15** (from ~1e-5), scaling with machine epsilon →
  summation-order noise, not a wrong index (an index error is dtype-independent).
- Boundary is sharp and follows the *branch*, not the window: 64→9 is an 8×8 window
  and is bit-exact; 64→8 is an 8×8 window and is not. It tracks the exactly-divisible
  `avg_pool2d` reassociation.

All four pairs in PR3's test are inside the bit-exact regime, so `atol=rtol=0` is
meaningful — but the message now says so explicitly instead of implying the op is
bit-exact in general.

Also re-verified on this A100 rather than inherited: **20/20** gradient comparisons
through real `F.interpolate` autograd are bit-identical (nearest / nearest-exact,
`size=` / `scale_factor=`, static / dynamic).

## 14. The full dynamic-shapes suite COMPLETED — 2621 tests, 0 attributable failures

`a100/logs/final_dynshapes_ON.log`, all three PRs ON, single instance against the
pinned venv, wheel-matched tests (cf30153c) from `/tmp/t213`:

```
Ran 2621 tests in 4656.207s
FAILED (failures=1, skipped=523, expected failures=2)
```

The single failure is **pre-existing and unrelated**:

```
FAIL: test_unbacked_reduction_cpu (TestInductorDynamicCPU)
AssertionError: expected to fail, but actually passed
```

It is an inverted-xfail: the test hardcodes `expect_fail = device == "cpu" and not
IS_ARM64 and not cpp_wrapper` and calls `self.fail()` when the case *passes*. On this
box it passes, so the test fails. **Verified identical with all three PRs reverted**
(FAILED both ways, same assertion), so it is a stale expectation in the 2.13 test
file on this configuration, not a regression.

This is the evidence PR2 needed: it changes a contract on a surface of 111 call
sites, and the suite that most exercises `ops.constant` under dynamic shapes shows
**zero attributable failures across 2621 tests**.

⚠️ Two caveats to state honestly in the PR:
- Only the **ON** arm was run to completion at this scale. The OFF arm was verified
  for the failing test specifically (identical), not for all 2621.
- `/tmp` was cleared mid-session, destroying `/tmp/envpinned` and `/tmp/t213` *after*
  the run finished. The log survived because it was written to NFS. Re-extract with
  `git archive cf30153c test/inductor | tar -x -C /tmp/t213` to re-run. This is the
  second time `/tmp` volatility has bitten — keep long-run artifacts on NFS.

## 15. PRIOR-ART SEARCH — 2026-08-27. Four relevant upstream items found.

Searched `repo:pytorch/pytorch` across op names, error strings, and symptoms.
**None of the three PRs is a duplicate**, but two findings change the plan.

### 15.1 PR #117538 — MERGED — created the code PR1 modifies (not a duplicate)

"Fixed issue in upsample_nearestnd lowering with scales" (vfdev-5, merged Jan 2024,
closed #116848 "black images with inference_mode + diffusers SD").

It fixed the **explicit `scales_x` inversion**: after #113749 honoured `scales_x` but
stored it un-inverted, this PR stored `1 / scale` and renamed `scales` → `inv_scales`.
So it is the PR that introduced the very name PR1 edits. It did **not** touch symbolic
output sizes or `ops.constant`.

Worth citing in PR1's body as the origin of `inv_scales` — it shows the reviewer we
know the history. Also note the author's comment: *"I do not add any test here as
potentially we would remove the lowering for upsample_nearest2d as there is a
decomposition."* That is a maintainer-adjacent voice saying this lowering may be
**deleted**. Expect PR1 to draw "why fix code we plan to remove?" — the honest answer
is that it is still exported and reachable by out-of-tree backends, and the fix is
30 lines.

### 15.2 ⭐ Issue #175154 — OPEN, `high priority`, `PT2-Bug-Bash`, NO merged fix

> ⚠️ **This subsection originally said `actionable`. It is NOT — see §16.1, which
> corrects it from the GitHub API. The rest of the subsection stands.**

"[inductor/aot_eager] interpolate under torch.compile produces incorrect result when
mode is `nearest`" (maybeLee, Feb 2026, assigned to Akshat-Raj).

```python
F.interpolate(torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64),
              scale_factor=1.3, mode='nearest')
# eager    -> [[[[1., 2.]]]]
# compiled -> [[[[1., 1.]]]]
```

**Reproduced here on 2.13.0+cu130, and it is broader than reported:** fails on CPU
*and* CUDA, float32 *and* float64, and with **static** shapes. The report only
mentions CPU/float64.

**Is this the same bug as my Issue A? NO — measured.** `a100/same_bug_as_175154.py`:

| | stock | with my `sym_float` fix |
|---|---|---|
| #175154 (`scale_factor=1.3`, static, CPU) | wrong | **still wrong** |
| Issue A (`size=`, dynamic, CUDA, 448→192) | 7 rows wrong | **0 rows wrong** |

Two distinct defects in the same function:
- **#175154** = the `scale_factor=` branch, `isize / (isize * scales[d])`
- **Issue A** = the `size=` branch with a symbolic `isize`, `isize / osize` lowering
  to an approximate SymInt divide

**This is the single most valuable finding of the search.** #175154 is already
labelled **`actionable`** — the label PyTorch requires before it will review a PR.
So a fix for it needs no waiting period. And PR #184848 (below) tried and failed,
leaving it open and assigned.

### 15.3 PR #184848 — CLOSED UNMERGED — attempted #175154, failed CI with 23 failures

"Fix nearest upsample scale shortcuts in compile" (`agentic` label). Touched
`_decomp/decompositions.py`, `_inductor/lowering.py`, and a CUDA backward kernel.

From the diff: it **kept `scale = isize / osize` unchanged** and instead added
fast-path shortcuts (`osize == isize`, `osize == 2 * isize` via `output_index >> 1`)
plus changed the scales branch to `1.0 / scales[d]`. Reviewer karthickai requested
changes — *"forward now matches eager, but the grad no longer does."* Closed by the
author as part of "a cleanup of open agentic PRs"; **the linked issue remains open**.

Two lessons for us:
1. The line my fix changes is the one that PR left alone — no collision.
2. **A forward-only fix here breaks the gradient.** Any Issue A PR must include a
   backward/gradient regression test, or it repeats #184848's fate. My 20/20 autograd
   check covers `size=`/`scale_factor=` × static/dynamic, which is the right shape of
   evidence — re-run it with the `sym_float` fix applied before sending.

### 15.4 Issue #159550 — CLOSED (no linked PR) — my Issue E, already filed

"_adaptive_avg_pool2d does not support dynamic shapes?" (Jul 2025), labels
`module: dynamic shapes`, `internal ramp-up task`, `triaged`. Its error:

```
cannot determine truth value of Relational: (((2*s2 + 1900)//s2))*(((2*s3 + 1076)//s3)) > 25
```

Mine is the identical expression shape and the identical `window_size > 25` branch in
`_adaptive_avg_pool2d`. **Do not file Issue E** — it is a duplicate of #159550. If
worth pursuing, comment on #159550 with the smaller repro instead. Note it was closed
with **no linked PR**, so the closure reason needs checking before commenting.

### 15.5 Confirmed clear (0 relevant hits)

- `guard_int lowering symbolic in:title` → 0
- `FloorDiv cannot be interpreted as an integer` → 0 → **PR3 has no prior art**
- `ops.constant symbolic in:title` → 0 → **PR2 has no prior art**
- `upsample_nearestnd` → only #117538 and a mypy PR

Nearest neighbour to PR2 in spirit is **#193959** (merged) — *"Don't match
pointless_cumsum_replacement on a symbolic fill"*: a symbolic value reaching code that
requires a Python constant, fixed by declining the match. Same defect *class*, different
mechanism (fx `Node` in a pattern matcher, not sympy in `ops.constant`). Useful to cite
as precedent that maintainers accept this class of fix.

### 15.6 Revised plan

| item | prior art | action |
|---|---|---|
| **Issue A fix** | #175154 OPEN, 3 failed PRs | ⚠️ **superseded by §16.5 — our fix breaks a gradient; do not send. And #175154 is NOT `actionable` (§16.1), so there is no wait-free path.** |
| PR1 (B) | #117538 merged (adjacent) | file Issue B; cite #117538 as the origin of `inv_scales` |
| PR2 (C) | none; #193959 same class | file Issue C; cite #193959 as precedent |
| PR3 (D) | none | file Issue D; send first |
| Issue E | **duplicate of #159550** | **do not file** |

## 16. RE-VERIFIED 2026-08-27 via the GitHub API — two of §15's claims were WRONG

§15 was assembled from rendered page summaries. Re-checking against
`api.github.com/repos/pytorch/pytorch/issues/<n>` (authoritative labels) corrected two
things and found a third.

### 16.1 ❌ CORRECTION: #175154 is NOT labelled `actionable`

Authoritative labels: `high priority`, `triaged`, `module: dispatch`, `oncall: pt2`,
`module: decompositions`, `PT2-Bug-Bash`. **No `actionable` label.** §15.2's claim that
"a fix there needs no waiting period" was wrong and is withdrawn. `PT2-Bug-Bash` is a
bug-bash bucket, not the review gate. The issues-before-PRs gate therefore applies to
*everything* we have.

Still true and still valuable: it is **OPEN**, `high priority`, a maintainer
(azahed98) wrote *"Reproduced. Marking high-priority due to silent incorrectness,"*
and there is **no merged fix**.

### 16.2 ⭐ NEW: #175154 has had THREE failed attempts, not one

| PR | approach | why it died |
|---|---|---|
| [#175177](https://github.com/pytorch/pytorch/pull/175177) | `scale = isize/osize` when `isize==osize` or `osize==2*isize` | **stale-closed.** frgossen: "Please add a test case"; also "You have this case twice." Author said "I'll add the requested changes ASAP" — never did. |
| [#178513](https://github.com/pytorch/pytorch/pull/178513) | `torch.floor(...)` instead of truncation + clamp to `isize-1` | **stale-closed. CI NEVER RAN** (workflows awaited approval); zero review comments. |
| [#184848](https://github.com/pytorch/pytorch/pull/184848) | native fast-path shortcuts + `1.0/scales[d]` | **the only technical failure:** CI 23 failures; karthickai: "forward now matches eager, but the grad no longer does." |

**Two of three died of neglect, not of being wrong.** That is a different situation
from "three people tried and the problem is hard."

### 16.3 Head-to-head evaluation of every candidate (`a100/eval_175154_attempts.py`)

All four approaches, scored on the four axes that matter together:

| variant | #175154 | our Issue A | grads | static no-op |
|---|---|---|---|---|
| stock | wrong | 7 wrong | 25/28 | 14/14 |
| **#175177 branch** | **FIXED** | 7 wrong | **25/28** | **14/14** |
| #178513 floor+clamp | wrong | 7 wrong | 25/28 | 14/14 |
| ours `sym_float` | wrong | **exact** | **24/28** ⚠️ | 14/14 |
| #178513 + `sym_float` | wrong | **exact** | **24/28** ⚠️ | 14/14 |

Findings:
- **#175177's approach is the only one that fixes #175154, and it costs nothing** — grads
  stay at the stock 25/28, static callers unchanged. It was stale-closed over a missing
  test and a duplicated branch, both trivial to fix.
- **#178513's floor+clamp fixes neither bug** on 2.13. Its premise (truncation vs floor)
  does not bite for these cases because the scaled index is non-negative, where trunc ==
  floor. So the PR nobody reviewed would not have worked anyway.
- **Our `sym_float` and #175154 are confirmed independent**: each fix repairs its own bug
  and leaves the other's untouched.
- ⚠️ **Our `sym_float` still costs a gradient** (25/28 → 24/28), reconfirming §17.

### 16.4 Root cause of #175154, now precisely understood

`scale_factor=1.3` on width 2 → `floor(2*1.3) = 2`, so **osize == isize == 2**. The
decomposition then computes `scale = isize/(isize*scales) = 2/2.6 = 0.769`, giving
`floor(1 * 0.769) = 0`, while the true geometric scale is `isize/osize = 1.0` →
`floor(1*1.0) = 1`. Eager uses the geometric ratio; the decomposition uses the
user-supplied float. **Different root cause from Issue A** (which is an approximate
SymInt divide, not a wrong choice of scale).

### 16.5 Revised recommendation

The highest-value *ready* contribution is **#175154 via #175177's approach**: it is a
confirmed-open `high priority` silent-incorrectness bug, the fix is measured to work with
zero gradient or no-op cost, and the two prior attempts died of neglect rather than
rejection — with the reviewer's objections already known and trivial (add a test; dedupe
the branch). Both are things we do well.

Do **not** send our `sym_float` Issue A fix until the ATen backward is handled too.

### 16.6 "No prior art" for PR2/PR3 — confirmed across a second, wider search

Absence of evidence needs more than one query, so PR2/PR3 were re-searched five more
ways (all `repo:pytorch/pytorch`):

| query | hits | overlap? |
|---|---|---|
| `upsample_nearest2d_backward in:title` | 1 (#96612, a primTorch meta impl, 2023) | no |
| `adaptive_pooling_fn OR _adaptive_pooling` | 0 | — |
| `guard_int in:title` | 4 (all API-renaming / removal PRs) | no |
| `"ops.constant"` | 108, none about a symbolic-value contract | no |
| `input_size guard backward inductor in:title` | 0 | — |

Nearest same-family issue is **#188323** (closed) — `dynamic=True` convolution lowering
crashing with `ValueError: Exponent must be non-negative` from SymPy shape evaluation.
Same defect *class* as ours (a symbolic value reaching code that needs a concrete one,
under dynamic shapes) but a different op and a different exception. Useful as evidence
that this class is recognised and triaged, not as prior art.

**Conclusion: PR2 and PR3 have no prior art. PR1 has an adjacent merged PR (#117538) that
created the code it edits, and no duplicate.**

### 16.7 Standing rule learned here

**Read labels from the API, not from a rendered page.** The `actionable` error in §15.2
came from trusting a page summary; `api.github.com/repos/pytorch/pytorch/issues/<n>`
returns the label list authoritatively and takes one call. The same call also gives
`state`, `updated_at`, `comments`, and `assignee` — everything needed to decide whether a
prior-art conclusion has gone stale. Do this before *every* filing decision, because the
whole plan hangs on which label an issue carries.

## 17. Re-check 2026-08-28 (H100 session) — prior art refreshed, two upstream corrections

§15/§16 were assembled on 2026-08-26/27. Re-run here because prior-art conclusions age
and the filing plan hangs on them. **Nothing invalidates a patch or a measurement**;
two *upstream-status* claims were wrong and are corrected below.

⚠️ **Method limit, stated up front.** `api.github.com` returned **HTTP 403 (rate limit
exceeded for this container's shared IP)** for the whole session, so labels below were
read from rendered pages — the exact thing §16.7 warns against. Treat every label claim
in this section as **provisional** and re-read it from the API before filing. The
source-code checks in §17.3 are not affected: those were fetched raw and read directly.

### 17.1 ❌ CORRECTION: PR #184848 was authored by **jansel**, and did not die of the CI failure

§15.3/§16.2 imply an outside contributor whose PR failed on technical grounds. Both
halves are wrong:

- **Author: `jansel`** — the Inductor lead, with an **`agentic`** label; the body says
  "Generated by my agent."
- **It was closed by its own author in a bulk sweep**, not rejected: *"being closed as
  part of a cleanup of open agentic PRs to focus review bandwidth on a smaller set; the
  linked issue remains open."* The 23 CI failures and karthickai's gradient objection are
  real and still the substantive lesson — but "closed unmerged" here means *withdrawn*,
  not *judged wrong*.

**Why this matters for us, in both directions:** the maintainer who owns this area has
already attempted #175154 and set it aside, so the ground is not unclaimed — but the
attempt was an agent-generated PR abandoned for review-bandwidth reasons, which is a
weaker precedent against a careful human fix than §16.2's framing suggested. It also
means **the `agentic` label is a liability**: a PR that reads as agent-generated is what
just got swept. Ours must read as measured human work — which the guard matrix supports.

### 17.2 ⭐ NEW: #185806 — the strongest external precedent found so far, and it cites upsample

[**#185806**](https://github.com/pytorch/pytorch/issues/185806) (closed, `high priority`,
`module: correctness (silent)`, `module: dynamic shapes`, assigned desertfire):
*"codegen_symbol uses float reciprocal for integer dim recovery, silently corrupts
dynamic batches when 1/K is not exactly representable in IEEE-754."*

`int64_t s46 = (1.0/1496.0)*arg4_1_size_0;` — the reciprocal is inexact, truncation gives
the wrong symbolic batch, and it propagates into every downstream allocation.

**Two things here are directly load-bearing for us:**

1. **It is the same defect class as Issue A, independently accepted as `high priority`
   silent incorrectness**: a float divide standing in for exact integer/rational
   arithmetic on a symbolic size. Cite it in Issue A as evidence the class is recognised
   — it is a better precedent than #193959 (§16.6), because it is about *inexact division
   on a symbolic size*, not merely a symbolic value in the wrong place.
2. ⭐ **It names the PR that made the CPP printer divide in floating point — and that PR
   was motivated by upsample under dynamic shapes.**
   [**#95698**](https://github.com/pytorch/pytorch/pull/95698) (merged Mar 2023,
   nkaretnikov, approved ezyang/jgong5) changed `_print_Rational` to emit
   `(1.0/45.0)*ks2` instead of `(1/45)*ks2`, because C++ integer semantics collapsed
   symbolic upsample scale factors to zero. ezyang's review: *"Nope. Make cpp implement
   div correctly."*

   So there is upstream precedent, in exactly our op, that **a symbolic-shape-derived
   upsample scale must be evaluated as a real fraction rather than folded to an integer**
   — which is the same principle PR1 relies on when it defers the divide instead of
   guarding. Worth citing in PR1: the maintainers already decided this once.

   The tension to disclose honestly: #185806 shows the *fix* to #95698 pushed too far the
   other way (float division where exact integer division was needed). Our PR1 is on the
   #95698 side of that line — the quotient is a genuine fraction, not a recovered integer
   — so the two are consistent, but a reviewer who knows #185806 will ask. Answer with
   `aten_ref_probe.py`: eager itself computes this as an fp32 divide
   (`static_cast<float>(input_size) / output_size`), so matching eager *requires* the
   float path, and `div_rn` is what makes it bit-exact.

### 17.3 ⭐ Both patched defects re-verified against `main` **today**, from source

Fetched `raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/lowering.py`
(335 KB, HTTP 200) and read the two functions. **Both defects are still present**, so
PR1 and PR3 remain live and are not duplicates of anything merged since:

```python
# upsample_nearestnd  -- Issue B, unchanged
i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]   # guarded
o_sizes = output_size                                        # NOT guarded
inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
    x = ops.mul(x, ops.constant(scale, torch.float32))        # <- crash site

# upsample_nearest2d_backward  -- Issue D, unchanged
inp_h = V.graph.sizevars.guard_int(inp_h)                    # guarded
*_batch, out_h, out_w = input_size                           # NOT guarded
if inp_h % out_h == 0 ...:  h_kernel_max = ceildiv(inp_h, out_h)
```

One incidental drift: the backward now carries a `# pyrefly: ignore [not-iterable]`
comment above the `input_size` unpack that is absent from the branch base
(`b1716d913a`). Harmless, but **it sits on the line PR3 edits — expect a rebase
conflict there** and keep the comment.

### 17.4 #175154 status re-read (provisional — see the 403 caveat)

Still **OPEN**, still no linked PR, still assigned to **Akshat-Raj** (author of the
stale-closed #175177). Labels as §16.1 recorded them: `high priority`, `triaged`,
`PT2-Bug-Bash`, `module: decompositions`, `module: dispatch`, `oncall: pt2` — **no
`actionable`**, so §16.1's correction stands and the issues-before-PRs gate still applies
to everything we have. #178513 shows as closed **Aug 22, 2026** (`Stale`), which is more
recent than §16.2 recorded; the three-failed-attempts picture is otherwise unchanged.

### 17.5 Searches run, and what they add

| query | result | effect on the plan |
|---|---|---|
| `is:open upsample inductor dynamic` | 3 hits, all unrelated (docs build, 2× MPS trackers) | no change: PR1/PR3 still unduplicated |
| `is:open label:PT2-Bug-Bash interpolate` | 1 hit: #175154 only | confirms it is the sole tracked interpolate-correctness issue |
| `inductor "cannot be interpreted as an integer"` | 1 hit: #100972 (2023, `math` NameError in Triton codegen, closed) | **PR3's exception text still has no prior art** |
| `label:"module: correctness (silent)" dynamic shapes` | 12 hits; **#185806** is the relevant one (§17.2) | new precedent for Issue A |
| `is:issue interpolate nearest dynamic inductor` | 8 hits, none new (3 DISABLED-test bots, MPS trackers) | no change |
| `is:pr interpolate nearest scale compile` | 20 PRs; the three known attempts + #185788/#180310 (unrelated 64-bit-indexing / ROCm) | no new attempt on #175154 since #184848 |

**Net: no finding is superseded and no patch is duplicated.** The additions are
#185806 + #95698 as precedent (§17.2) and the corrected read of #184848 (§17.1).

### 17.6 2.3.1 artifact re-verified on H100 this session

Independent of the upstream work. `patch/patch.py status` → PRISTINE at start;
`repro/repro.py` → **exit 2** (bug active, `TypeError: Cannot convert expression to
float`); `tests/run_tests.py` → **15 passed / 0 failed / 9 skipped**. After
`patch.py apply`: `repro.py` → **exit 0**, tests → **18 passed / 0 failed / 6 skipped**.
Reverted to PRISTINE afterwards. Hardware: 1× H100 80GB HBM3, `tan-1gpu-chip-0-1`,
torch 2.3.1+cu121, python 3.11.15, 198-CPU cgroup quota. Matches the documented
15/18 exactly.

---

## 18. Re-verification on a SECOND A100 — 2026-08-28 (`tan-1gpu-chip-w-0-2`)

Hardware: 1× A100-SXM4-80GB, **sm_80**, driver 575.57.08, 198-CPU cgroup quota
(`nproc` 256 — the host count, not the budget). torch 2.13.0+cu130 (wheel
cf30153c), triton 3.7.1, python 3.12.14. All three patches ON (`tools/state.py
status` verified before every measurement). This is a different box from §1–§16's
A100 and from §17's H100.

**Purpose:** §17 was measured on H100, where `37→74` is clean. Re-running on A100
independently re-tests the architecture-dependence claim rather than inheriting it.

### 18.1 Reproduced without change

| claim | §1–§16 value | here |
|---|---|---|
| 6 new tests, all patches ON | 6/6 pass | **6/6 pass** (47.1 s) |
| adversarial PR1 sweep | 42/42 bit-exact | **42/42 bit-exact** |
| strict no-op digest | 63 cases, 4 pre-existing mismatches | **63 cases, the same 4** |
| `aten_ref_probe` truediv vs eager | 45/3305 coords | **45/3305** |
| `aten_ref_probe` `div_rn` vs eager | 0/3305 | **0/3305** |
| divide is loop-invariant | args only, never `xindex` | **confirmed, 2 `div_rn` calls** |
| Issue A rows wrong | 7/192 · 2/363 · 36/74 · 2/363 | **identical** |
| PR3: forward defers, backward cannot | 7 sizes → 1 lowering | **confirmed** |
| PR2 population | 13 sympy-concrete, 0 symbolic | **13 concrete, 0 symbolic** |

⭐ **`37→74` is 36/74 wrong here, on a second, independent A100.** Combined with
H100's 0/74 (§17, `logs/h100_20260828.md`), the architecture split is now measured
on three boxes and is not a one-off. **Still file Issue A with `448→192` as the
headline** — see §18.3 for why that ratio is the robust choice, now by construction
rather than by luck.

### 18.2 ⭐ Issue A's mechanism, located at bit level

`evidence/issueA_arch_divide_bits.py` reads the fp32 bit pattern each divide
produces on this device, for the cited ratios:

| ratio | eager (correctly rounded) | inductor `truediv` | `div_rn` | scale exactly representable? |
|---|---|---|---|---|
| 448→192 | `0x40155555` | `0x40155556` (**+1 ULP**) | `0x40155555` | no |
| 384→363 | `0x3f8767ab` | `0x3f8767ac` (**+1 ULP**) | `0x3f8767ab` | no |
| 37→74 | `0x3f000000` | `0x3effffff` (**−1 ULP**) | `0x3f000000` | **yes (0.5)** |
| 32→64 | `0x3f000000` | `0x3f000000` (equal) | `0x3f000000` | yes |

Two consequences the earlier write-ups did not state:

1. **`37→74`'s scale is exactly representable** (`0.5`, a power of two). fp32
   precision therefore cannot explain its failure — the error is *purely* Triton's
   approximate reciprocal (`x * rcp(y)` rather than a true divide) landing one ULP
   low. That is why the case is SM-dependent: how `rcp` errs is a hardware detail.
2. **The error has a sign, and the sign decides reproducibility.** The two
   downsample ratios err **+1 ULP** (compiled picks a *higher* source pixel: 62→63);
   `37→74` errs **−1 ULP** (compiled picks a *lower* one: 1→0). A GPU whose divide
   errs high sees the first two and not the third — exactly the observed
   A100/H100 split.

### 18.3 ⭐ The ULP model is a PREDICTOR, so filing ratios can be chosen on purpose

A host-side model — eager uses correctly-rounded fp32, inductor uses the SM's
`truediv`, both floored as `UpSample.h` does — reproduces the GPU's wrong-row
counts exactly. Fitting four known ratios would prove nothing, so
`evidence/issueA_predict_validate.py` makes the model **commit in advance** on
ratios this repo has never run, then runs real `torch.compile(F.interpolate)`:

| ratio | predicted | measured | ratio | predicted | measured |
|---|---|---|---|---|---|
| 448→368 | 9/368 | **9/368** | 448→96 | 3/96 | **3/96** |
| 448→384 | 7/384 | **7/384** | 256→128 | 0 | **0** |
| 500→120 | 6/120 | **6/120** | 64→256 | 0 | **0** |
| 100→370 | 4/370 | **4/370** | 255→256 | 0 | **0** |
| 363→319 | 3/319 | **3/319** | 1000→999 | 0 | **0** |
| 384→246 | 3/246 | **3/246** | 100→70 | 0 | **0** |

**12/12 exact** — not just the counts but the exact row indices and the exact
substituted pixel. Five predicted-clean cases are included deliberately: a model
that only ever predicts "wrong" is not falsifiable.

⚠️ One methodological correction, worth keeping because it inverted a result: an
earlier version of the classifier compared against **exact rational** `(d*i)//o`
instead of eager's fp32. That measures eager's own (correct, expected) departure
from rational math, which compiled and eager agree about — and it reported
`448→192` as `0/192`, the opposite of the truth. **The reference must be eager.**
A second bug: perturbing a scale with `math.nextafter` steps in *float64*, and
rounding back to fp32 lands on the same value, so the ULP perturbation was a
silent no-op (it reported 0 ULP-robust ratios while 178 were observably
miscompiling). Step the fp32 **bit pattern** instead.

With those fixed, `evidence/issueA_ratio_classes.py` classifies 3586 ratios by
whether a ±1 ULP scale error moves any source pixel:

| class | count | meaning for filing |
|---|---|---|
| **ULP-robust** (both directions move a row) | 53 | reproduces on any SM whose divide is inexact either way |
| one-sided (only one direction) | 1634 | reproduces only on SMs erring that way — the `37→74` trap |
| ULP-immune (neither) | 1899 | safe |

178 of the 3586 miscompile on this device today. **Reclassifying the cited
ratios: `448→192` is ULP-robust; `384→363` and `37→74` are one-sided.** So
`448→192` is the correct headline on evidence, not merely because it happened to
reproduce twice. Best additional robust candidates: **448→368 (9 rows)**,
448→384 (7), 500→120 (6).

Artifacts: `logs/issueA_ratio_classes.json`.

### 18.4 ⚠️ CORRECTION: `div_rn` is NOT faster than `truediv` — it is ~2% slower at large shapes

§10 reports `div_rn/truediv` = **0.957–0.984** ("not slower … marginally faster").
That does not survive re-measurement, and **§10's wording should be corrected
before PR1 is sent.**

Re-running `perf_divrn.py` unchanged on this A100 gave **1.009–1.212** — up to 21%
*slower*, the opposite direction. A claim that flips between two runs of the same
script on the same GPU model is not measuring the divide. The cause is in the
harness: `perf_divrn.py` times each variant **once, in a fixed order** (`div_rn`
first), with 20 warmup iterations and no clock control. This box idles at
**210 MHz against a 1410 MHz max**, so whichever variant runs first pays the
clock-ramp — an order effect.

`evidence/perf_divrn_controlled.py` fixes the method: clock stabilization first,
**CUDA-event** (device-side) timing, **15 A/B/A/B alternations** so drift cancels,
median-of-ratios, and — following this repo's own rule — an **A-vs-A control run
first** to prove the harness can resolve the effect at all.

| shape | control (rn vs rn) | A/B (rn vs truediv) |
|---|---|---|
| 1×3×64×64 → 128×128 | 0.9791 [0.925, 1.207] | 0.9844 [0.914, 1.102] |
| 1×3×256×256 → 512×512 | 0.9798 [0.850, 1.160] | 0.9841 [0.872, 1.193] |
| 8×64×128×128 → 256×256 | 1.0000 [0.998, 1.045] | **1.0170** [1.0164, 1.0302] |
| 4×128×256×256 → 512×512 | 1.0000 [0.9995, 1.0003] | **1.0176** [1.0170, 1.0186] |

The control straddles 1.0 at every shape, so the harness is unbiased. Reading it:

- The two **small** shapes are dominated by launch overhead — ±10% run-to-run in
  the control, so nothing about the divide is resolvable there.
- The two **large** shapes have tight bounds in both control (±0.5%) and A/B
  (1.0164–1.0302), and the A/B interval **excludes 1.0**. So the cost is real,
  consistent, and **≈1.8%**.

**The honest claim for PR1: `div_rn` costs ≈2% at large shapes and is
unresolvable at small ones — not "free", and certainly not "faster".** The
structural argument (§10's loop-invariance, re-confirmed here) is unaffected and
is still what answers the #164144/B200 objection; it just should not be paired
with a sub-1.0 ratio. A ~2% loop-invariant cost is a much easier thing to defend
than a claimed speedup a reviewer might fail to reproduce.

### 18.5 ⚠️ CORRECTION: PR2's compile-time cost, and a cache trap that silently zeroes the probe

**The trap first, because it invalidates the coverage number if missed.**
`pr2_blast_radius.py` instruments `OpsWrapper.constant`, which only runs while a
graph is being **lowered**. With inductor's cache warm, compilation is served from
cache, no lowering happens, and the probe printed:

```
total ops.constant calls in this workload : 0
distinct inductor call sites hit          : 0  (of ~111 static)
```

That reads as *"the check is never reached"* — the opposite of PR2's
justification — when in fact **nothing was measured**. It is easy to hit here
because `TORCHINDUCTOR_CACHE_DIR` defaults into **NFS `$HOME`**, a cache shared
with every other container in the fleet and warm from earlier sessions. Cold
cache on the same box: **1221 calls, 5 sites, 13 sympy-concrete, 0 symbolic** —
reproducing §13.2's load-bearing finding (1157/13/0; the small total delta is
version drift, the decisive **13 concrete / 0 symbolic** is identical).

`pr2_blast_radius.py` now forces a private cold cache dir itself and **aborts with
an explanation if the count is 0**, so this cannot be misread again.

**The cost number.** §13.2 quotes **1.23 s vs 1.25 s (0.98×)**. Three identical
runs of the original script here gave **0.9864 / 1.0431 / 1.1246** — straddling
1.0, i.e. below the harness's resolution; the recorded 0.98× was one draw from
that spread. Interleaved on a **cold** cache the measurement becomes reproducible:

| rep | with check | without | ratio |
|---|---|---|---|
| 1 | 5.29 s | 5.13 s | 1.0330 |
| 2 | 5.26 s | 5.06 s | 1.0392 |
| 3 | 5.25 s | 5.04 s | 1.0412 |

median **1.0392**, min 1.0330, max 1.0412 — the interval **excludes 1.0**.
So the check costs a real, consistent **≈3.9% of lowering time** (cold-cache
compile, ~5 s workload), not 0.98×. It remains **compile-time only** and never
appears in the generated kernel, which is the point that matters for review —
but PR2 should say "≈4% of lowering time on a cold cache", not "no cost".

### 18.6 Two harness defects fixed

- `noop_digest.py` read `sys.argv[1]` at the very end, **after** all 63 GPU
  compilations, so forgetting the argument burned the whole run and then raised
  `IndexError` with nothing written. Now validated before any work (fails in 5.5 s).
- `pr2_blast_radius.py` — the cold-cache forcing and zero-count abort above.

### 18.7 What this session did NOT change

No patch, PR body, issue draft, or test was modified. Nothing has been submitted
to PyTorch; the CLA is still unsigned and `lintrunner -a` has still never run.
The corrections in §18.4 and §18.5 are to **claim wording in evidence documents
and PR bodies**, not to code.

### 18.8 The 2.3.1 teaching artifact re-verified on this A100

Independent of the upstream work. `venv_image_server_onediff` (torch 2.3.1+cu121,
python 3.11.15, CUDA available on this A100):

| step | expected | got |
|---|---|---|
| `patch/patch.py status` at start | PRISTINE | **PRISTINE** |
| `repro/repro.py` unpatched | exit 2 (bug active) | **exit 2** |
| `tests/run_tests.py` unpatched | 15 pass / 0 fail | **15 / 0** (9 skipped) |
| `patch/patch.py apply` → `repro.py` | exit 0 | **exit 0** |
| `tests/run_tests.py` patched | 18 pass / 0 fail | **18 / 0** (6 skipped) |
| `repro/verify_fix.py` | bit-exact, 7 sizes | **7/7 bit-exact, max diff 0** |
| restored afterwards | PRISTINE | **PRISTINE** |

Matches the documented 15/18 exactly, now on A100 as well as H100.

⚠️ Measurement hygiene note: `repro/repro.py` detects that
`TORCHINDUCTOR_CACHE_DIR` points at network `$HOME` and redirects itself to a
node-local `/tmp` dir. That is the same hazard that silently zeroed
`pr2_blast_radius.py` (§18.5) — the difference is that `repro.py` handles it and
says so. Worth copying that pattern into any new probe.

### 18.9 PR1's `11695` ULP figure independently reproduced

`evidence/probes/ulp_proof.py` → **11695** `(i,o)` pairs in `1..512` where an
approximate reciprocal flips a floored index, and **0** double-rounding
disagreements. Reproduced exactly.

Note that 11695 and §18.3's "178 miscompiling here" are **different populations,
and both are correct**: 11695 counts pairs over the full 512×512 grid using a
*modeled* reciprocal, while 178 counts ratios miscompiling on *this device* within
the 3586-ratio subset swept. Do not present either as a correction of the other.
Usefully, `ulp_proof.py`'s examples show **both** error signs (`rn=1 approx=0` and
`rn=0 approx=1`), which independently corroborates the ±1 ULP sign mechanism
§18.2 measured on-device.

### 18.10 The no-op proof now holds ACROSS architectures, not just across patch states

`logs/noop_ON_a100.json` (this A100, patches ON) compared against the recorded
`logs/noop_ON.json` (H100, ON) and `logs/noop_OFF.json` (H100, OFF):

| comparison | result-hash diffs | code-hash diffs |
|---|---|---|
| H100-ON vs **A100-ON** | **0 / 63** | 50 / 63 |
| **A100-ON** vs H100-OFF | **0 / 63** | — |

So every one of the 63 real ATen upsample configurations produces **bit-identical
results** on H100 and A100, with and without the patches. This is strictly stronger
than §3 (which compared ON vs OFF on one box) and independently re-confirms that the
50/63 generated-Triton-source differences are noise: they differ across a *hardware*
change that provably did not move a single result bit.

The 4 compiled≠eager cases in the A100 digest are the known Issue A ratios and
nothing else.

⚠️ Do not read the digest's `37→74` entry as contradicting §18.2's architecture
split: that case is `(37, 41) → (74, 82)`, so dim 3 is `41→82` — a different ratio
from the `37→74` in `mismatch_repro.py`, whose dim 3 is a static 4. The digest
compares compiled against eager **on the same box**, which catches Issue A on both
architectures; only the *per-row count* for the bare `37→74` ratio is SM-dependent.

---

# §19. H100 re-verification and same-hardware no-op A/B — 2026-08-30

Hardware re-probed, not inherited: `tan-1gpu-chip-0-2`, **1× NVIDIA H100 80GB HBM3**,
driver 575.57.08, 198-CPU cgroup quota. torch 2.13.0+cu130, triton 3.7.1, python
3.12.14. This is a **third** box (previous: two A100s, one earlier H100).

## 19.1 ⭐ The no-op claim, finally measured as a same-hardware A/B

Every earlier no-op number compared a stored digest against a fresh run. On this box
that method **falsely reports 8/63 differences** — because the seeded CUDA tensors
themselves are not identical across architectures, so *eager* hashes differ A100 vs
H100 too. A cross-hardware digest comparison measures the GPU, not the patch.

Run both arms on the same GPU in one session (`state.py` toggled between them):

| comparison | identical |
|---|---|
| eager result hash, ON vs OFF | **63/63** |
| **compiled result hash, ON vs OFF** | **63/63** ← the load-bearing number |
| generated Triton source hash, ON vs OFF | 63/63 (noisy in general; do not rely on it) |

**Strict no-op confirmed on H100.** Artifacts: `logs/noop_h100_20260830_{ON,OFF}.json`.

> **Method rule to carry forward:** a no-op A/B is only valid **within one box, one
> session**. Comparing to a digest from other hardware conflates the patch with the SM.
> Always confirm the *eager* hashes match first — if they do not, the comparison is
> void before you look at the compiled ones.

## 19.2 Guard matrix — 6/6, all four states

`tools/guard_matrix.sh`, full log in `logs/guard_matrix_h100_20260830.log`:

| state | forward | concrete-sympy | one-graph | const-reject | const-accept | backward |
|---|---|---|---|---|---|---|
| all ON  | OK | OK | OK | OK | OK | OK |
| PR1 off | **FAILED** | OK | **FAILED** | – | – | OK |
| PR2 off | OK | – | – | **FAILED** | OK | – |
| PR3 off | OK | – | – | – | – | **FAILED** |

State restored to all-ON afterwards (verified). `adversarial_pr1.py`: **42 bit-exact,
0 problems.**

## 19.3 Issue A on H100 — and a correction to §18.2's mechanism

Measured directly (`arange` along the sampled dim, dim 3 static at 4), **one ratio per
process**, with the output size written as a module global so the divisor is emitted as
a literal (the conservative form — see the discriminator below):

| mode | ratio | rows differing, H100 | prior A100 |
|---|---|---|---|
| nearest | 448→192 | **7/192** | 7/192 |
| nearest | 384→363 | **2/363** | 2/363 |
| nearest | 448→368 | **9/368** | 9/368 (predicted) |
| nearest-exact | 384→363 | **2/363** | 2/363 |
| nearest | 41→82 | **40/82** | (new) |
| nearest | 37→74 | 0/74 *(literal divisor)* / **36/74** *(symbolic divisor)* | 36/74 |
| nearest | 74→148, 101→202 | 0 | — |

### ⚠️ ⭐ The real discriminator: whether the DIVISOR is a runtime value or a literal

Chasing why `37→74` reported both `0/74` and `36/74` on the *same GPU in the same
session* produced the most useful mechanistic result of this session, and it invalidates
part of §18.2's model.

**It is not the architecture, not the scale's bit pattern, and not compile history. It
is which operands of the divide survive as runtime kernel arguments** — which in turn
depends on something as innocuous as **Python variable scope**:

| how the output size is written | emitted Triton | `37→74` |
|---|---|---|
| `o` is a **module global** | `tmp0 = (ks0 / 74).to(tl.float32)` | **0/74 wrong** |
| `o` is a **function local** (closure) | `tmp0 = (ks0 / ks1).to(tl.float32)` | **36/74 wrong** |

Same GPU, same torch, same ratio, same arithmetic on paper. With a literal divisor the
divide is folded/exact; with **both** operands as runtime arguments the hardware
reciprocal is used and errs. Verified directly on the emitted divide:

| `ks0 / o` as emitted | runtime result | correctly rounded | error |
|---|---|---|---|
| `37 / 74` (divisor literal) | `0x3f000000` | `0x3f000000` | none |
| **`41 / 82`** (divisor literal) | **`0x3effffff`** | `0x3f000000` | **−1 ULP** |
| `74 / 148`, `101 / 202` | `0x3f000000` | `0x3f000000` | none |

So two things are now separated that §18.2 had conflated:

1. **Some ratios err even with a literal divisor** — `41→82` does, `37→74` does not,
   though all four share the correctly-rounded pattern `0x3f000000`. The ratio's *scale
   value* predicts nothing; the reciprocal's error on that specific argument pair does.
2. **Any ratio can be made to err by keeping the divisor symbolic too.** That is the
   `ks0 / ks1` form, and it is what a closure over a local variable produces.

**Corollary — a probe must reproduce the emitted form.** Constant-folding the divide in
a hand-written Triton kernel measures a path Inductor did not generate; that is what
first reported `37/74` as wrong-low and sent me hunting a codegen difference that does
not exist.

**Consequence for filing, unchanged in direction:** headline **448→192, 384→363,
448→368**. All three are wrong with a *literal* divisor — the weaker, more conservative
form — on both A100 and H100, so they do not depend on the reporter's SM, their Python
scoping, or compile order. `37→74` is now demoted out of the issue entirely rather than
merely caveated: it is the case whose verdict flips on incidental details.

`evidence/issueA_runtime_divide.py` runs one ratio per process and prints both the row
count and the emitted divide's ULP error, so the two effects cannot be confused again.



## 19.4 Emitted Triton, verified verbatim for the issue body

`run_and_get_code` on `F.interpolate(size=(192,4), mode="nearest")`, 448→192:

```python
# dynamic=True   (ks0 is a kernel argument -> the divide is loop-invariant)
tmp0 = (ks0 / 192).to(tl.float32)
tmp1 = (x1).to(tl.float32)
tmp3 = tmp2.to(tl.int64)

# dynamic=False  (folded to a literal, which is why only the dynamic path is wrong)
tmp1 = tl.full([1], 2.3333333333333335, tl.float32)
```

## 19.5 The 2.3.1 artifact, re-run on this box

`patch.py status` → PRISTINE · `repro.py` → **exit 2** · `run_tests.py` → **15 passed /
0 failed**. Unchanged from every prior box.

## 19.6 Prior art re-read from the GitHub API (the 2026-08-28 caveat, closed)

`api.github.com` was reachable this session, so every label in `upstream/issues.md` is
now API-sourced rather than read off a rendered page. **#175154 is confirmed still
`open` and still NOT `actionable`.** Full table in `upstream/issues.md`.

## 19.7 ⚠️ Two documentation defects found and fixed

Neither touches a patch, a test, or a measurement — both are citation errors that a
maintainer would have caught:

1. **"One concern per PR" is not a PyTorch rule.** Five of our documents attributed it
   to `CONTRIBUTING.md`. It appears in **neither** `CONTRIBUTING.md` nor the Ultimate
   Guide wiki (both fetched and grepped 2026-08-30). Splitting into three PRs is still
   right, but it is *our* judgment and is now labelled as such.
2. **`AI_POLICY.md` requires disclosure, and no PR body had it.** Verbatim:
   *"AI-generated content in comments, issues, or PRs must be clearly disclosed and
   contained… must be accompanied by human commentary explaining its relevance"*, and
   *"We do not accept contributions created by fully autonomous agents."* Submitting
   without a disclosure line would have violated policy on all three PRs. A template is
   now in each `PR*_BODY.md`, and `SUBMIT.md` quotes the rule.
