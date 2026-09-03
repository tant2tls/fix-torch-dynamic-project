# Upstream PyTorch contribution — three PRs and four issue reports

A report of the upstream work that came out of this project. The teaching artifact in this
directory (torch 2.3.1, `repro/` + `patch/` + `tests/`) is what *found* the bug; this file
is what happened when the same defect was chased on current `main`.

**Everything referenced here now lives in this repository:** the issue drafts and
per-PR submission docs in [`upstream/`](upstream/), the `git am`-ready patches in
[`upstream/patches/`](upstream/patches/), the full A100 evidence in
[`evidence/RESULTS_a100.md`](evidence/RESULTS_a100.md), and the H100 re-verification
in [`evidence/logs/h100_20260828.md`](evidence/logs/h100_20260828.md). The only
external artifact is the local `pytorch` checkout the patches were generated from
(branch `prseries` on `b1716d913a`) — the patches reproduce it exactly, so it is not
needed here.

**Start with [`upstream/README.md`](upstream/README.md)** for the send order.

| | |
|---|---|
| **Verified on** | 1× A100-SXM4-80GB, driver 575.57.08 |
| **torch** | 2.13.0+cu130 (release wheel, git `cf30153c`), triton 3.7.1, python 3.12.14 |
| **CPU checks** | torch 2.13.0+cpu |
| **Branch** | `prseries` on `b1716d913a`, +299/−5 across 3 files |
| **Status** | staged and verified; **nothing pushed** (needs a GitHub account, CLA, and an `actionable` issue) |

---

## 1. The process gate: issues before PRs

PyTorch will not review a new contributor's PR without a linked issue carrying the
**`actionable`** label:

> "Only PRs that address issues labeled `actionable` will be considered for review."
> "you must wait for a maintainer to review it and mark it actionable before preparing and
> sending a PR for it."
> — [The Ultimate Guide to PyTorch Contributions](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)

From `CONTRIBUTING.md`: sign the **CLA** first; **never paste AI-generated fix explanations
into an issue**; **leave the Reviewers list empty** (a triage squad assigns — do not
@-mention maintainers); `lintrunner -a` before pushing; "you are
personally responsible for what you send."

So the order is **file issue → wait for `actionable` → push the PR**, three separate PRs.

## 2. What was found

Four independent defects, all in TorchInductor's handling of symbolic (dynamic) shapes.
Three have fixes; the strongest one deliberately does not.

### ⭐ A — `F.interpolate(mode="nearest")` returns wrong pixels under `dynamic=True` on CUDA

The most consequential finding: **currently-shipping silent wrong results from ordinary user
code.** Stock 2.13.0+cu130, no patches, no custom op, no `inference_mode`.

| ratio | rows wrong | first disagreement |
|---|---|---|
| 448→192 | 7/192 | out row 27: eager src 62, compiled src 63 |
| 384→363 | 2/363 | out row 121: eager 127, compiled 128 |
| 448→368 | 9/368 | out row 23: eager 27, compiled 28 |
| 41→82 | 40/82 | out row 2: eager 1, compiled 0 |
| nearest-exact 384→363 | 2/363 | out row 60: eager 63, compiled 64 |

**CUDA only, dynamic only** — CPU is exact everywhere, `dynamic=False` is exact everywhere.
It is an off-by-one in the *source index*, not a tolerance issue. (Repro trick: make the
input `arange` along the sampled dim, so each output value *is* the index it was gathered
from and a mismatch is an exact integer.)

Mechanism, from the emitted Triton: the dynamic path computes the scale on-device as
`(ks0 / 74).to(tl.float32)` — an approximate divide — where the static path folds it to a
literal (`tl.full([1], 0.5, ...)`). It lives in the **decomposition**
(`arange/mul/_unsafe_index`), a different code path from the lowering the PRs touch.

Modelling ATen's `UpSample.h` exactly (float32 scale, float32 multiply, floor, clamp) over
9 ratios / 3305 output coordinates:

| divide | coordinates disagreeing with eager |
|---|---|
| `I / O` (what is emitted today) | **45** |
| `tl.math.div_rn(I, O)` | **0** |

**No fix attempted, on purpose.** There the divide is *per-element* in a memory-bound
gather, which is exactly the case that
[PR #164144](https://github.com/pytorch/pytorch/pull/164144) hit: it made `truediv` emit
`div_rn` globally, merged, was reverted several times, and was finally gated behind
`TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING` by
[#165566](https://github.com/pytorch/pytorch/pull/165566) after ~28% B200 throughput loss
([#164301](https://github.com/pytorch/pytorch/issues/164301)). File the issue; let
maintainers choose the remedy.

### B — `upsample_nearestnd` crashes on a symbolic output size → **PR 1**

The lowering guards its input sizes and not its output sizes, divides one by the other, and
hands the quotient to `ops.constant`, which requires a concrete value:

```
NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
```

This is the same defect the 2.3.1 artifact in this directory diagnosed, still latent on
`main`.

### C — `ops.constant` accepts a symbolic value and fails ~12 frames later → **PR 2**

No enforcement at the call, so the failure surfaces in `fx/proxy.py` naming neither the op
nor the fix. (On 2.3.1 the same mistake surfaced as `TypeError: Cannot convert expression to
float` from `sympy/core/expr.py` — the error this project originally chased.)

### D — `upsample_nearest2d_backward` crashes on a symbolic `input_size` → **PR 3**

```
TypeError: 'FloorDiv' object cannot be interpreted as an integer
```

### E — `adaptive_avg_pool2d` under dynamic shapes — **held**

`cannot determine truth value of Relational`. Pre-existing, confirmed identical with all
patches reverted, untouched by these PRs. Filing five issues at once from a new account
reads worse than filing four good ones.

## 3. The three PRs

```
cee885e016  [inductor] Fix symbolic output_size in upsample_nearestnd     -> Issue B
3d9d172c6f  [inductor] Reject symbolic values in ops.constant             -> Issue C
c87c2a8580  [inductor] Guard input_size in upsample_nearest2d_backward    -> Issue D
```

| file | + | − |
|---|---|---|
| `test/inductor/test_custom_lowering.py` | 240 | 0 |
| `torch/_inductor/lowering.py` | 41 | 5 |
| `torch/_inductor/virtualized.py` | 23 | 0 |

**Send PR 3 first** — 7 source lines, one test, no design argument. Then PR 1. Then PR 2,
which changes a contract on a 111-call-site surface and will attract the most debate.

### The one design argument worth understanding

PR 1 refuses to guard; PR 3 guards. That is not inconsistent, and the reason is structural:

| | forward (PR 1) | backward (PR 3) |
|---|---|---|
| the quotient becomes | a scalar multiplier | a `range()` bound |
| so it lives in | the kernel **argument** list | the kernel **body** |
| deferrable? | yes — one kernel serves every size | no — the window is unrolled at lowering time |
| measured, 7 distinct sizes | **1** lowering invocation (vs **7** with `guard_int`) | necessarily one per distinct window |

A 2×2 pooling window is 4 unrolled loads; 8×8 is 64. Those are different kernels, not one
kernel with a different argument — so specializing in the backward is the only way to lower
it at all, while in the forward it would cost a recompile per output size and eventually
drop the caller to eager.

### Why `ops.div_rn` in PR 1 does not repeat #164144

Correctness requires it (45 disagreements vs 0, above). Cost does not object, because here
the divisor is the symbolic output size — a *kernel argument*, uniform across the grid:

```python
tmp0 = tl.full([1], 128.0, tl.float32)
tmp1 = (ks0).to(tl.float32)
tmp2 = triton.language.div_rn(tmp0, tmp1)   # depends on ks0/ks1, never xindex
```

One scalar op per launch, hoisted out of the vectorised body. Re-measured on A100 with an
A-vs-A control, CUDA events and 15 alternations (`evidence/perf_divrn_controlled.py`):
`div_rn/truediv` = **0.984, 0.984, 1.017, 1.018**. Only the two large shapes are
resolvable — there the interval (1.016–1.030) **excludes 1.0**, so the real cost is
**≈2%**; at the two small shapes the A-vs-A control itself swings ±10%, so nothing about
the divide can be read off them.

⚠️ **CORRECTED 2026-08-28.** The earlier figures (0.965 / 0.957 / 0.983 / 0.984, "not
slower") came from timing each variant **once, in a fixed order**, on a GPU that idles at
210 MHz against a 1410 MHz max — the first-measured variant ate the clock ramp. Re-running
that same script later gave 1.009–1.212, the opposite direction. See `RESULTS_a100.md` §18.4.

⚠️ **Claim limit:** A100 (SM 8.0), four shapes. Says nothing about B200. The honest sentence
is **"the divide is loop-invariant here; ≈2% at large shapes on A100"** — never "div_rn is
free" and never "not slower."

## 4. Evidence

- **6 new tests**, CPU and GPU, and each **fails with only its own fix reverted** (the guard
  matrix — a test that passes with its fix reverted is not a regression test):

  | state | forward | concrete-sympy | one-graph | const-reject | const-accept | backward |
  |---|---|---|---|---|---|---|
  | all ON | OK | OK | OK | OK | OK | OK |
  | PR1 off | **FAIL** | OK | **FAIL** | – | – | OK |
  | PR2 off | OK | – | – | **FAIL** | OK | – |
  | PR3 off | OK | – | – | – | – | **FAIL** |

- **Strict no-op for existing callers: 0/63 differ.** 63 real `F.interpolate`/`nn.Upsample`
  configurations (`size=` and `scale_factor=`, 1-D/2-D/3-D, static and dynamic, forward and
  real autograd backward) — compiled results bit-identical with and without the patches.
- **Adversarial sweep: 42/42 bit-exact** — mixed symbolic/concrete dims, explicit `scales_x`
  on one dim only, nearest-exact at ULP-sensitive ratios, 1-D/3-D, fp16/bf16, non-contiguous
  input, degenerate sizes, downsampling. Coverage *verified* by instrumenting the predicate,
  so the deferred branch is known to have run rather than silently taking the concrete path.
- **Full dynamic-shapes suite: `Ran 2621 tests`, 1 failure — pre-existing and unrelated**
  (`test_unbacked_reduction_cpu`, an inverted-xfail that fails identically with all patches
  reverted). Zero attributable failures. This is what makes PR 2's contract change
  defensible.
- **PR 2 blast radius, measured:** 1221 `ops.constant` calls over a 24-case workload —
  98.94% plain values, 13 sympy-but-concrete, **0 sympy-symbolic**. Compile time with the
  check vs without, cold cache and interleaved: **≈3.9% slower** (1.033–1.041), i.e. a real
  but compile-time-only cost — it never reaches the generated kernel. ⚠️ The probe must run
  on a **cold** inductor cache; warm, it records 0 calls and reads as "never reached."
- **20/20** gradient comparisons through real `F.interpolate` autograd bit-identical.

## 5. Things that were nearly claimed wrongly

Kept because the corrections are the actual content of a careful review.

- **"All 12 `upsample_nearest*` overloads are `CompositeImplicitAutograd."** False — only
  the three `.vec` ones are. The lowering is unreachable because
  `upsample_nearest{1,2,3}d` are in `torch._decomp.decomposition_table` and the
  decompositions fire first (verified: **0 lowering hits** across 8 entry points). A
  reviewer who checks the dispatcher would have caught this.
- **"The backward is decomposed before Inductor sees it."** Also false — the backward op is
  *not* in `decomposition_table`. The *forward* is decomposed, so the backward graph
  contains `_unsafe_index_put` and `upsample_nearest2d_backward` is never emitted
  (`bwd graph ops: ['_unsafe_index_put', 'expand', 'new_zeros']`).
- **Generated-Triton-source hashes are not no-op evidence.** They differ in **50/63** cases
  between two runs of the *identical* state (they embed cache paths and kernel names). Only
  result hashes count. The A-vs-A control run is what caught this before it went in a PR.
- **A test that registers a lowering per loop iteration must vary the custom op NAME.** The
  FX graph cache keys on the op name, and a flag captured in the lowering's closure
  (`exact`) is invisible to that key — so iteration two silently reused iteration one's
  kernel. This produced a **false failure** (35.2% of elements wrong, which looks exactly
  like a broken fix) and, worse, a **false pass** in the test that was supposed to pin the
  `sympy.Integer` path. Found on A100; would have failed CI.
- **`upsample_nearest2d_backward` is not bit-exact against eager for every ratio.** grad
  64×64 → input 8×8 differs on 155/192 elements, max|d| 3.8e-06. Triaged before trusting it:
  identical with a *static* `input_size` (so unrelated to PR 3), float64 drops it to ~7e-15
  (summation-order noise, not a wrong index — an index error would be dtype-independent),
  and the boundary follows the `avg_pool2d` branch rather than the window size (64→9 is an
  8×8 window and *is* bit-exact). PR 3's test pairs sit inside the exact regime deliberately,
  and the commit message says so.

## 6. Process traps that silently invalidate results

Each of these cost real time and each produced a result that *looked* fine:

1. **Test files must match the wheel's commit, not `main`.** The wheel is release/2.13
   (`cf30153c`); `main`'s tests import `assert_size_stride_grouped`, which that wheel does
   not export — so the suite died at *import* and still exited `rc=0`. **A fake pass.** Use
   `git archive <wheel-commit> test/inductor | tar -x -C /tmp/t213`.
2. **Never toggle patch state while a long suite runs.** A multi-hour run was compiling
   against a mixture of states. Copy the venv (`cp -a` to `/tmp/envpinned`, verify distinct
   inodes) and run long jobs against the copy.
3. **One writer per log.** Two suite instances were launched and both wrote the same file,
   interleaving output. Quarantined rather than trusted.
4. **`/tmp` is not durable here.** It was cleared mid-session, destroying the pinned venv
   and the extracted test tree. The 2621-test log survived only because it was written to
   NFS. Keep long-run artifacts on NFS.
5. **Check patch state before believing anything.** The CPU env was found in
   `pr1 = INCONSISTENT`, which made an issue repro appear to pass. `../tools/state.py`
   does exact-string replacement in both directions and *refuses* on a file matching neither
   form — a state manager that refuses beats one that guesses.
6. **Verify a failure is pre-existing before attributing it.**
   `test_reused_inline_asm_realized` fails on this box with all patches reverted.

## 7. What remains

- [ ] Sign the CLA
- [ ] File Issues A, B, C, D (`upstream/issues.md`); wait for **`actionable`**
- [ ] Re-run the prior-art searches — they age
- [ ] Rebase onto current `origin/main`; re-run the guard matrix afterwards
- [ ] Re-run `pr2_blast_radius.py` — the six hand-guard line numbers and the 13/1157 counts
      are `main`-dependent and will drift
- [ ] `lintrunner -a`; push one branch per PR, Reviewers empty
- [ ] Optional: run the 2621-test suite once with patches OFF for a symmetric comparison

## 8. Claim discipline

The accurate sentence for a CV, SoP, or email — **nothing is merged, nothing is even
pushed**:

> Diagnosed a TorchInductor dynamic-shape crash on a pinned torch version, shipped a
> verified local fix, then confirmed upstream had independently solved the reachable path at
> two layers while the caller-side defect stayed latent on `main` — and prepared three
> upstream fixes plus four issue reports against current `main`, each with a regression test
> that fails when only its own fix is reverted, verified as a strict no-op over 63 real
> upsample configurations on an A100.

**Never** write "fixed a bug in PyTorch," "my PR was merged," or "found an open PyTorch
bug" without saying which of the four findings and what its status is. Issue A is the only
one that is a live user-visible wrong-results bug, and it has no fix.
