# Sending these three PRs

Everything here was verified on **1× A100-SXM4-80GB** (driver 575.57.08), torch
**2.13.0+cu130** / triton 3.7.1 / python 3.12.14, plus CPU checks on 2.13.0+cpu.
Full evidence: `../evidence/RESULTS_a100.md`. Issue drafts: `issues.md`.

**Per-PR submission docs — read the one for the PR you are sending:**

| doc | PR | send order |
|---|---|---|
| `SUBMIT_PR3.md` | backward guard (`+7`) | **first** |
| `SUBMIT_PR1.md` | forward fix (`+30/−4`) | second |
| `SUBMIT_PR2.md` | `ops.constant` contract (`+23`) | last |

This file is the shared context: the process gate, the stack, and the cross-cutting
evidence. The three per-PR docs carry the argument, the objections, and the claim limits
specific to each.

## Read this first: file the issues, then the PRs

PyTorch will not review a PR from a new contributor without a linked issue carrying
the **`actionable`** label:

> "Only PRs that address issues labeled `actionable` will be considered for review."
> "you must wait for a maintainer to review it and mark it actionable before preparing
> and sending a PR for it."

So: **write and file Issues A, B, C, D → wait for `actionable` → then prepare the PRs.**
Add the eager backward reproduction as a comment on existing #97135 rather than opening
a duplicate. Do not push first. `issues.md` has working notes, prior-art search, and
the filing order; rewrite the issue text in your own words before filing.

## ⚠️ `AI_POLICY.md` — read before writing a single word of a PR or issue

Verified verbatim against `https://raw.githubusercontent.com/pytorch/pytorch/main/AI_POLICY.md`
on **2026-08-30**. Three rules bear directly on this submission:

> "*AI-generated content in comments, issues, or PRs must be clearly disclosed and
> contained* (e.g. using a code block or a quote block). AI-generated content must be
> accompanied by human commentary explaining its relevance."

> "*For pull requests, please carefully read the code before submitting it for review*,
> especially if AI tools helped write or rewrite it. Make sure the implementation is
> something you understand… Before submitting, simplify where possible and make sure the
> core change is easy for maintainers to review."

> "All contributions should involve a human who understands and can take responsibility
> for the work produced with AI assistance. *We do not accept contributions created by
> fully autonomous agents*, and we may close pull requests that appear to have been
> generated without meaningful human involvement."

**What this means concretely, and it is not optional:**

1. **Disclose AI assistance** wherever it was used, in the PR description. A single
   honest sentence is enough — see the template at the bottom of each `PR*_BODY.md`.
   Non-disclosure is the violation; using the tools is not.
2. **You must be able to defend every line under review.** A reviewer will ask why
   `div_rn` and not `truediv`, and why PR1 defers while PR3 guards. Both answers are in
   `SUBMIT_PR1.md`/`SUBMIT_PR3.md` — read them until you can give them without the doc.
   If you cannot, do not send that PR.
3. **Do not let the PRs read as agent-generated.** This is not hypothetical: PR
   **#184848** in our own prior art was labelled `agentic` and closed by its author in a
   bulk cleanup. Verbose, over-hedged, exhaustively-enumerated prose is the tell — the
   `PR*_BODY.md` drafts are deliberately trimmed for this reason.
4. **The issue bodies carry no fix explanation at all** — `CONTRIBUTING.md` separately
   forbids AI-generated solution text in an issue. Observed behaviour + minimal repro +
   expected vs actual, nothing more.

## The stack

Three commits (branch `prseries` in a local pytorch checkout) on `b1716d913a`:

```
cee885e016  [inductor] Fix symbolic output_size in upsample_nearestnd     -> Issue B
3d9d172c6f  [inductor] Reject symbolic values in ops.constant             -> Issue C
c87c2a8580  [inductor] Guard input_size in upsample_nearest2d_backward    -> Issue D
```

`+299 / −5` across 3 files:

| file | + | − |
|---|---|---|
| `torch/_inductor/lowering.py` | 41 | 5 |
| `torch/_inductor/virtualized.py` | 23 | 0 |
| `test/inductor/test_custom_lowering.py` | 239 | 0 |

**Three separate PRs, not one.** My call, not a quoted rule — "one concern per PR" is
not in `CONTRIBUTING.md` or the wiki (checked 2026-08-30). The reasoning: B is a lowering fix, C is a
cross-cutting contract change, D is an unrelated function. Bundled, one objection
stalls all three.

**Suggested order — send D first.** It is 7 lines plus one test, needs no design
argument, and establishes the account. Then B. Then C, which is the one likely to
attract debate because it changes a shared contract.

Mention in both B and C that they interact: C enforces the contract that B is the only
in-tree violator of. They are independent (either can land alone) but a reviewer
should not be surprised.

## What each PR claims, and the limit of each claim

**PR 1 / Issue B — forward lowering.** Defers the divide to the kernel instead of
guarding the output size. Guarding would specialize per output size and, past
`recompile_limit`, drop the caller to eager.
- Not ATen-reachable: `upsample_nearest{1,2,3}d` are in
  `torch._decomp.decomposition_table`, so the decompositions fire before lowering —
  measured **0 lowering hits** across 8 entry points. **Say this explicitly.** Do not say
  "all 12 overloads are CompositeImplicitAutograd"; measured on 2.13.0 there are **19**
  overloads and only **6** are (the `.vec` overload of each of the six nearest/nearest-exact
  ops), while **18** are in the decomposition table — and a
  reviewer who knows this code will check. Overclaiming reachability is the fastest way to
  lose review.
- `div_rn` not `truediv`: modelling `UpSample.h` exactly over 9 ratios / 3305
  coordinates, the approximate divide disagrees with eager on **45**, `div_rn` on **0**.
- Cost: the divisor is a kernel argument, so the divide is **loop-invariant** — visible
  in the emitted Triton, which computes it from `ks0`/`ks1` and never from `xindex`.
  Measured `div_rn/truediv` ≈ **1.02 at large shapes** (interval 1.016–1.030, excludes
  1.0); unresolvable at small, launch-bound shapes. So: **≈2%, not free, not faster.**
  ⚠️ The older "0.96–0.98, not slower" figure was a clock-ramp artifact of timing each
  variant once in a fixed order — see `RESULTS_a100.md` §18.4 and use
  `evidence/perf_divrn_controlled.py`.
  ⚠️ Measured on A100 only. #164144's regression was on B200. Do not claim B200.

**PR 2 / Issue C — `ops.constant` contract.** Rejects a symbolic value at the call
instead of ~12 frames downstream in `fx/proxy.py`.
- Frame it as *codifying an existing convention*: `_full` and the addcmul/addcdiv
  helpers already hand-dispatch on exactly this.
- Implemented as an explicit `OpsWrapper.constant`, not a check in `_default`, so no
  other op pays for it.
- `sympy.Integer` / `sympy.Float` are `sympy.Expr` but concrete; `is_number` keeps them
  accepted, and a test pins that.
- This is the PR most likely to get pushback ("why here, why not a debug assert").

**PR 3 / Issue D — backward guard.** Guards `input_size` the way the input sizes above
it are already guarded.
- Note *why* it guards rather than deferring, since PR 1 does the opposite: the
  quotient is a Python `range()` bound over which the pooling window is unrolled, so it
  must be concrete at lowering time.
- Not autograd-reachable, and get the reason right: the backward op is *not* in
  `decomposition_table`; the **forward** is decomposed, so the backward graph gets
  `_unsafe_index_put` and this op is never emitted (0 hits across 8 autograd configs). The
  test calls the aten op directly.

## Evidence to put in each PR description

- **Guard matrix** — each test fails with only its own fix reverted:

  | state | forward | concrete-sympy | one-graph | const-reject | const-accept | backward |
  |---|---|---|---|---|---|---|
  | all ON  | OK | OK | OK | OK | OK | OK |
  | PR1 off | FAIL | OK | FAIL | – | – | OK |
  | PR2 off | OK | – | – | FAIL | OK | – |
  | PR3 off | OK | – | – | – | – | FAIL |

- **Strict no-op** — 63 real `F.interpolate`/`nn.Upsample` configurations (size= and
  scale_factor=, 1-D/2-D/3-D, static and dynamic, forward and autograd backward):
  compiled results bit-identical with and without the patches, **0/63 differ**.

- **Adversarial sweep of PR 1** — 21 configurations × {cpu, cuda}, **42/42 bit-exact**
  (`../evidence/adversarial_pr1.py`): mixed symbolic/concrete dims, explicit `scales_x` on one
  dim only, nearest-exact at ULP-sensitive ratios, 1-D and 3-D, float16/bfloat16,
  non-contiguous input, degenerate sizes (output 1, input 1, equal), and downsampling.
  Coverage verified by instrumenting the predicate — the deferred `div_rn` branch
  actually ran (`defer_per_dim=(True, True)`) rather than silently taking the concrete
  path. This is worth quoting in PR 1: it is much stronger than the six shipped tests
  alone, and it pre-empts "did you try mixed dims / explicit scales / non-2D".

- **Regression suites**, both states, identical: `test_indexing` 56 OK,
  `test_optimize_indexing` 15 OK, `test_codegen_triton` 16 OK (2 skip).

- **Full dynamic-shapes suite, patches ON: `Ran 2621 tests`, 1 failure.** That failure is
  `test_unbacked_reduction_cpu`, an inverted-xfail (`expected to fail, but actually
  passed`) that fails **identically with all three patches reverted** — pre-existing, zero
  attributable failures. This is the number PR 2 lives on.
  ⚠️ Only the ON arm was run at this scale; the OFF arm was verified for that one test.

⚠️ **Do not quote generated-Triton-source hashes as no-op evidence.** They differ in
50/63 cases between two runs of the *identical* state (they embed cache paths and kernel
names). Only result hashes are load-bearing. The control run that establishes this is
`ctrl_OFF_a.json` vs `ctrl_OFF_b.json`.

## Steps

```bash
# 0. CLA first: https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md
# 1. fork, then in your pytorch checkout:
git remote add fork git@github.com:<you>/pytorch.git

# 2. REBASE — upstream HEAD moved twice in a single session previously
git fetch origin main
git rebase origin/main prseries      # resolve, then re-run the tests below

# 3. lint (required; CI runs it)
lintrunner -a   # or: spin lint

# 4. run the six tests on a GPU box
python test/inductor/test_custom_lowering.py -v \
  TestCustomLowering.test_upsample_nearestnd_symbolic_output_size \
  TestCustomLowering.test_upsample_nearestnd_concrete_sympy_output_size \
  TestCustomLowering.test_upsample_nearestnd_symbolic_output_size_one_graph \
  TestCustomLowering.test_ops_constant_rejects_symbolic_value \
  TestCustomLowering.test_ops_constant_accepts_concrete_values \
  TestCustomLowering.test_upsample_nearest2d_backward_symbolic_input_size
# expect: Ran 6 tests ... OK

# 5. THE STEP THAT MAKES THEM REGRESSION TESTS — revert only the source hunk,
#    keep the tests, and watch them fail. Do this per PR before pushing.
#    `../tools/state.py` automates it against an installed torch:
python ../tools/state.py set pr1=off      # then re-run -> forward + one-graph FAIL
python ../tools/state.py set pr1=on pr2=off   # -> const-reject FAILs
python ../tools/state.py set pr2=on pr3=off   # -> backward FAILs
python ../tools/state.py set pr1=on pr2=on pr3=on

# 6. push one branch per PR
git push fork prseries:upsample-symbolic-output-size
```

## Pre-push checklist

- [ ] CLA signed
- [ ] Issues B/C/D filed and marked **`actionable`** — this gates everything
- [ ] Rebased onto current `origin/main`; tests re-run after the rebase
- [ ] `lintrunner -a` clean
- [ ] Guard matrix re-run post-rebase (step 5)
- [ ] Each PR body links its issue and states its reachability limit
- [ ] **Reviewers list left empty** — the triage squad assigns; do not @-mention
- [ ] Re-probe the GPU and re-run numbers if the container was rescheduled
- [ ] `test_reused_inline_asm_realized` fails on this box with all patches reverted —
      pre-existing and unrelated; do not report it as caused by these changes

## Held back deliberately

- **Issue A** (wrong pixels from plain `F.interpolate` under `dynamic=True` on CUDA) is
  the strongest finding but has **no upstream-ready fix**. Its emitted scale divide is
  loop-invariant. A forward-only prototype still loses one gradient case because native
  eager CUDA backward is not always the transpose of its own forward map. Add the fresh
  reproduction to existing #97135, then coordinate forward and backward semantics before
  attempting a fix.
- **Issue E** (`adaptive_avg_pool2d` under dynamic shapes) — pre-existing, confirmed
  identical with all patches reverted. Hold until the others are triaged; five issues at
  once from a new account reads worse than three good ones.
