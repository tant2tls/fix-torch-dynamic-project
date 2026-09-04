# CLAUDE.md — `fix_torch_dynamic_project/`

Guidance for Claude Code working in this folder. **Read this first.**

## What this folder is

**The single project for the whole TorchInductor upsample / dynamic-shape
investigation.** Every report, result, and finding lives here — nothing is kept
outside it any more.

**This is the develop phase.** Everything is tracked and visible, including the
internal-facing material (`origin/`, `notebook/`). The **public cut is a separate,
later step** — `PUBLISH.md` records what must be excluded or rewritten at that point.
Do not pre-emptively hide or sanitize things now; an earlier attempt to do so made
the working copy harder to navigate than the leak was worth.

Consolidated **2026-08-28** from `fix_nextfort_compile_dynamic/` and
`../../prwork/`. Both are **superseded** — do not add to them. (The former is now a
2-file build stub, because internal `Dockerfile:63` / `setup.sh:41` still copy
`index_propagation.py` from it by path.)

> **Two readers.** A **professor** reads `README.md` §0 and stops. **You, next
> session**, need `upstream/` and `evidence/`. Optimize `README.md` for the first and
> this file for the second.

## Status (updated 2026-09-04)

| item | state |
|---|---|
| 2.3.1 teaching artifact (`repro/`, `patch/`, `tests/`, `demo.sh`) | **complete and green** — 15 pass unpatched / 18 patched, re-verified on A100 2026-08-28 |
| 3 upstream patches (`upstream/patches/`) | written; apply clean in sequence to `main` @ `b1716d913a` |
| 6 regression tests | 6/6 pass on **A100 (×2), H100, and RTX PRO 6000 Blackwell**; each fails with only its own fix reverted |
| 4 issue drafts (`upstream/issues.md`) | written, **none filed** |
| PR bodies (`upstream/PR{1,2,3}_BODY.md`) | local review drafts; stale perf, code-hash, and PR3 reachability wording corrected 2026-09-04 |
| CLA | **not signed** |
| `lintrunner -a` | **never run** (no PyTorch build on this box) |

**Nothing has been submitted to PyTorch.** The only honest status is *"written,
GPU-verified, staged."*

⚠️ **Two performance claims were CORRECTED on 2026-08-28** (`evidence/RESULTS_a100.md`
§18.4, §18.5). Both were measurement artifacts, and both had propagated into the
submission docs — now fixed in `SUBMIT.md`, `SUBMIT_PR1.md`, `SUBMIT_PR2.md`, `PR.md`,
`issues.md`:

- **`div_rn` is ≈2% SLOWER at large shapes, not 0.96–0.98× faster.** The old figure
  came from timing each variant once in a fixed order on a GPU idling at 210 MHz.
  The structural loop-invariance argument (what actually answers #164144/B200) is
  unaffected.
- **PR2's check costs ≈3.9% of lowering time, not 0.98×.** The old figure straddled
  1.0 across identical runs; cold-cache interleaving makes it reproducible.

Neither correction touches a patch, a test, or a fix — only claim wording.

## The process gate that governs everything upstream

PyTorch's current policy says a new contributor's linked issue must already be
labelled **`actionable`**. Treat the stored patches as internal experiments:
**file the human-authored issue → wait for the label → prepare a fresh PR**.
Only after the label should the work be rebased onto current `main` and run
through focused tests, broader tests, lint, type, and pre-commit checks.

Sign the CLA; keep one concern per PR. Leaving Reviewers empty is this repo's
submission plan, not a quotation from `CONTRIBUTING.md`. Any AI-generated
material used on GitHub must be disclosed and contained with human commentary.
Never post raw assistant output or AI-written solution text in an issue, and
never file, comment, push, or open a PR autonomously. Details:
`upstream/issues.md`.

## Layout — six directories, each with its own README

```
README.md      ⭐ THE PROFESSOR-FACING DOCUMENT. §0 motivates from the real
               ComfyUI/SDXL/VAE dynamic-compile failure, then §1–§9 walk
               fail -> root cause -> fix -> verify -> how it was localized.
PR.md          the upstream story (three PRs, four issues) in narrative form
CLAUDE.md      this file
PUBLISH.md     what to exclude/rewrite when the public cut is made (LATER)

origin/        ⭐ the primary sources: the 711-line PRODUCTION TRACEBACK, the
               original brief, the first write-up, the stock 2.3.1 file
repro/         the 2.3.1 reproduction (bug.py, repro.py, ablation.py,
               why_it_survives.py, unet_like.py, verify_fix.py, check_upstream.py)
patch/         patch.py apply|revert|status for the 2.3.1 one-line fix
tests/         state-aware suite + pytest shim (torch stays the only dependency)
demo.sh        10 steps: fail -> explain -> fix -> verify; restores its state

upstream/      the three PRs: README (send order + provenance), issues.md,
               SUBMIT*.md, PR*_BODY.md, patches/, commit_msgs/, final/,
               report.md (the lab notebook, sessions A–E)
evidence/      RESULTS_a100.md + 21 portable scripts + logs/ (raw results,
               including the labelled INVALID runs) + probes/ (the raw
               exploratory scripts) + dev_harnesses/ + suites/
tools/         state.py (the ONLY sanctioned patch toggle), guard_matrix.sh
notebook/      superseded drafts, kept so decisions can be re-examined
```

Each of `origin/`, `upstream/`, `evidence/`, `evidence/logs/`, `evidence/probes/`,
`tools/`, `notebook/` has a README explaining which file answers which question.
**Start from those rather than guessing from filenames** — several scripts have
near-identical names (`probe_bw2.py` … `probe_bw5.py`).

## The four findings

1. **⭐ Issue A — WRONG PIXELS, no upstream-ready fix.** Stock torch 2.13, no
   patches, no custom op: `F.interpolate(mode="nearest")` under `dynamic=True`
   on **CUDA** picks a different source pixel than eager. CPU and static compile
   are exact. It lives in the **decomposition**, not the lowering the PRs touch.
   The scale divide is loop-invariant; the earlier per-element claim was wrong.
   Both a decomposition fix and a symbolic-printer prototype repair forward but
   reduce the gradient matrix from 25/28 to 24/28. The reason is upstream of
   Inductor: native eager CUDA nearest backward is not the transpose of its own
   forward map at sensitive ratios. This reproduces the open #97135
   `needs reproduction` symptom; add a human-authored reproduction there and
   coordinate forward/backward semantics before preparing a fix.
2. **Issue B** — `upsample_nearestnd` crashes on a symbolic output size → **PR1**.
3. **Issue C** — `ops.constant` accepts a symbolic value, fails ~12 frames later → **PR2**.
4. **Issue D** — `upsample_nearest2d_backward` crashes on symbolic `input_size` → **PR3**.
5. **Issue E** — `adaptive_avg_pool2d`: **duplicate of upstream #159550, do not file.**

**Send order: PR3 → PR1 → PR2.** PR3 is 7 lines with no design argument; PR2 changes
a shared contract and will attract the most debate.

⚠️ **PRs 1 and 3 are NOT ATen-reachable today** — the decompositions fire before the
lowering. Say so explicitly in both. Overclaiming reachability loses the review.

## Rules learned the hard way — violating these silently invalidates results

- **`tools/state.py` is the only sanctioned way to toggle patches.** Exact-string
  replacement both directions; it **refuses** on a file matching neither form. Run
  `state.py status` before drawing *any* conclusion — an env was once found
  `pr1 = INCONSISTENT`, which made a repro appear to pass.
- **Never toggle state while a long suite runs.** A multi-hour run was rendered
  worthless that way. `cp -a` the venv for long runs.
- **Match test files to the WHEEL's commit, not `main`.** `main`'s tests import
  symbols the 2.13 wheel does not export, so the suite dies at *import* and still
  prints `rc=0` — a fake pass.
- **Generated-Triton hashes are NOT no-op evidence.** They differ in **50/63** cases
  between two runs of the *identical* state. Only **result** hashes count (63/63
  identical). Run the A-vs-A control first (`evidence/logs/ctrl_OFF_{a,b}.json`).
- **⭐ A warm inductor cache silently zeroes any lowering-time probe.** `ops.constant`
  only runs while a graph is **lowered**, so with the cache warm
  `pr2_blast_radius.py` recorded **0 calls / 0 sites** — which *reads as* "the check
  is never reached", the exact opposite of PR2's justification, when in truth
  nothing was measured. `TORCHINDUCTOR_CACHE_DIR` defaults into **NFS `$HOME`**,
  shared fleet-wide and warm from earlier sessions. Cold: 1221 calls / 13
  sympy-concrete. The probe now forces its own cold dir and **aborts on a zero
  count**; copy that pattern into any new lowering-time probe (`repro/repro.py`
  already does the same redirect).
- **⭐ Benchmark a divide with an A-vs-A control, or don't quote the number.** This
  box idles at **210 MHz against 1410 MHz**, so a script that times each variant
  **once, in a fixed order** (as `perf_divrn.py` does) charges the clock ramp to
  whichever ran first. That produced `div_rn/truediv` = 0.957–0.984 in one session
  and 1.009–1.212 in another **on the same GPU model**. Use
  `evidence/perf_divrn_controlled.py`: CUDA events, ≥15 alternations, clock
  stabilization, control run first. Real answer: ≈**2% slower** at large shapes.
- **A ratio-vs-eager model must use eager's arithmetic.** Comparing against exact
  rational `(d*i)//o` measures eager's own (expected) departure from rational math
  and **inverted** a result (`448→192` reported as `0/192`). Eager is fp32:
  `min(int(f32(d * f32(i/o))), i-1)`. Relatedly, `math.nextafter` steps in
  **float64** — rounding back to fp32 lands on the same value, so a ULP
  perturbation built on it is a silent no-op. Step the **bit pattern**.
- **A test registering a lowering per loop iteration must vary the custom op NAME.**
  The FX graph cache keys on the op name, so a flag in the lowering's closure is
  invisible to it. This produced a false failure *and* a false pass.
- **⭐ Never trust a snapshot directory; derive from git.** The first consolidation
  copied a `final/` snapshot that still contained the **abandoned type-punned
  `inv_scales`** (the silent wrong-numerics blocker) and the cache-collision test bug.
  `upstream/final/` is now verified byte-identical to branch `prseries`, and the
  patches are verified to reproduce it exactly. Re-check after any change.
- **Read GitHub labels from the API, not a rendered page.** The claim that #175154
  was `actionable` came from a page summary; it is not.
- **Re-probe hardware every session.** The GPU changed A100 → H100 mid-task on
  2026-08-28. Never inherit a GPU conclusion.
- **Verify a failure is pre-existing before blaming your change.**
  `test_reused_inline_asm_realized` fails on these boxes with all patches reverted.

## Hardware — do not inherit, re-probe

| when | box | covers |
|---|---|---|
| 2026-08-25/27 | H100 80GB (×2/×4, varying) | the 2.3.1 artifact; `upstream/report.md` sessions A–E |
| 2026-08-26/27 | 1× **A100**-SXM4-80GB | `evidence/RESULTS_a100.md` §1–§16: full evidence, 2621-test suite, perf, adversarial sweep |
| **2026-08-28** | 1× **H100** 80GB HBM3 | `evidence/logs/h100_20260828.md`: 6/6 tests, full guard matrix, Issue A; plus the whole 2.3.1 demo re-run |
| **2026-08-28** (later, `tan-1gpu-chip-0-1`) | 1× **H100** 80GB HBM3 | `evidence/RESULTS_a100.md` **§17**: prior art refreshed, PR #184848 authorship corrected, both defects re-verified against `main` from source, 2.3.1 artifact re-run **15/0 → 18/0** |
| **2026-08-28** (later still, `tan-1gpu-chip-w-0-2`) | 1× **A100**-SXM4-80GB sm_80 | `evidence/RESULTS_a100.md` **§18**: second independent A100. 6/6 tests, 42/42 adversarial, 63-case no-op, `37→74` = 36/74 again. **Two perf claims corrected** (§18.4 `div_rn`, §18.5 PR2), Issue A's mechanism located at **bit level**, and a validated **predictor** for choosing filing ratios. 2.3.1 artifact re-run 15/0 → 18/0 |
| **2026-08-30** (`tan-1gpu-chip-0-2`) | 1× **H100** 80GB HBM3 | `evidence/RESULTS_a100.md` **§19**: strict no-op re-measured as a **same-box ON/OFF A/B** (63/63 compiled results identical — cross-hardware digest comparisons are invalid, the eager hashes differ too), guard matrix 6/6 all four states, 42/42 adversarial, `37→74` re-explained as **divisor-form** dependence, prior art re-read from the **GitHub API**, Issue A's snippet run verbatim. Two documentation defects fixed. 2.3.1 artifact re-run 15/0 |
| **2026-09-03** | 1× **RTX PRO 6000 Blackwell Server Edition**, sm_120 | `evidence/logs/blackwell_20260903.md`: guard matrix 6/6, 42/42 adversarial, same-device 63-case no-op, controlled perf, SymInt rounding-flag gap, and eager CUDA backward inconsistency. This is Blackwell evidence, not B200 evidence. |
| **2026-09-04** | 1× **RTX PRO 6000 Blackwell Server Edition**, sm_120 | `evidence/logs/blackwell_20260904.md`: canonical environment rebuilt, stock findings re-run, guard matrix 6/6, 42/42 adversarial, issue ratios and follow-up invariants rechecked, and GitHub labels/duplicate searches refreshed. |

⚠️ **`37→74` is not architecture-dependent — it is DIVISOR-FORM dependent (corrected
2026-08-30, `RESULTS_a100.md` §19.3).** The earlier reading ("36/74 on A100, 0/74 on
H100, so the SM decides") was measured correctly but explained wrongly. On **one** H100
the same ratio gives **0/74 or 36/74** depending on which divide Inductor emits:

| output size written as | emitted | `37→74` |
|---|---|---|
| a module **global** | `ks0 / 74` (divisor literal) | **0/74** |
| a closure **local** | `ks0 / ks1` (both runtime) | **36/74** |

Only the second form puts both operands through the hardware reciprocal. So plain
Python scoping in the *repro script* flips the verdict — which is why the same ratio
read differently on different boxes.

**File Issue A with `448→192`, `384→363`, `448→368`, `41→82`** — all four are wrong under
**both** forms, on A100 and H100. **Do not include `37→74`**: it is clean in the
conservative form and reads as non-reproducible. `evidence/issueA_runtime_divide.py`
prints both columns per ratio (one process each) and labels every ratio ROBUST /
scope-dependent / clean.

The §18.2 ULP-sign model still holds for the literal-divisor form and is still what
explains *why* `41→82` fails while `74→148` does not — the two are complementary, not
competing. What §18.2 could not explain is a single GPU giving two answers.


**Why, as of §18.2 — the mechanism, not just the observation.** Inductor's
`truediv` is an approximate reciprocal, and its error has a **sign**: it lands
**+1 ULP above** eager for `448→192`/`384→363` (compiled picks a *higher* source
pixel) but **−1 ULP below** for `37→74` (a *lower* one). A GPU whose divide errs
high reproduces the first two and not the third. `37→74`'s scale is `0.5`, exactly
representable, so fp32 precision cannot explain it at all — that case is *purely*
the reciprocal. **`448→192` is additionally ULP-robust** (a row moves under an
error of either sign), which is why it is the right headline on evidence rather
than on luck. `evidence/issueA_ratio_classes.py` classifies 3586 ratios this way;
`evidence/issueA_predict_validate.py` proves the model predicts unseen ratios
**12/12 exactly**. Other robust candidates: **448→368 (9 rows)**, 448→384 (7),
500→120 (6).

## How to re-verify

The default environment for upstream work in this checkout is the Conda
environment `dynamic`; invoke it with `conda run -n dynamic ...` so commands do
not depend on shell activation. Reconstruct it from `requirements-upstream.txt`.
Keep the torch 2.3.1 artifact in a separate environment built from
`requirements.txt`.

The 2.3.1 artifact (needs a torch 2.3.1 env):

```bash
python patch/patch.py status     # ALWAYS first
python repro/repro.py            # exit 2 = bug reproduced
python repro/ablation.py         # 4/9 FAIL; scale_factor= passes, size= fails
python patch/patch.py apply && python repro/repro.py   # exit 0
python tests/run_tests.py        # 15/0 unpatched, 18/0 patched
python patch/patch.py revert
```

The upstream PRs (needs a torch 2.13 env whose torch you may patch):

```bash
PY=<that python>
$PY tools/state.py status
PY=$PY bash tools/guard_matrix.sh    # ~15 min: the central claim
$PY evidence/noop_digest.py          # strict no-op, 63 ATen configs
$PY evidence/adversarial_pr1.py      # 42 configs designed to break PR1
```

`evidence/verify_issue_repros.py` must run with **all patches reverted**.

## Next actions, in order

1. **Re-run every duplicate search and current label check.** The saved results age;
   use GitHub's current issue state, not a cached rendered summary.
2. **Write Issues A, B, C, and D in my own words.** Keep them problem-focused. For
   Issue A, use robust `448→192`, `384→363`, `448→368`, and `41→82`; omit the
   divisor-form-sensitive `37→74`. Cite #185806 as defect-class precedent, not as a
   duplicate. Do not include AI-generated solution explanations.
3. **Comment on existing #97135** with the new eager forward/backward invariant repro;
   do not open a duplicate CUDA-nearest-backward issue.
4. **Wait for `actionable`.** Do not upload the stored patches or open draft PRs
   while the associated issues are unlabelled.
5. **Only after that gate**, sign the CLA, create fresh branches from current `main`,
   rebuild, re-run focused and broad tests plus lint/type/pre-commit checks, personally
   review the diffs, then push separate PRs in the planned PR3 → PR1 → PR2 order.
6. **Public cut — LATER.** `PUBLISH.md` lists exclusions and rewrites.
7. Optional: the unrelated shape-variety sweep is remembered, never measured; do not
   conflate it with this bug.

## Claim discipline — the highest-risk sentences in the repo

- The 2.3.1 bug was a **compile-time crash, not a miscompilation**. Numerics were
  bit-exact throughout. Never write "compiled incorrectly."
- It is **already fixed upstream in torch ≥ 2.4** (dispatch layer). Never claim an
  open PyTorch bug from it.
- **Never write "fixed a bug in PyTorch"** or "my PR was merged." Nothing is pushed.
- The accurate sentence: *diagnosed a compiler bug on a pinned torch version, shipped
  a verified local fix, confirmed upstream had independently closed the reachable path
  at two layers while the caller-side defect stayed latent on `main`, and prepared
  three upstream fixes plus four issue reports — each with a regression test that
  fails when only its own fix is reverted.*
- Every number carries its hardware and method. Keep **verified here** /
  **original context** / **remembered** distinct; `README.md` §9 is the template.
