# `evidence/` — the harness behind every number

Each script is standalone and portable (no absolute paths, no imports from a
sibling). Run with a python whose `torch` is the one you intend to measure.

**Always run `../tools/state.py status` first.** A number is meaningless if the
patch state it was measured in is ambiguous — an env was once found in state
`pr1 = INCONSISTENT`, which made an issue repro appear to pass.

## Subdirectories

| dir | what it holds |
|---|---|
| `logs/` | raw results, **including the runs labelled INVALID** and why — see `logs/README.md` |
| `probes/` | the ~30 raw exploratory scripts the findings came from — see `probes/README.md` |
| `dev_harnesses/` | the harnesses cited by `../upstream/report.md` sessions C–E. Rough; superseded by the scripts here. ⚠️ **Three are byte-identical copies of `probes/` files** — `stress_configs.py`=`probes/stress.py`, `probe_backward_bug.py`=`probes/probe_bw2.py`, `backward_fix_test.py`=`probes/bw_fix_test.py`. Both names are kept only because `report.md` cites the `dev_harnesses/` ones; there is no difference to look for. |
| `suites/` | the two wheel-matched PyTorch test files used for the regression runs (`test_indexing.py`, `test_torchinductor_dynamic_shapes.py`) |

⚠️ `suites/` must match the **wheel's** commit (`cf30153c`, release/2.13), not `main`.
`main`'s copies import symbols the 2.13 wheel does not export, so the suite dies at
*import* and still exits `rc=0` — a fake pass. See `logs/README.md`.

## Documents

| file | what it holds |
|---|---|
| `RESULTS_a100.md` | ⭐ **the evidence document**, §1–§18: guard matrix, no-op proof, perf, prior-art search, and the corrections where an earlier claim was measured wrong. ⚠️ Named for the A100 session but no longer A100-only — **§17 is the 2026-08-28 H100 re-check** (refreshed prior art, the corrected read of PR #184848, both defects re-verified against `main` from source), and **§18 is a second, independent A100**. Read §18 before quoting any performance number: it corrects §10's `div_rn` figure (≈2% *slower*, not faster) and §13.2's PR2 cost (≈3.9% of lowering time, not 0.98×), locates Issue A's mechanism at bit level, and turns the choice of filing ratios into a measured decision. |
| `logs/h100_20260828.md` | the H100 re-verification: 6/6 tests, full guard matrix, Issue A + the `37→74` architecture difference |
| `logs/noop_{ON,OFF}.json` | result+code hashes over 63 ATen configs, patches on and off |
| `logs/ctrl_OFF_{a,b}.json` | ⭐ the **A-vs-A control**: two runs of the *identical* state. Proves generated-Triton hashes differ in 50/63 cases anyway, so only *result* hashes are load-bearing |
| `logs/final_dynshapes_ON.*` | the 2621-test dynamic-shapes suite (summary + tail) |
| `test_custom_lowering.py` | the PyTorch test file containing all 6 new tests |
| `verify_claims.py` | claim re-checker from the earlier single-PR plan (5/5 on 2.13.0). Superseded, but the *pattern* is worth reusing: a script that re-asserts every claim a document makes. |

## Scripts by what they establish

**The patches are correct**
| script | claim |
|---|---|
| `noop_digest.py` | strict no-op: 63 real ATen upsample configs, bit-identical with/without patches |
| `adversarial_pr1.py` | 42 configs written to *break* PR1 — mixed dims, explicit scales, fp16/bf16, non-contiguous, degenerate. Instruments the predicate to prove the deferred branch actually ran |
| `verify_issue_repros.py` | **run with all patches reverted** — confirms the issue bodies match genuinely stock behaviour |
| `iso_fwd.py` | isolates the forward lowering from the decomposition |

**The design choices are right, not just plausible**
| script | claim |
|---|---|
| `aten_ref_probe.py` | models `UpSample.h` exactly; approximate divide disagrees with eager on 45/3305 coordinates, `div_rn` on 0 |
| `loop_invariant.py` | reads the emitted Triton: the divide depends on `ks0`/`ks1`, never `xindex` — so it is hoisted |
| `perf_divrn.py` | ⚠️ **SUPERSEDED for the headline number** — times each variant once in a fixed order with no clock control, so on a GPU idling at 210 MHz the first variant eats the clock ramp. Gave 0.957–0.984 in one session and 1.009–1.212 in another on the same GPU model. Use `perf_divrn_controlled.py` |
| `perf_divrn_controlled.py` | ⭐ the **controlled** replacement: CUDA-event timing, 15 A/B/A/B alternations, clock stabilization, and an **A-vs-A control first** to prove the harness can resolve the effect. Verdict: `div_rn` ≈**1.8% slower** at large shapes (interval excludes 1.0), unresolvable at small ones — *not* faster. See `RESULTS_a100.md` §18.4 |
| `triton_div_probe.py` | which divide Triton actually emits |
| `pr3_why_guard.py` | defer vs guard: 1 lowering invocation for 7 sizes vs 7 |
| `pr2_blast_radius.py` | 1221 `ops.constant` calls over 24 workloads; 13 sympy-but-concrete → `is_number` is load-bearing. ⚠️ **Forces its own cold cache**: with a warm cache it recorded **0 calls / 0 sites**, which reads as "the check is never hit" when nothing was measured. It now aborts on a zero count. Cost ≈**3.9% of lowering time**, cold-cache and reproducible (§18.5) |

**The bugs are real and pre-existing**
| script | claim |
|---|---|
| `mismatch_repro.py`, `eager_mismatch.py` | Issue A: compiled vs eager pixel mismatch |
| `issueA_divide_position.py` | locates the divide in the decomposition, not the lowering |
| `issueA_runtime_divide.py` | ⭐ **the current explanation of which ratios are visible** (§19.3). One process per ratio; prints the row count under a **literal** divisor (`ks0 / 74`) and a **symbolic** one (`ks0 / ks1`), plus the emitted divide's ULP error, and labels each ratio ROBUST / scope-dependent / clean. Supersedes the "architecture-dependent" reading below: `37→74` flips on **one** GPU depending on Python variable scope. |
| `issueA_arch_divide_bits.py` | why ratios differ at bit level: inductor's `truediv` lands **+1 ULP** above eager for 448→192 / 384→363 but **−1 ULP** below for 37→74. ⚠️ Still valid for the *literal-divisor* form; it is **not** the whole story — see `issueA_runtime_divide.py`. |
| `issueA_ratio_classes.py` | classifies 3586 ratios into ULP-robust (53) / one-sided (1634) / immune (1899), so filing ratios are chosen **by construction**. `448→192` is robust; `384→363` and `37→74` are one-sided — which is the `37→74` trap. ⚠️ Its reference must be **eager fp32**, not exact rational, and ULP steps must walk the **fp32 bit pattern** (`math.nextafter` steps in float64 and is a silent no-op) |
| `issueA_predict_validate.py` | ⭐ the falsification test: the model **commits first**, then real `torch.compile(F.interpolate)` runs. **12/12 exact** on ratios never tested here — row indices *and* substituted pixels — including 5 predicted-clean cases, so it is falsifiable rather than biased toward "wrong" |
| `issueA_candidate_fix.py`, `issueA_fix_validate.py` | ⚠️ a `sym_float` fix that **works forward but costs a gradient** (25/28 → 24/28) — the same failure that killed upstream PR #184848. Why Issue A ships as an issue with no fix |
| `bwd_mismatch_triage.py` | the backward op is not bit-exact for every ratio; float64 drops the diff to ~7e-15, so it is summation-order noise, not a wrong index |
| `adaptavg_preexisting.py` | Issue E reproduces identically with all patches reverted |
| `same_bug_as_175154.py` | proves our Issue A and upstream #175154 are **distinct** defects |
| `eval_175154_attempts.py` | scores all four candidate fixes for #175154 on four axes at once |

**The methodology is sound**
| script | claim |
|---|---|
| `cache_collide.py` | ⭐ the FX-graph-cache collision: same op name across iterations → a false failure *and* a false pass. Read this one before writing any test that registers a lowering in a loop |
