# `evidence/logs/` — raw results

Full logs and hash digests, kept unedited. Where a run is **not** usable as evidence,
it is kept anyway and labelled — knowing which runs were invalid, and why, is part of
the result.

## The dynamic-shapes suite (the number PR2 lives on)

`test_torchinductor_dynamic_shapes.py` matters most for PR2, because it exercises
`ops.constant` across hundreds of lowerings under dynamic shapes. Four attempts; only
the last is usable:

| file | status |
|---|---|
| `final_dynshapes_ON.log` | ✅ **THE valid run.** All three patches ON, single instance, pinned venv, wheel-matched tests. `Ran 2621 tests in 4656s`, `FAILED (failures=1, skipped=523, expected failures=2)`. The one failure is `test_unbacked_reduction_cpu`, an **inverted-xfail** (`expected to fail, but actually passed`) that fails identically with all patches reverted — pre-existing, **zero attributable failures**. |
| `final_dynshapes_ON.summary.log` | just the `Ran`/`FAIL` lines from the above, for quick citation |
| `final_dynshapes_ON.tail.log` | the last 60 lines |
| `big_dynshapes_ON.log` | ⚠️ **invalid** — state was toggled by other work *while it ran*, so it compiled against a mixture of states. Kept as the reason for the rule "never toggle state during a long run." |
| `INVALID_two_writers.log` | ⚠️ **invalid** — two instances were launched and both wrote this file, interleaving output. Quarantined by filename. |
| `dynshapes_ON.log`, `dynshapes_PRS_ON.log` | early short runs. The first died at *import* (`cannot import name 'assert_size_stride_grouped'`: `main`'s tests against a release/2.13 wheel) and **still exited `rc=0`** — a fake pass. This is why tests must be matched to the **wheel's** commit, not `main`. |

## No-op digests (the strict-no-op claim)

Each JSON maps 63 real `F.interpolate`/`nn.Upsample` configurations to a hash triple.

| file | what it is |
|---|---|
| `noop_ON.json` / `noop_OFF.json` | patches on vs off. **Result hashes differ in 0/63** — the strict-no-op claim. |
| `ctrl_OFF_a.json` / `ctrl_OFF_b.json` | ⭐ **the A-vs-A control**: two runs of the *identical* state. Result hashes: 0/63 differ. **Generated-code hashes: 50/63 differ** — because they embed cache paths and kernel names. This is the run that proves **code hashes are not usable as evidence**; only result hashes are. Run this control before believing any A-vs-B comparison. |
| `h_stock.json` / `h_patched.json` | earlier per-case **generated-code** hashes, 12 cases, from `../probes/codegen_noop.py`. ⚠️ **The two files are byte-identical — that is the result, not a copy/paste slip.** Unlike `noop_digest.py`, `codegen_noop.py` *normalizes* the code first (`norm()` rewrites AOT IDs, `triton_fused_*` kernel names, and cache-path hashes), which removes exactly the run-to-run noise that makes 50/63 raw code hashes differ in the A-vs-A control. So these show the patch is a no-op **on normalized generated code** for 12 concrete/`scale_factor=` paths. Superseded as evidence by `noop_ON/OFF.json`, which compares **results** and needs no normalization — prefer those when citing. |

## Per-session records

| file | what it is |
|---|---|
| `h100_20260828.md` | H100 re-verification: 6/6 tests and guard matrix. Its initial A100-vs-H100 reading for `37→74` was later explained by literal-vs-symbolic divisor form, not architecture. |
| `blackwell_20260903.md` | RTX PRO 6000 Blackwell verification, including the new SymInt rounding-flag gap and eager CUDA nearest-backward inconsistency. It must not be quoted as B200 data. |
| `blackwell_20260904.md` | fresh Blackwell release-preparation verification: canonical environment reconstruction, stock repros, guard matrix, adversarial checks, Issue A filing ratios, and current upstream gate status. It must not be quoted as B200 data. |
| `noop_blackwell_20260903_ON.json` / `noop_blackwell_20260903_OFF.json` | same-device 63-case Blackwell comparison; eager, compiled, and normalized-code differences are all 0/63 between patch states |
| `perf_blackwell_20260903.json` | controlled RTX PRO 6000 division benchmark; no slowdown resolved here, but it does not supersede the ≈1.8% A100 result |
| `a100_20260828_second.md` | the **second A100** session (`tan-1gpu-chip-w-0-2`): what reproduced unchanged, and the two performance claims it **corrected**. The prose version is `../RESULTS_a100.md` §18 — read that; this is the terse working note. |
| `noop_ON_a100.json` | the 63-case no-op digest from that A100, patches ON. ⭐ Compared against `noop_ON.json` (H100) it shows **0/63 result-hash differences across a hardware change** while 50/63 code hashes differ — the strongest form of the "code hashes are not evidence" result. |
| `perf_controlled.json` | the **controlled** `div_rn` vs `truediv` measurement (CUDA events, 15 A/B/A/B alternations, A-vs-A control first). Verdict: `div_rn` ≈**1.8% slower** at large shapes — **supersedes** the 0.96–0.98× figure in `../RESULTS_a100.md` §10. |
| `issueA_ratio_classes.json` | 3586 upsample ratios classified ULP-robust (53) / one-sided (1634) / immune (1899), so Issue A's filing ratios are chosen by construction. `448→192` is robust; `37→74` is one-sided and therefore sensitive to literal-vs-symbolic divisor form. |

## Reproducing

```bash
python ../noop_digest.py OUT.json            # the digest; the path argument is required
PY=<python> bash ../../tools/guard_matrix.sh # the guard matrix
python ../perf_divrn_controlled.py OUT.json  # the CONTROLLED perf number
python ../issueA_predict_validate.py         # the ULP model vs real F.interpolate
```

⚠️ Run `../../tools/state.py status` first. A number measured in an ambiguous patch
state is worthless.

⚠️ **Probes that instrument *lowering* need a cold inductor cache.** With the cache
warm, `../pr2_blast_radius.py` recorded **0 calls / 0 sites** — which reads as "the
checked code is never reached" when in truth nothing was measured. It now forces its
own cold cache dir and aborts on a zero count. `TORCHINDUCTOR_CACHE_DIR` defaults
into NFS `$HOME` here, shared fleet-wide and warm from earlier sessions.

The dynamic-shapes suite needs tests extracted at the **wheel's** commit:

```bash
git archive cf30153c test/inductor | tar -x -C /tmp/t213
```

⚠️ Keep long-run artifacts on NFS, not `/tmp`. `/tmp` was cleared mid-session twice,
destroying the pinned venv and the extracted tests *after* a multi-hour run; the log
survived only because it was written to NFS.
