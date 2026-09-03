> ⚠️ **SUPERSEDED — kept for the record only.** This described the earlier *two-issue*
> plan. Current plan: `../../upstream/README.md` (four issues, send order PR3 -> PR1 -> PR2).
> Its relative links are stale by design; `report.md` now lives at `../../upstream/report.md`.

# Upstream contribution: three PRs, two issues, issues first

Two real TorchInductor bugs of the same defect class, plus the diagnostic that generalizes
their failure mode — packaged the way PyTorch asks for it. **Nothing has been sent.**

PyTorch requires an issue labeled **actionable** before a PR will be reviewed, so the order
is: file issue → wait for label → PR. Details in
[report.md §10.6](../../upstream/report.md) — path fixed after this draft was moved into
`notebook/superseded/`; the filenames below still refer to the old flat layout.

```
ISSUE.md      file FIRST  -> PR #1   (forward lowering; repro verified on stock torch)
ISSUE3.md     file SECOND -> PR #3   (backward lowering; independent bug)
PR1_BODY.md   paste-ready body — the forward fix
PR3_BODY.md   paste-ready body — the backward fix
PR2_BODY.md   paste-ready body — the diagnostic (propose on the thread AFTER #1 lands)
0001-*.patch  PR #1: lowering.py    +24 −2 · test +124 −1
0002-*.patch  PR #2: virtualized.py +23 · test +26
0003-*.patch  PR #3: lowering.py     +7 · test +34
report.md     the lab notebook: every measurement, every wrong turn, all three sessions
dev/          the harnesses — see report.md §6 and §10
```

Verified against `main` @ **`7337468`**: PR #1 applies standalone; #2 and #3 apply in
sequence (they extend the same test file). All measurements below are on **2× H100 80GB,
torch 2.13.0+cu130, CUDA 13.0** unless marked CPU.

> ⚠️ **Read `AI_POLICY.md` before posting.** PyTorch requires AI-assisted content to be
> disclosed and contained, and does not accept contributions from fully autonomous agents.
> Treat every `*_BODY.md` and `ISSUE*.md` here as a draft: read the code, confirm you can
> defend each claim, rewrite in your own voice.

---

## PR #1 — forward: symbolic `output_size`

`upsample_nearestnd` guards its input sizes but not its output sizes, divides one by the
other, and passes the quotient to `ops.constant`, which needs a concrete value:

```python
i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]   # concrete
o_sizes = output_size                                        # not concrete
inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
...
x = ops.mul(x, ops.constant(scale, torch.float32))
```

→ `NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>`.

**Fix:** keep `i / o` when `o` is concrete; when symbolic, defer the division to the kernel
via **`ops.div_rn`**.

- **Why `div_rn`, not `truediv`?** Same reason `_floor_div_floating` already uses `_div_rn`,
  in its own words: *"Triton's default division uses an approximate reciprocal, which can
  produce a result slightly below the true quotient and cause floor() to round down by
  one."* Eager computes this scale as an fp32 divide (`compute_scales_value`,
  `ATen/native/UpSample.h`). Confirmed on GPU: the kernel contains
  `triton.language.div_rn`, and **30/30** shape×mode cases are bit-exact.
- **Why defer, not guard?** Guarding yields a *faster* kernel (1.59 ms vs 12.7 ms for one
  8×32×512²→2048² call) but specializes per output size and, past `recompile_limit`, drops
  to eager. Over 12 sizes × 5 iters: **7.09 s / 8 graphs** guarded vs **0.81 s / 1 graph**
  deferred. Crossover at one fixed size ≈ **213 iterations**. The 8× is the *dynamic shape*,
  not the rounding mode — a plain `truediv` measures 12.70 ms. PR #1 states all of this.
- **Nobody regresses:** the deferred path only fires when the size is symbolic, which today
  means the compile crashes.

## PR #3 — backward: symbolic `input_size` (a second, independent bug)

Found by auditing siblings for the same asymmetry. `upsample_nearest2d_backward` guards the
grad's spatial sizes but not `input_size`, then feeds it to `ceildiv`, whose result becomes a
`range()` bound in `_adaptive_pooling_fn`:

```
TypeError: 'FloorDiv' object cannot be interpreted as an integer
```

(`input_size` arrives as `(s28, s28)`; `ceildiv(64, s28)` → `((s28 + 63)//s28)`.)

**Fix: guard it** — the *opposite* choice from PR #1, deliberately. The quotient is a Python
loop bound consumed at lowering time, so there is nothing to defer it to. Both patches say so
in comments. Measured: baseline **0/6** bit-exact (6/6 crash) → fixed **6/6**; autograd
gradients **20/20** bit-identical.

## PR #2 — the contract, enforced once

`ops.constant` takes a concrete leaf; violations surface ~12 frames downstream in
`fx/proxy.py`, naming neither the op nor the remedy. **Not a new rule** — four lowerings
already hand-guard it (`_full`, `addcmul`, `addcdiv`, `alpha` in `add`/`sub`) with the comment
*"Use index_expr for sympy expressions … constant otherwise"*. Both bugs above are sites that
omitted it. Enforced in an explicit `OpsWrapper.constant` so no other op pays: 124 ns/call,
**25/25** lowerings unchanged static and dynamic. PR #2 raises the "gate it behind config?"
question itself.

## Evidence

Measured against real installed torch, not re-implementations.

| claim | result | where |
|---|---|---|
| generated Triton uses `div_rn` | **yes** | H100 |
| symbolic output size, with fix | **30/30** bit-exact (15 pairs × 2 modes) | H100 |
| 20 distinct output sizes | **1 graph**, 20/20 bit-exact | H100 |
| ATen `interpolate` paths | **12/12** unchanged | H100 |
| generated code, existing paths (shipped patch) | **12/12 byte-identical** | H100 |
| perf: guard vs defer, single call | 1.59 ms vs 12.7 ms | H100 |
| perf: 12 sizes × 5 iters, end to end | 7.09 s / 8 graphs vs **0.81 s / 1 graph** | H100 |
| configs (`cpp_wrapper`, cudagraphs, max_autotune, both `division_rounding`) | all bit-exact | H100 |
| dtypes fp16 / bf16 / fp64, ranks 1-D / 3-D | all bit-exact | H100 |
| backward bug, baseline → fixed | **0/6 → 6/6** bit-exact | H100 |
| backward autograd no-regression | **20/20** gradients bit-identical | H100 |
| PR #1 tests, patched / unpatched | **2 pass / 0 pass** | CPU + H100 |
| PR #3 test, patched / unpatched | **pass / fail** (`TypeError`) | CPU + H100 |
| approx-reciprocal index flips | **11695** (i,o) pairs in 1..512 | CPU (numpy) |

```bash
python upstream/dev/triton_verify.py                        # div_rn on Triton + bit-exactness
python upstream/dev/check_installed_fix.py --device cuda     # the forward fix, end to end
python upstream/dev/backward_fix_test.py                     # PR #3, baseline vs fixed
python upstream/dev/perf_attrib.py                           # the 8x, attributed
python upstream/dev/tradeoff.py                              # guard vs defer, end to end
python upstream/dev/stress_configs.py                        # configs, dtypes, ranks
python upstream/dev/run_pr1_tests.py                         # PR #1 tests, state-aware
python upstream/verify_claims.py                             # independent re-derivation
```

## Before pushing

No open verifications remain — the Triton `div_rn` path is confirmed on an H100. Still not
done here: **`lintrunner -a`** (no PyTorch build on this box), and executing PyTorch's own
test files (not authorised; bodies validated verbatim via `dev/`, and the file `ast.parse`s
clean). Run `python test/inductor/test_custom_lowering.py -k upsample` in your fork first.

Per PR: file the issue, wait for `actionable`, sign the CLA, fork, **rebase** (HEAD has moved
five times across four sessions), `git am`, run the test, **revert only the code hunk and
re-run to watch it fail**, `lintrunner -a`, push. Leave Reviewers empty — no merge rule covers
`lowering.py`, so triage assigns; label `module: inductor`.

**Claim discipline.** Until something merges: *"filed two issues and submitted fixes plus
regression tests to PyTorch for latent type-contract violations in Inductor lowerings."*
Never "fixed a bug in PyTorch" — `repro/check_upstream.py` collapses that in one command.

