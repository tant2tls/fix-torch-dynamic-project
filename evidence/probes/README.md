# `evidence/probes/` — the exploratory scripts

The raw investigation, in the order the questions came up. These are **working
probes**, not the polished harness: some are one-shot, some print without asserting,
some were answered and abandoned. Kept because this is where the findings actually
came from, and because a claim's first measurement is often more honest than its
final write-up.

For the **documented, portable versions** behind the shipped claims, see `../` and
`../README.md`. If a number appears in a PR body, cite the script one level up, not
one of these.

## "Is the fix correct?"

| script | the question it answered |
|---|---|
| `ulp_proof.py` | ⭐ **without a GPU**: does an approximate-reciprocal fp32 divide actually change the floored index? This is the argument `div_rn` rests on. |
| `codegen_check.py` | what does the generated kernel contain on the symbolic path? |
| `codegen_noop.py` | is the shipped patch byte-identical on paths that already compile? |
| `isolate.py`, `t2.py`, `t3.py` | narrowing scratchpads for the forward crash |
| `stress.py` | reviewer objections, tested rather than argued |
| `broad2.py` | blast radius of the `OpsWrapper.constant` override across many lowerings |
| `diag_check.py` | does the override take effect without breaking dispatch? |
| `hotpath.py` | which lowering is actually on the hot path |

## "Should it guard, or defer?"

| script | the question it answered |
|---|---|
| `perf.py` | does the runtime divide cost anything vs a folded constant? |
| `perf_why.py` | ⭐ **why was `div_rn` 8× slower?** Generated kernels side by side. The answer was the **dynamic shape**, not the rounding mode — swapping in plain `truediv` measured the same. That reframed the whole PR1 argument. |
| `hybrid_check.py` | would a hybrid (fold when already concrete) be better? |

## The backward bug (Issue D), found by auditing sibling lowerings

| script | the question it answered |
|---|---|
| `probe_backward.py` | is `upsample_nearest2d_backward` reachable with a symbolic `input_size`, and does it break? |
| `probe_bw2.py` … `probe_bw5.py` | successive narrowing of that failure |
| `confirm_root.py` | confirm `ceildiv(inp, out)` goes symbolic when `out_h`/`out_w` are unguarded |
| `bw_fix_test.py` | does guarding fix it — **and is guarding the right call here?** Yes: the quotient is a `range()` bound, so there is nothing to defer. |
| `bw_reach.py` | can ATen/autograd reach it? No — 0 hits across 8 configs. |
| `bw_noreg.py` | is real autograd through `interpolate` unchanged? 20/20 bit-identical. |

## Issue repros and staged tests

| script | what it is |
|---|---|
| `issue_repro.py`, `issue3_repro.py` | minimal repros as pasted into the issue drafts |
| `stage_pr{1,2,3}_test.py` | test bodies staged per PR before being merged into one file. Shipped version: `../test_custom_lowering.py`. |
| `bw.patch`, `diag.patch`, `divrn.patch` | intermediate patch forms. ⚠️ **Superseded** by `../../upstream/patches/`, which is verified byte-identical to the git branch — do not apply these. |

## Caveat

These have **not** all been re-run on current hardware, and a few carry assumptions
from the session that produced them. Treat their output as historical unless you
re-run it; every claim that survived is reproduced by the scripts in `../`.
