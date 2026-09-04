# Next-session handoff

Use this page to resume the PyTorch contribution work without reconstructing the
investigation from filenames. Link and label state below was last checked on
**2026-09-04** and must be refreshed immediately before filing.

## First five minutes

```bash
cd /root/fix-torch-dynamic-project
conda run -n dynamic python tools/state.py status
conda run -n dynamic python repro/check_upstream.py
```

Expected local wheel state after the Blackwell verification is PR1=ON, PR2=ON,
PR3=ON. Run `evidence/verify_issue_repros.py` only with all three patches OFF, using
`tools/state.py` as the only patch toggle.

Read these local files in order:

1. [`../PR.md`](../PR.md#7-exact-next-steps-for-contributing-to-pytorch) — current
   gates and the three contribution phases.
2. [`issues.md`](issues.md) — evidence-backed working notes, not paste-ready issue
   text.
3. [`SUBMIT_PR3.md`](SUBMIT_PR3.md), [`SUBMIT_PR1.md`](SUBMIT_PR1.md), then
   [`SUBMIT_PR2.md`](SUBMIT_PR2.md) — the planned submission order.
4. [`../evidence/logs/blackwell_20260904.md`](../evidence/logs/blackwell_20260904.md)
   — latest environment and Blackwell command transcript.
5. [`../evidence/README.md`](../evidence/README.md) — which harness supports each
   claim.

## Existing issues and prior art

| Link | Last checked state | Why it matters |
|---|---|---|
| [#97135 — incorrect nearest-upsample gradient on CUDA](https://github.com/pytorch/pytorch/issues/97135) | Open; `needs reproduction` | Add the human-authored eager forward/backward invariant reproduction here. Do not open a duplicate backward issue. |
| [#175154 — compiled nearest interpolation mismatch](https://github.com/pytorch/pytorch/issues/175154) | Open; not `actionable` | Related but distinct: its `scale_factor=`/CPU/static behavior does not cover Issue A. |
| [#185806 — float reciprocal corrupting dynamic dimensions](https://github.com/pytorch/pytorch/issues/185806) | Closed | Strong defect-class precedent for Issue A. |
| [#159550 — dynamic adaptive average pooling](https://github.com/pytorch/pytorch/issues/159550) | Closed | Issue E is a duplicate; do not file it. |
| [PR #164144 — eager-exact division](https://github.com/pytorch/pytorch/pull/164144) | Closed/reverted | Explains why any `div_rn` argument must include measured cost and scope. |
| [PR #165566 — division-rounding flag](https://github.com/pytorch/pytorch/pull/165566) | Closed | Introduced the flag that still does not cover `IntTrueDiv`. |
| [PR #95698 — floating symbolic upsample division](https://github.com/pytorch/pytorch/pull/95698) | Closed | Direct precedent for preserving fractional symbolic upsample scales. |
| [PR #193959 — reject symbolic fill](https://github.com/pytorch/pytorch/pull/193959) | Closed | Precedent for preventing symbolic values from reaching an incompatible lowering path. |

Fresh issue searches:

- [Nearest upsample + dynamic shapes](https://github.com/pytorch/pytorch/issues?q=repo%3Apytorch%2Fpytorch+is%3Aissue+%22upsample+nearest%22+%22dynamic+shapes%22+in%3Atitle)
- [Nearest interpolation + wrong dynamic result](https://github.com/pytorch/pytorch/issues?q=repo%3Apytorch%2Fpytorch+is%3Aissue+%22interpolate+nearest%22+wrong+dynamic+in%3Atitle)
- [`upsample_nearestnd`](https://github.com/pytorch/pytorch/issues?q=repo%3Apytorch%2Fpytorch+upsample_nearestnd)
- [`ops.constant` + symbolic](https://github.com/pytorch/pytorch/issues?q=repo%3Apytorch%2Fpytorch+is%3Aissue+%22ops.constant%22+symbolic)
- [`upsample_nearest2d_backward` + Inductor](https://github.com/pytorch/pytorch/issues?q=repo%3Apytorch%2Fpytorch+upsample_nearest2d_backward+inductor)

Use the GitHub API to verify labels instead of relying on page summaries:

```bash
for n in 97135 175154 185806 159550; do
  curl -fsSL -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/pytorch/pytorch/issues/$n" |
    jq '{number,state,title,labels:[.labels[].name],updated_at,html_url}'
done
```

## Current PyTorch source and contribution rules

- [PyTorch `main`](https://github.com/pytorch/pytorch)
- [`upsample_nearestnd` and backward lowering](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/lowering.py)
- [`OpsWrapper`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/virtualized.py)
- [Native upsample semantics](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/UpSample.h)
- [CUDA nearest implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UpSampleNearest2d.cu)
- [Contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
- [AI policy](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md)
- [Issue and PR workflow](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)
- [Finding or reporting issues](https://github.com/pytorch/pytorch/wiki/Finding-Or-Reporting-Issues)
- [Contributor License Agreement](https://code.facebook.com/cla)
- [New issue chooser](https://github.com/pytorch/pytorch/issues/new/choose)

The last read-only source inspection used PyTorch `main` commit
[`4f5a382575c6`](https://github.com/pytorch/pytorch/commit/4f5a382575c6e21b1a6541b8ecd697734a2469cd).
It confirmed the three source conditions were still present, but it was not a build or
test of current `main`.

## Local artifacts by proposed contribution

| Work item | Issue notes | PR reasoning | Local PR draft | Patch reference |
|---|---|---|---|---|
| Issue A — CUDA dynamic wrong pixels | [`issues.md`](issues.md#issue-a) | No PR yet; coordinate with #97135 first | None | Forward prototypes are evidence only |
| Issue B / PR1 — symbolic forward output size | [`issues.md`](issues.md#issue-b) | [`SUBMIT_PR1.md`](SUBMIT_PR1.md) | [`PR1_BODY.md`](PR1_BODY.md) | [`patches/0001-…patch`](patches/0001-inductor-Fix-symbolic-output_size-in-upsample_neares.patch) |
| Issue C / PR2 — `ops.constant` contract | [`issues.md`](issues.md#issue-c) | [`SUBMIT_PR2.md`](SUBMIT_PR2.md) | [`PR2_BODY.md`](PR2_BODY.md) | [`patches/0002-…patch`](patches/0002-inductor-Reject-symbolic-values-in-ops.constant.patch) |
| Issue D / PR3 — symbolic backward input size | [`issues.md`](issues.md#issue-d) | [`SUBMIT_PR3.md`](SUBMIT_PR3.md) | [`PR3_BODY.md`](PR3_BODY.md) | [`patches/0003-…patch`](patches/0003-inductor-Guard-input_size-in-upsample_nearest2d_back.patch) |

The issue notes and PR bodies contain AI-assisted drafting. They are local evidence and
review material, not text to submit unchanged. The human author must understand every
claim, rewrite the issue reports in their own words, approve the exact external text,
and follow PyTorch's disclosure policy.
