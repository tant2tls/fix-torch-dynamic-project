# `upstream/` — everything needed to submit

## Read in this order

1. **`issues.md`** — the four issue bodies, the prior-art search, and the filing
   order. **File issues first**; PyTorch will not review a new contributor's PR
   without a linked issue labelled **`actionable`**.
2. **`SUBMIT.md`** — shared context: the process gate, the three-commit stack, and
   the cross-cutting evidence to quote in every PR.
3. **`SUBMIT_PR3.md` → `SUBMIT_PR1.md` → `SUBMIT_PR2.md`** — in send order. Each
   carries that PR's argument, the objections a reviewer will raise, and the limits
   of what it may claim.
4. **`PR{1,2,3}_BODY.md`** — paste into GitHub once the issue is `actionable`.
   Replace `#NNNNN` with the issue number.

## Send order: PR3 → PR1 → PR2

| PR | change | why this position |
|---|---|---|
| **PR3** | `lowering.py` **+7** — guard `input_size` in the backward | 7 lines, one test, no design argument. Establishes the account. |
| **PR1** | `lowering.py` **+30 −4** — defer the divide in the forward | needs the `div_rn` and recompile arguments, both measured |
| **PR2** | `virtualized.py` **+23** — reject symbolic values in `ops.constant` | changes a shared contract across 111 call sites; most debate |

**Three separate PRs, not one** — my own call, not a quoted rule. ("One concern per
PR" is *not* in `CONTRIBUTING.md` or the wiki; verified 2026-08-30 by grepping both.
What the docs *do* say is that overly verbose or low-quality contributions stop being
accepted, and `AI_POLICY.md` asks you to "simplify where possible and make sure the
core change is easy for maintainers to review" — which points the same way.) Bundled,
one objection stalls all three. Mention in both PR1 and PR2 that they interact: PR2
enforces the contract PR1 is currently the only in-tree violator of. They are
independent (either can land alone), but a reviewer should not be surprised.

## Contents

| path | what it is |
|---|---|
| `patches/000{1,2,3}-*.patch` | `git am`-ready, **regenerated from the branch and verified byte-identical to it**. A stacked series: apply in order (0002 and 0003 do not apply to pristine `main` alone — PR1's test-file `import sympy` comes first) |
| `commit_msgs/pr{1,2,3}.txt` | the full commit messages, with the measurements and the disclosures inline |
| `final/` | `lowering.py`, `virtualized.py`, `test_custom_lowering.py` as shipped, for direct diffing |

Base: `main` @ **`b1716d913a`**. `+299 / −5` across 3 files (code `+60 −4`, tests
`+239 −1`).

## Non-negotiables before pushing

- Sign the **CLA**.
- **Rebase** — upstream HEAD moved twice in a single session previously.
- Run **`lintrunner -a`** (never yet run here — no PyTorch build on this box).
- Re-run the **guard matrix** after the rebase.
- **Leave the Reviewers list empty.** The triage squad assigns; do not @-mention.
- **No AI-generated fix explanation in an issue body** — `CONTRIBUTING.md` forbids it.
- **Disclose that PRs 1 and 3 are not ATen-reachable today.** The decompositions
  fire first. Overclaiming reachability is the fastest way to lose the review.

## Provenance of `final/` and `patches/` — verified, not assumed (2026-08-28)

Both were checked against branch `prseries` @ `c87c2a8580` rather than trusted:

- `patches/000{1,2,3}` are **byte-identical** to `git format-patch` output from the
  branch, and applying all three to pristine `b1716d913a` yields a tree with an
  **empty diff** vs the branch.
- `final/{lowering,virtualized,test_custom_lowering}.py` are byte-identical to the
  branch versions, and are exactly what the three patches produce.
- `commit_msgs/pr{1,2,3}.txt` match the branch commit messages (modulo a trailing
  blank line).
- `tools/state.py`'s ON-forms are present verbatim in `final/`, so the toggle tool
  and the shipped code cannot drift.

⚠️ **Why this check exists.** The first assembly of this folder copied `final/` from
a **stale snapshot** that still contained the abandoned type-punned `inv_scales`
design — `o if isinstance(o, sympy.Expr)...` with `scale_fn` testing
`isinstance(scale, sympy.Expr)` **without `is_number`**. That is precisely the
wrong-numerics blocker (0/8 bit-exact, out-of-range reads, interpreter segfaults) that
was found and fixed in review. It also carried the FX-graph-cache test bug (a single
`"ups2d"` op name reused across loop iterations) that produces a **false pass**.

The patches were correct throughout; only the `final/` snapshot was stale. Verify
`final/` against the branch after any change, and never treat it as the source of
truth — the branch is.

### Line counts, from `git diff --numstat` (authoritative)

| PR | code | test |
|---|---|---|
| PR1 | `lowering.py` +30 −4 | +181 −1 |
| PR2 | `virtualized.py` +23 | +24 |
| PR3 | `lowering.py` +7 | +34 |
| **total** | **+60 −4** | **+239 −1** → `+299 / −5` |

Several documents were corrected on 2026-08-28 to match these; earlier drafts said
`+29/−3`, `+170/−1`, `+26`, and `+298/−5`. If you regenerate the patches, re-derive
these numbers rather than copying them.
