# `notebook/` — the working record

What was tried, what was superseded, and why. Nothing here is load-bearing for the
current claims; it exists so that a decision can be re-examined instead of re-argued.

Kept **tracked and visible** in develop phase. The public cut is a later, separate
step (`PUBLISH.md`).

| path | what it is |
|---|---|
| `superseded/ISSUE.md`, `superseded/ISSUE3.md` | the earlier **two-issue** plan. Superseded by `../upstream/issues.md`, which has four (A: wrong pixels, B: forward crash, C: `ops.constant` contract, D: backward crash) plus E held as a duplicate of upstream #159550. |
| `superseded/ISSUE_early_draft.md`, `superseded/ISSUE3_early_draft.md` | ⚠️ **byte-identical duplicates** of the two files above (verified by `md5sum`) — the "early draft" names promise an earlier revision that does not exist. Nothing to diff. Candidates for deletion at the public cut; kept for now only because develop phase does not delete. |
| `superseded/upstream_README_old.md` | the earlier send order, written when the plan was two issues and PR2 was "propose on the thread after #1 lands." Current order is **PR3 → PR1 → PR2**, all three as real PRs. |
| `superseded/PR1_BODY_old.md`, `superseded/PR2_BODY_old.md` | ⚠️ **kept as a cautionary example.** These carried line counts (`+29/−3`, `+170/−1`, `+26`) that did not match the branch, and PR1's body described the **abandoned type-punned `inv_scales`** design plus a reachability claim its own submit doc explicitly forbade. All corrected against `git diff --numstat` and the shipped source. If you ever wonder why `../upstream/README.md` insists on deriving numbers from git rather than copying them between documents, this is why. |

## The three corrections worth remembering

1. **`fallback()` does not fix the 2.3.1 bug** — it raises `NotImplementedError`
   because the value is still symbolic when the constant is constructed.
   `index_expr` is required. (The original write-up in `../origin/` says otherwise.)
2. **The `guard_int` approach was abandoned.** Guarding the output size specializes
   per size and, past Dynamo's recompile limit, drops the caller to **eager** — worse
   than the crash for a serving workload. The shipped fix defers the divide via
   `ops.div_rn` instead.
3. **A snapshot directory drifted from the branch and nearly shipped a reverted bug.**
   The first consolidation copied a `final/` snapshot still containing the
   type-punned design (silent wrong numerics, out-of-range reads) and a test with an
   FX-graph-cache collision that produced a false pass. `../upstream/final/` is now
   verified byte-identical to the branch, and the patches are verified to reproduce
   it exactly.
