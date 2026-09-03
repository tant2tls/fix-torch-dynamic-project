# PR 3 — `[inductor] Guard input_size in upsample_nearest2d_backward`

| | |
|---|---|
| **Commit** | `c87c2a8580` (branch `prseries`, on `b1716d913a`) |
| **Files** | `torch/_inductor/lowering.py` (+7), `test/inductor/test_custom_lowering.py` (+34) |
| **Blocked on** | Issue D marked **`actionable`** |
| **Send order** | **FIRST** — smallest, no design argument, establishes the account |
| **Verified on** | 1× A100-SXM4-80GB, torch 2.13.0+cu130; CPU on 2.13.0+cpu |

> **Send this one first.** 7 source lines, one test, one obvious fix, no new op, no
> performance question. Nothing here needs a design debate, which is exactly what you want
> from a first PR as an unknown contributor.

---

## 1. The defect

`upsample_nearest2d_backward` guards the grad tensor's spatial sizes but not `input_size`:

```python
inp_h = V.graph.sizevars.guard_int(inp_h)
inp_w = V.graph.sizevars.guard_int(inp_w)
*_batch, out_h, out_w = input_size          # NOT guarded
```

`out_h`/`out_w` are then used as divisors — `inp_h % out_h` is evaluated eagerly, and
`ceildiv(inp_h, out_h)` becomes a `sympy.FloorDiv` — and that quotient becomes a `range()`
bound inside `_adaptive_pooling_fn`, so lowering aborts:

```
TypeError: 'FloorDiv' object cannot be interpreted as an integer
```

Fix: guard them the same way the two lines above already do.

## 2. Why this guards where PR 1 refuses to — answer this pre-emptively

A reviewer reading both PRs will notice PR 1 argues *against* `guard_int` and PR 3 adds one.
That looks inconsistent unless you give the structural reason:

| | forward (PR 1) | backward (PR 3) |
|---|---|---|
| what the quotient becomes | a scalar multiplier | a `range()` bound |
| where it lives | kernel **argument** | kernel **body** |
| can it be deferred? | yes — one kernel serves every size | no — the window is unrolled at lowering time |
| measured cost of 7 sizes | 1 lowering invocation | necessarily 1 per distinct window |

A 2×2 window is 4 unrolled loads; an 8×8 window is 64. Those are *different kernels*, not
one kernel with a different argument. So specializing here is not a tradeoff against a
cheaper alternative — it is the only way to lower this at all. Put that sentence in the PR.

## 3. Reachability — and get the mechanism right

**Not reachable from autograd today, but not for the reason one would guess.** The backward
op is **not** in `torch._decomp.decomposition_table`. What actually happens is that the
*forward* is decomposed to `arange/mul/_unsafe_index` before autograd runs, so the backward
graph contains `_unsafe_index_put` and `aten.upsample_nearest2d_backward` is never emitted:

```
fwd graph ops: ['_to_copy', '_unsafe_index', 'add', 'arange', 'mul', 'sum', 'unsqueeze']
bwd graph ops: ['_unsafe_index_put', 'expand', 'new_zeros']
```

Measured **0 hits** on this lowering across 8 real autograd configurations (nearest /
nearest-exact × `size=` / `scale_factor=` × static / dynamic). The test therefore calls the
aten op directly.

Do **not** write "the backward is decomposed before Inductor sees it" — it isn't, and a
reviewer who checks `decomposition_table` will find that out.

## 4. Evidence

- **1 new test**, CPU and GPU. Fails before (TypeError, 4/4 shape pairs), passes after,
  bit-exact with `atol=rtol=0`. Covers both branches: the exactly-divisible path that routes
  to `avg_pool2d` and the general adaptive-pooling path. **Fails with only this fix
  reverted.**
- **Real autograd is unaffected:** 20/20 gradient comparisons through `F.interpolate` remain
  bit-identical (nearest / nearest-exact, `size=` / `scale_factor=`, static / dynamic).
  Re-measured on this A100, not inherited.
- Full dynamic-shapes suite: 2621 tests, 1 pre-existing unrelated failure.

## 5. The tolerance caveat — disclose it, don't let a reviewer find it

`upsample_nearest2d_backward` is **not** bit-exact against eager for every ratio. grad
64×64 → input 8×8 differs on 155/192 elements, max|d| 3.8e-06. Since the test asserts
`atol=rtol=0`, this has to be addressed or it looks like the shape pairs were cherry-picked
without knowing why.

Triaged before trusting it:

- **Identical with a static `input_size`** → independent of this change.
- **float64 drops it to max|d| ~7e-15** (from ~1e-5), scaling with machine epsilon →
  summation-order noise, not a wrong index. An index error would be dtype-independent.
- **The boundary follows the branch, not the window size:** 64→9 is an 8×8 window and is
  bit-exact; 64→8 is an 8×8 window and is not. It tracks the exactly-divisible `avg_pool2d`
  reassociation.

All four pairs in the test sit inside the bit-exact regime, chosen deliberately. So
`atol=rtol=0` is meaningful — but say "these pairs are bit-exact," never "this op is
bit-exact."

## 6. Pre-push

- [ ] Issue D filed and labelled **`actionable`**; CLA signed
- [ ] Rebased onto current `origin/main`; test re-run after the rebase
      — ⚠️ **expect a conflict on the exact line this patch edits.** As of 2026-08-28
      `main` carries a `# pyrefly: ignore [not-iterable]` comment immediately above the
      `*_batch, out_h, out_w = input_size` unpack that is **not** in the branch base
      (`b1716d913a`). Keep the comment; apply the guard below it. Verified by reading
      `main`'s `lowering.py` directly — see `../evidence/RESULTS_a100.md` §17.3.
- [ ] `lintrunner -a` clean
- [ ] `python ../tools/state.py set pr3=off` → the backward test FAILS; back to `on` → passes
- [ ] Reviewers list **empty**
- [ ] PR body states the reachability mechanism correctly (forward decomposition, not
      backward) and includes the tolerance caveat
