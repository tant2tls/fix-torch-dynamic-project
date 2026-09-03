# PR 2 — `[inductor] Reject symbolic values in ops.constant`

| | |
|---|---|
| **Commit** | `3d9d172c6f` (branch `prseries`, on `b1716d913a`) |
| **Files** | `torch/_inductor/virtualized.py` (+23), `test/inductor/test_custom_lowering.py` (+24) |
| **Blocked on** | Issue C marked **`actionable`** |
| **Send order** | **last** — this is the one that will attract debate |
| **Verified on** | 1× A100-SXM4-80GB, torch 2.13.0+cu130, triton 3.7.1; CPU on 2.13.0+cpu |

> **Read this before writing the PR body.** This is not a bug fix; it is a contract change
> on a surface of **111 call sites**. It will be judged on whether the contract already
> exists and whether the check costs anything. Both are measured below. Lead with the
> six-hand-guards argument, not with the error message.

---

## 1. The defect

`ops.constant` takes a concrete leaf value. Nothing enforces that at the call, so a
lowering that derives one from sizes it never made concrete gets no error at the mistake.
The value flows on and dies while tracing:

```
NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
```

raised from `torch/fx/proxy.py`, about a dozen frames below the lowering that broke the
contract, naming neither the op nor the fix. (On torch 2.3.1 the same mistake surfaced in
`IndexPropagation` as `TypeError: Cannot convert expression to float` from
`sympy/core/expr.py` — equally uninformative.) Debugging it means reading the whole
lowering to find which size was never guarded.

## 2. The actual argument: six hand-guards, two disagreeing predicates

An earlier draft said "several lowerings already dispatch on this by hand" and cited
`_full`. That understates the case and cites the *weaker* example. The real finding:

**Six** sites in `lowering.py` enforce this contract by hand, using two predicates that do
not agree:

| predicate | sites |
|---|---|
| `isinstance(value, sympy.Basic)` | `_add_with_alpha_fma` (687), `tensor` (4261), `_full` (4409), `addcmul` (8577), `addcdiv` (8648) |
| `isinstance(x, sympy.Expr) and not x.is_number` | `get_constant_or_index_expr` in `var_mean_welford_` (7576) |

They diverge on exactly the sympy-but-concrete values:

| value | `sympy.Basic` | `Expr and not is_number` |
|---|---|---|
| `sympy.Integer(5)` | True | False |
| `sympy.Float(2.5)` | True | False |
| `sympy.Rational(1,2)` | True | False |
| `32*s0` (symbolic) | True | True |

So the five `sympy.Basic` sites route a concrete `sympy.Integer` to `ops.index_expr` while
`var_mean`'s routes it to `ops.constant`. Both work today — `index_expr` accepts concrete
values too — but only one expresses the contract, and **a new call site has a five-in-six
chance of copying the one that does not.**

This PR adopts `var_mean`'s predicate (the narrower, correct one) and enforces it once.
That reframes the PR from *"add a check"* to *"codify the correct one of two predicates
already in tree"* — a much easier review.

## 3. `is_number` is load-bearing, not defensive

The obvious simplification a reviewer will propose is dropping `is_number` and testing
`isinstance` alone. Measured refutation — instrumenting `OpsWrapper.constant` over a
24-case compile workload (pointwise, reductions, norms, softmax, pooling, upsample,
cat/cumsum/cast, matmul, static and dynamic):

| population | count | share |
|---|---|---|
| total `ops.constant` calls | 1221 | 100% |
| plain (non-sympy) values | 1208 | 98.94% |
| sympy, **concrete** | 13 | 1.06% |
| sympy, **symbolic** | 0 | 0% |

All 13 sympy values are `sympy.Integer`, all from `get_constant_or_index_expr` in
`var_mean_welford_`, reached via `layer_norm` (7) and `mean`/`var` (6).
**An `isinstance`-only check would have rejected all 13.**

> ⚠️ **This probe must run on a COLD inductor cache, and it now forces one.**
> `ops.constant` is only called while a graph is *lowered*; with the cache warm,
> compilation is served from cache and the instrumentation records **0 calls / 0
> sites** — which reads as "the check is never reached", the opposite of this PR's
> justification. `TORCHINDUCTOR_CACHE_DIR` defaults into NFS `$HOME` here, shared
> fleet-wide and warm from earlier sessions. The totals moved 1157 → 1221 between
> torch builds; the decisive **13 concrete / 0 symbolic** is stable.

## 4. Cost: compile-time only, ≈4% of lowering

Re-measured 2026-08-28 on a cold cache with the two variants interleaved (three
alternations):

| | whole 24-case workload |
|---|---|
| with the check (this PR) | **5.25–5.29 s** |
| without (pre-PR behaviour) | **5.04–5.13 s** |
| ratio | **1.039×** (min 1.033, max 1.041) |

⚠️ **The old "1.23 s vs 1.25 s = 0.98×" was noise, not a measurement.** Three
identical runs of the original script gave 0.9864 / 1.0431 / 1.1246 — straddling
1.0, i.e. below that harness's resolution; 0.98× was one draw from that spread. On a
cold cache with interleaving the result becomes reproducible and the interval
**excludes 1.0**, so the honest number is a consistent **≈3.9% of lowering time**,
not "below noise."

`ops.constant` runs once per IR node during lowering, not once per element at
runtime, and the check never appears in the generated kernel. **Lead with that** —
"≈4% of *compile* time, zero runtime cost" is a defensible claim, whereas "0.98×"
invites "that's benchmark noise, so you didn't measure it."

## 5. Design choices to defend explicitly

**Why an explicit `OpsWrapper.constant` and not a branch in `_default`?** So no other op
pays for the `isinstance`. `_default` is the hot path for every op; `constant` is one op.

**Why `TypeError` and not an assert?** The message names the op, prints the offending
expression *and its free symbols*, and states both ways out — make the inputs concrete with
`V.graph.sizevars.guard_int` (which specializes on them), or keep them dynamic and use
`ops.index_expr` (which accepts symbolic expressions). An assert would be stripped under
`-O` and would not carry the remedy.

**Why not convert the six existing sites in this PR?** They are correct as written and each
is a separate behavioural question. This only makes the contract *checkable*, so the next
violation reports itself at the call. Say this — otherwise a reviewer assumes the cleanup
was forgotten.

**Note the interaction with PR 1.** PR 2 enforces the contract that PR 1 is currently the
only in-tree violator of. They are independent — either can land alone — but if PR 2 lands
first, PR 1's crash changes message and still crashes. Mention it in both PR bodies so
nobody is surprised.

## 6. Evidence

- **2 new tests:** a symbolic value raises with a message naming both alternatives;
  `sympy.Integer`/`sympy.Float` keep working through `full_like`, `add(alpha=)`, and
  `addcmul`. `test_ops_constant_rejects_symbolic_value` **fails with only this fix
  reverted**.
- **Full dynamic-shapes suite, patches ON: `Ran 2621 tests`, 1 failure — pre-existing and
  unrelated.** That failure is `test_unbacked_reduction_cpu`, an inverted-xfail
  (`AssertionError: expected to fail, but actually passed`); it fails identically with all
  three patches reverted. **This is the single most important number in this PR** — a
  contract change over 111 call sites with zero attributable failures on the suite that
  most exercises `ops.constant` under dynamic shapes. Quote it.
- Targeted suites, identical in both states: `test_indexing` 56 OK,
  `test_optimize_indexing` 15 OK, `test_codegen_triton` 16 OK (2 skip).

> ⚠️ Only the **ON** arm of the 2621-test suite was run to completion. The OFF arm was
> verified for the failing test specifically, not for all 2621. Word it that way.

## 7. Pre-push

- [ ] Issue C filed and labelled **`actionable`**; CLA signed
- [ ] Rebased onto current `origin/main`; **re-run `pr2_blast_radius.py`** — the six
      hand-guard line numbers and the 13/1157 counts are `main`-dependent and will drift
- [ ] `lintrunner -a` clean
- [ ] `python ../tools/state.py set pr2=off` → `test_ops_constant_rejects_symbolic_value` FAILS;
      back to `on` → passes
- [ ] Reviewers list **empty**
- [ ] PR body leads with the six-hand-guards/two-predicates table, then the 2621-test result
