# PR #2 body — paste into GitHub

**Branch:** `inductor-ops-constant-contract` (stacked on PR #1)
**Patch:** `0002-inductor-Reject-symbolic-values-in-ops.constant.patch`
**Files:** `torch/_inductor/virtualized.py` (+23), `test/inductor/test_custom_lowering.py` (+24)

**Do not open this at the same time as PR #1.** Raise the idea on the issue thread first,
after PR #1 lands. It is a separate concern (a codebase-wide invariant, not one lowering),
and bundling it risks stalling the uncontroversial fix. If a maintainer says no, drop it —
the evidence stands either way.

---

Follow-up to #NNNNN (PR #MMMMM). Depends on that PR's test additions.

### Motivation

`ops.constant` takes a concrete leaf value. A lowering that derives one from sizes it never
made concrete can pass a symbolic expression instead, which no backend can codegen. Nothing
checks this at the call, so the failure surfaces well downstream:

```
NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
```

raised by `fx.proxy` while tracing, about a dozen frames below the lowering that actually
broke the contract, naming neither the op nor what to do instead. That is how the bug in
#NNNNN presented, and locating it meant reading up the stack past the frame that raised.

This is not a new rule — several lowerings already dispatch on it by hand:

```python
# Use index_expr for sympy expressions (e.g. from .item()), constant otherwise
if isinstance(value, sympy.Basic):
    value_expr = ops.index_expr(value, dtype)
else:
    value_expr = ops.constant(value, dtype)
```

That appears in `_full` and in both the `addcmul` and `addcdiv` helpers, and `alpha`
handling in `add`/`sub` does the same. Four sites spell out the same invariant with nothing
enforcing it; `upsample_nearestnd` is the one that omitted it. This enforces it in one
place and names both alternatives in the message, so the error is actionable where raised.

### Implementation notes

- Added as an explicit `OpsWrapper.constant` rather than a check inside `_default`, so no
  other op pays for it.
- It **unwraps first, then delegates to `_default`**, so `constant` keeps exactly the
  wrapping/unwrapping every other op gets. Both orderings matter: checking before `_unwrap`
  would reject an `OpsValue` wrapping a concrete number, and returning
  `OpsWrapper._wrap(_ops.constant(...))` directly would make `constant` the only op that does
  not unwrap its argument. No current caller passes a wrapped value — all 112 `ops.constant`
  call sites pass plain scalars — so this is about not leaving a divergence behind.
- sympy numbers (`sympy.Integer`, `sympy.Float`, `sympy.Rational`) are `sympy.Expr` but
  concrete, so the `is_number` clause keeps them accepted.
  `test_ops_constant_accepts_concrete_values` compiles `full_like` / `add(alpha=)` /
  `addcmul` to cover the existing hand-guarded paths.
- I checked 25 lowerings (pooling, pad, `masked_fill`, softmax, layernorm, `where`, scalar
  div, pow, cumsum, nearest and bilinear upsample, hardtanh, logsumexp, norm, tril,
  addcmul, addcdiv, full_like, floor_divide, add with alpha, …) static and with
  `dynamic=True`: no behaviour change, 25/25 either way.

### Cost

One `isinstance` plus one `_unwrap` per call. A constant-heavy graph makes **160
`ops.constant` calls per compile**, so about **60 µs total — 0.002% of that compile**.
Generated code is **byte-identical across 96 configurations** (24 constant-using ops ×
CPU/CUDA × static/dynamic), so this is a strict no-op on output.

### Testing

```bash
python test/inductor/test_custom_lowering.py -k ops_constant
```

`test_ops_constant_rejects_symbolic_value` fails without the change;
`test_ops_constant_accepts_concrete_values` passes either way and exists to catch a
false positive.

### Open question for reviewers

Whether the check should be unconditional, as here, or gated behind a debug config. I kept it
unconditional given the measured cost above, and because a silently-accepted violation just
fails later and worse — but I do not have a strong view. Happy to gate it if you prefer.

---

**Do not add reviewers** — leave the Reviewers section empty per `CONTRIBUTING.md`; triage
assigns. Label `module: inductor`.

---

### ⚠️ AI-assistance disclosure — required by `AI_POLICY.md`, edit to match the truth

Keep one accurate sentence; **do not submit without it.** See `SUBMIT.md` for the quoted rule.

> Disclosure: I used an AI assistant while investigating this and while drafting parts of
> this description. The call-site audit, the unwrap-then-delegate ordering, and the cost
> measurement are mine; I have read the change and can answer for every line of it.

Delete this section from the pasted body.

