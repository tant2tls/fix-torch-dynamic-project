# PR #3 body — paste into GitHub

**Branch:** `inductor-upsample-backward-symbolic-input-size`
**Patch:** `0003-inductor-Guard-input_size-in-upsample_nearest2d_back.patch`
**Files:** `torch/_inductor/lowering.py` (+7), `test/inductor/test_custom_lowering.py` (+34)

**This is a second, independent bug** found while investigating #NNNNN — the same
guarded-input / unguarded-output asymmetry in the *backward* lowering, with a different
symptom and a different correct fix. The code change does not depend on PR #1; only the test
file overlaps, so apply PR #1 first or expect a trivial context conflict.

**File it as its own issue** (same template, link it to #NNNNN as related) and wait for the
`actionable` label before opening the PR.

---

Fixes #MMMMM

### The problem

`upsample_nearest2d_backward` guards the grad tensor's spatial sizes but not `input_size`:

```python
inp_h = V.graph.sizevars.guard_int(inp_h)
inp_w = V.graph.sizevars.guard_int(inp_w)
*_batch, out_h, out_w = input_size          # not guarded
```

`out_h` / `out_w` are then used as divisors:

```python
if inp_h % out_h == 0 and inp_w % out_w == 0:   # evaluated eagerly, in Python
    ...
h_kernel_max = ceildiv(inp_h, out_h)            # -> sympy FloorDiv if symbolic
```

and `h_kernel_max` becomes a `range()` bound inside `_adaptive_pooling_fn`
(`itertools.product(range(kernel_maxes[0]), ...)`). With a symbolic `input_size`, lowering
fails:

```
TypeError: 'FloorDiv' object cannot be interpreted as an integer
```

Concretely, `input_size` arrives as `(s28, s28)` and `ceildiv(64, s28)` yields
`((s28 + 63)//s28)`.

### The fix

Guard `out_h` / `out_w` the same way the input sizes above are guarded.

Note this is the *opposite* choice from PR #1, deliberately: there the divisor feeds a value
that can be computed in the kernel, so deferring avoids a shape specialization. Here the
quotient is a Python loop bound over which the pooling window is unrolled at lowering time —
there is nothing to defer it to, so guarding is the only available fix. The comment in the
patch says so, to keep the two from looking inconsistent.

`ranges=list(input_size)` further down is left as-is on purpose. It still holds the original
(possibly symbolic) sizes, but the guards added here make those symbols equal to the values
used for the loop bounds, so the two cannot disagree within one compilation. Checked by
reusing a single compiled callable across six differing `input_size` values on CPU and CUDA:
each recompiles rather than reusing a kernel built for another size, and all are bit-exact.

### Reachability

Not reachable from autograd today: the backward is decomposed before Inductor sees it. Zero
lowering hits across `F.interpolate` and `nn.Upsample` backward, with `size=` and
`scale_factor=`, static and dynamic, and with the input's spatial dims marked dynamic. The
test therefore calls `aten.upsample_nearest2d_backward` directly.

So, as with PR #1: hardening a reachable-but-not-currently-reached lowering, not a
user-visible regression.

### Testing

```bash
python test/inductor/test_custom_lowering.py -k upsample_nearest2d_backward
```

`test_upsample_nearest2d_backward_symbolic_input_size` fails before this change
(`TypeError`, 4/4 shape pairs) and passes after (bit-exact, `atol=rtol=0`), covering both the
exactly-divisible path that routes to `avg_pool2d` and the general adaptive-pooling path.

Real autograd is unaffected: **20/20** gradient comparisons through `F.interpolate` remain
bit-identical (`nearest` / `nearest-exact` × `size=` / `scale_factor=` 2.0 / 1.5 /
non-divisible / downsample × static / dynamic), and the static direct backward op is
unchanged. Verified on an H100, torch 2.13 + CUDA 13.0.

---

**Do not add reviewers**; leave Reviewers empty per `CONTRIBUTING.md`. Label
`module: inductor`.

---

### ⚠️ AI-assistance disclosure — required by `AI_POLICY.md`, edit to match the truth

Keep one accurate sentence; **do not submit without it.** See `SUBMIT.md` for the quoted rule.

> Disclosure: I used an AI assistant while investigating this and while drafting parts of
> this description. The diagnosis, the reason this guards rather than defers, and the
> measurements are mine; I have read the change and can answer for every line of it.

Delete this section from the pasted body.

