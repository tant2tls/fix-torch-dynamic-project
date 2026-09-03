# PyTorch issues to file (then link the PRs)

**Why this file exists.** PyTorch gates review on an issue label:

> "Only PRs that address issues labeled `actionable` will be considered for review."
> "Maintainers will not review the PR if it is not associated with an issue with the `actionable` label."
> — [The Ultimate Guide to PyTorch Contributions](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)

So the order is: **file issue → wait for `actionable` → then open the PR linked to it.**
`CONTRIBUTING.md` adds: as a new contributor, "never send a PR that doesn't have a
corresponding issue with the 'actionable' label," and "you must wait for a maintainer
to review it and mark it actionable before preparing and sending a PR for it."


**Working notes, not paste-ready issue bodies.** The quoted sections below collect
measured facts and minimal repros. Before filing, I must rewrite each report concisely
in my own words, remove solution discussion, re-run its claims, and personally approve
the exact text.

Two more rules from `CONTRIBUTING.md` that shape what goes in these issues:

- **Do not paste AI-generated explanations of how to fix it into the issue.** "You
  should NEVER include AI-generated explanation of how to solve the problem." Keep the
  issue to *observed behaviour + minimal repro + expected vs actual*. The fix discussion
  belongs in the PR.
- **You are personally responsible for what you send.** Every number below was measured
  on the hardware named; re-run before filing.
- **Reviewers:** leaving the field empty is this repository's submission plan; it is
  not a quoted requirement from `CONTRIBUTING.md`.
- Sign the **CLA** before the PR.

**Environment for every claim below** (re-probe before filing; this container's GPU
changes between sessions):

```
torch 2.13.0+cu130   (release wheel, git cf30153c4c131c8164ee7798e5022d810682e2cb)
triton 3.7.1
NVIDIA A100-SXM4-80GB, driver 575.57.08
python 3.12.14
```

---

## Prior art check (done 2026-08-26; refreshed 2026-08-28; **re-verified from the API 2026-08-30**)

> ### ✅ 2026-08-30 refresh — labels read from `api.github.com`, not rendered pages
>
> This closes the caveat the 2026-08-28 pass had to leave open (the API was 403
> rate-limited that session). Every label below is from
> `GET /repos/pytorch/pytorch/issues/<n>`. **No finding is superseded and no patch of ours
> is duplicated.**
>
> | ref | state | labels (API, verbatim) | bearing on us |
> |---|---|---|---|
> | [#175154](https://github.com/pytorch/pytorch/issues/175154) | **open** | `high priority`, `triaged`, `module: dispatch`, `oncall: pt2`, `module: decompositions`, `PT2-Bug-Bash` | ⚠️ **NOT `actionable`** — confirmed again. There is no wait-free path to a PR on it. A *different* defect from our Issue A (proved by `evidence/same_bug_as_175154.py`): it is the `scale_factor=` branch, reproduces **statically and on CPU**, in fp64. |
> | [#185806](https://github.com/pytorch/pytorch/issues/185806) | closed | `high priority`, `module: crash`, `triaged`, **`module: correctness (silent)`**, `oncall: pt2`, `module: dynamic shapes`, `module: inductor`, `oncall: export`, `module: aotinductor`, `bot-triaged` | ⭐ **the precedent to lead Issue A with.** A float reciprocal used for integer dim recovery, silently corrupting dynamic batches — same defect class, accepted at high priority. |
> | [#164144](https://github.com/pytorch/pytorch/pull/164144) | closed | `Merged`, **`Reverted`**, `module: inductor`, … | the `truediv`→`div_rn` change: merged, reverted, then gated. Why Issue A must stand on correctness alone. |
> | [#165566](https://github.com/pytorch/pytorch/pull/165566) | closed | `Merged` | the gating PR (`TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING`). |
> | [#193959](https://github.com/pytorch/pytorch/pull/193959) | closed | `Merged`, `topic: bug fixes`, `release notes: inductor` | precedent that "a lowering mishandling a symbolic value" is accepted. |
> | [#184848](https://github.com/pytorch/pytorch/pull/184848) | closed | `module: inductor`, `module: dynamo`, **`agentic`** | authored by **`jansel`** — *withdrawn* in a bulk cleanup of agentic PRs, **not rejected on the merits**. |
> | [#95698](https://github.com/pytorch/pytorch/pull/95698) | closed | `Merged`, `open source` | ⭐ cite in **PR1**: made the CPP printer emit `(1.0/45.0)*ks2` because symbolic upsample scales collapsed under C++ integer semantics. Upstream precedent, our exact op. |
> | [#159550](https://github.com/pytorch/pytorch/issues/159550) | closed | `triaged`, `oncall: pt2`, `module: dynamic shapes`, `internal ramp-up task` | our Issue E is a **duplicate — do not file it.** |
> | [#117538](https://github.com/pytorch/pytorch/pull/117538) | closed | `Merged`, `module: inductor` | the PR that introduced `inv_scales`. Useful context in PR1. |
>
> **Fresh duplicate searches, 2026-08-30** (`repo:pytorch/pytorch`, via the search API):
>
> | query | hits | verdict |
> |---|---|---|
> | `upsample nearest dynamic shapes in:title` | 0 | — |
> | `interpolate nearest wrong dynamic in:title` | 0 | — |
> | `upsample_nearestnd` | 2 | both old merged PRs (#117538, #105317) — no issue |
> | `ops.constant symbolic` | 43 | none about upsample or the `ops.constant` contract |
> | `upsample_nearest2d_backward inductor` | 1 | a refactor PR (#179466), unrelated |
>
> **Still true, re-read from source against `main`:** `upsample_nearestnd` guards
> `i_sizes` but not `o_sizes`; `upsample_nearest2d_backward` leaves `input_size`
> unguarded. **PR1 and PR3 are still live.** Expect a rebase conflict on the
> `# pyrefly: ignore [not-iterable]` comment now sitting on PR3's `input_size` unpack
> line — keep the comment.
>
> ⚠️ Prior art ages. Re-run these before filing; the searches are cheap.

Searched `repo:pytorch/pytorch` for each symptom. No existing issue covers any of the
four findings:

| Query | Hits | Relevant? |
|---|---|---|
| `upsample_nearest2d inductor dynamic in:title` | 0 | — |
| `interpolate nearest dynamic shapes wrong in:title` | 0 | — |
| `ops.constant symbolic in:title` | 0 | — |
| `upsample_nearest2d_backward FloorDiv` | 1 | no (a refactor PR) |
| `Cannot convert expression to float inductor` | 46 | none about upsample |

**One strongly related precedent, and it must be cited in Issue A:**
[**#164144 "Fix truediv numerics between eager and compile"**](https://github.com/pytorch/pytorch/pull/164144)
made inductor's `truediv` emit Triton `div_rn` for eager parity. It **merged, was
reverted several times, and was then gated** behind `TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING`
by [#165566](https://github.com/pytorch/pytorch/pull/165566) after a throughput
regression on B200 ([#164301](https://github.com/pytorch/pytorch/issues/164301):
~6511 → ~4692 GB/s). That history is why Issue A leads with *wrong pixels*, not with
*"use div_rn"* — the mechanism is the same one that already proved contentious on
performance grounds, so the report has to stand on the correctness bug alone and the
PR has to show its divide is loop-invariant, not per-element.

Also relevant as precedent that this class of bug is accepted:
[#193959](https://github.com/pytorch/pytorch/issues/193959) `[inductor] Don't match
pointless_cumsum_replacement on a symbolic fill` — same shape of defect (a lowering
mishandling a symbolic value).

### The Stack Overflow / other-project links are NOT usable as evidence

Checked, and they are a different problem despite the identical error text:

- [SO 44263889 "Can't convert expression to float"](https://stackoverflow.com/questions/44263889/cant-convert-expression-to-float) — plain sympy misuse (`math.sqrt` on a `Symbol`), no PyTorch.
- [SO 60787563 "TypeError: can't convert expression to float, symbolic x polynomial interpolation"](https://stackoverflow.com/questions/60787563/typeerror-cant-convert-expression-to-float-with-symbolic-x-polynomial-interp) — sympy polynomial interpolation, no PyTorch.
- [coremltools#885](https://github.com/apple/coremltools/issues/885) — CoreML conversion, not TorchInductor.

`TypeError: Cannot convert expression to float` is generic sympy: it is raised by
`sympy/core/expr.py::__float__` whenever `float()` is called on a non-numeric
expression. **Do not cite these three in the issues** — they would read as padding and
invite "this is not the same bug." They are only useful as context for *why the error
message is unhelpful*, which is Issue C's point.

---

## Issue A — `F.interpolate(mode="nearest")` returns wrong pixels under `dynamic=True` on CUDA

**This is the strongest of the four: currently-shipping silent wrong results, reachable
from ordinary user code, no custom op and no `inference_mode`.** File this one first.

**Title:** `torch.compile(dynamic=True) F.interpolate(mode="nearest") selects the wrong source pixel on CUDA`

**Body:**

> ### 🐛 Describe the bug
>
> With dynamic shapes on CUDA, a compiled `F.interpolate(..., mode="nearest")` picks a
> different source pixel than eager for some input/output size ratios. It is an
> off-by-one in the source index, not a floating-point tolerance issue: the output is a
> different pixel, exactly wrong by one row/column.
>
> Static compilation is bit-exact. CPU is bit-exact. Only `dynamic=True` on CUDA differs.
>
> The input below is `arange` along the sampled dimension, so each output value *is* the
> source index it was gathered from — a mismatch is an exact integer.
>
> ```python
> import torch
> import torch.nn.functional as F
>
> def rows(t):
>     return t[0, 0, :, 0].to(torch.int64).tolist()
>
> for mode, i, o in [("nearest", 448, 192), ("nearest", 384, 363),
>                    ("nearest", 448, 368), ("nearest", 41, 82),
>                    ("nearest-exact", 384, 363)]:
>     x = (torch.arange(i, device="cuda", dtype=torch.float32)
>          .view(1, 1, i, 1).expand(1, 1, i, 4).contiguous())
>     f = lambda t: F.interpolate(t, size=(o, 4), mode=mode)
>     torch._dynamo.reset()
>     eager = rows(f(x))
>     compiled = rows(torch.compile(f, dynamic=True)(x))
>     wrong = [(d, e, g) for d, (e, g) in enumerate(zip(eager, compiled)) if e != g]
>     print(f"{mode} {i}->{o}: {len(wrong)}/{o} rows differ; first {wrong[:1]}")
> ```
>
> ### Actual
>
> ```
> nearest 448->192: 7/192 rows differ; first [(27, 62, 63)]
> nearest 384->363: 2/363 rows differ; first [(121, 127, 128)]
> nearest 448->368: 9/368 rows differ; first [(23, 27, 28)]
> nearest 41->82: 40/82 rows differ; first [(2, 1, 0)]
> nearest-exact 384->363: 2/363 rows differ; first [(60, 63, 64)]
> ```
>
> All of these reproduce on **A100 (SM 8.0) and H100 (SM 9.0)** — verified on H100
> 2026-08-30 and on two separate A100s.
>
> Worth stating so a non-reproduction is not mistaken for absence: **the visibility of a
> given ratio depends on the form of the emitted divide.** Inductor emits `ks0 / 74` when
> it can see the output size as a constant and `ks0 / ks1` when it cannot; in the second
> form both operands are runtime values and the approximate reciprocal is used. `37->74`
> is wrong only in the second form (36/74 vs 0/74), so I have left it out. Every ratio
> above is wrong in **both** forms.


>
> ### Expected
>
> All four print `0/N rows differ`, as they do with `dynamic=False` and as they do on CPU.
>
> ### Where it comes from
>
> With a symbolic output size the generated Triton computes the scale in-kernel from the
> size arguments:
>
> ```python
> tmp0 = (ks0 / 192).to(tl.float32)  # <- approximate divide, on device
> tmp1 = (x1).to(tl.float32)
> tmp3 = (tmp1 * tmp0).to(tl.int64)  # floor -> source index
> ```
>
> Statically the same expression is folded to a literal (`tl.full([1], 2.3333333, ...)`),
> which is why only the dynamic path differs.
>
> Eager computes the scale on the host as a correctly-rounded float32 divide
> (`compute_scales_value` in `ATen/native/UpSample.h`:
> `static_cast<float>(input_size) / output_size`, then
> `min(int64(floorf(dst_index * scale)), input_size - 1)`). Triton's `/` lowers to an
> approximate reciprocal (`x * rcp(y)`), so the quotient can be off by one ulp in
> **either** direction and change the floor. Measured fp32 bit patterns on A100:
>
> | ratio | eager (correctly rounded) | emitted `/` | error |
> |---|---|---|---|
> | 448->192 | `0x40155555` | `0x40155556` | **+1 ulp** (compiled picks a *higher* source row) |
> | 384->363 | `0x3f8767ab` | `0x3f8767ac` | **+1 ulp** |
> | 41->82 | `0x3f000000` | `0x3effffff` | **-1 ulp** (a *lower* row) |
>
> The error goes in either direction, and it does not track the scale's value: `41/82`,
> `37/74`, `74/148` and `101/202` all have the same correctly-rounded pattern
> `0x3f000000`, but only `41/82` comes back wrong from the device. Which argument pairs
> the reciprocal gets wrong is what decides, which is why I am listing measured ratios
> rather than a rule.
>
> Comparing the two divides directly against that exact ATen model, over
> `448->192, 384->363, 37->74, 32->64, 32->100, 63->31, 100->70, 1000->999, 255->256`
> (3305 output coordinates total):
>
> | divide used | coordinates disagreeing with eager |
> |---|---|
> | `I / O` (what is emitted today) | **45** |
> | `tl.math.div_rn(I, O)` | **0** |
>
> `448->192` accounts for 7 of the 45 — matching the 7 wrong rows above exactly — and
> `37->74` for 36. (`37->74` is in this sweep, which compares the two *divides* directly
> at a fixed form, but not in the end-to-end list above, where its visibility depends on
> which divide gets emitted. Both measurements are on the same H100.)
>
> ### Versions
>
> torch 2.13.0+cu130, triton 3.7.1, python 3.12.14. Reproduces on the release wheel on
> both **A100-SXM4-80GB** and **H100 80GB HBM3** (driver 575.57.08). CPU and
> `dynamic=False` are unaffected.

**Note for the PR that follows, not for the issue:** the fix touches the *decomposition*
(`arange/mul/_unsafe_index`), which is a different code path from the
`upsample_nearestnd` lowering in Issue B. Do not merge these two into one PR — see
the one-concern-per-PR reasoning below. And any fix here must confront #164144/#165566 head-on:
in the decomposition the scale divide is loop-invariant in the emitted kernel. The
remaining blocker is semantic: the native CUDA backward path is not always the
transpose of its own forward map. Add the eager reproduction to #97135 before proposing
code, and benchmark any coordinated change.

Worth noting in that PR: both paths expose a scalar shape divide, but they have
different downstream contracts. The lowering hoists its kernel-argument divide
(measured `div_rn/truediv` ≈ 1.02 at large shapes on A100); the decomposition's
forward fix must also preserve the native CUDA backward contract, which currently
fails the eager transpose check at sensitive ratios.

---

## Issue B — `upsample_nearestnd` lowering crashes on a symbolic output size

**Title:** `[inductor] upsample_nearestnd fails to lower with a symbolic output_size (ops.constant gets a sympy expression)`

**Body:**

> ### 🐛 Describe the bug
>
> `upsample_nearestnd` (`torch/_inductor/lowering.py`) guards its input sizes but not its
> output sizes, then divides one by the other and hands the quotient to `ops.constant`,
> which requires a concrete value:
>
> ```python
> i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]   # guarded
> o_sizes = output_size                                        # not guarded
> inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
> ...
> x = ops.mul(x, ops.constant(scale, torch.float32))
> ```
>
> When the output size is symbolic and the caller passed no explicit `scales_x` (i.e.
> `size=` rather than `scale_factor=`), the quotient stays a `sympy.Mul` and lowering
> aborts:
>
> ```
> NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
> ```
>
> ### Reachability — please read before triaging
>
> **The ATen `upsample_nearest*` ops do not reach this code.**
> `upsample_nearest{1,2,3}d` are in `torch._decomp.decomposition_table`, and Inductor
> applies those decompositions (`arange/mul/_unsafe_index`) before lowering. I measured
> zero lowering hits across `F.interpolate` with `size=` (static and dynamic) and with
> `scale_factor=`, `nn.Upsample`, raw `aten.upsample_nearest2d.default`, raw `.vec`,
> `inference_mode`, and `torch.export`.
>
> So this is **not** a user-visible regression today through `F.interpolate`. It is
> reachable by anything that calls the lowering directly — a custom lowering, or an
> out-of-tree backend reusing it — and it is reachable on `main`: driven that way,
> 12/12 shape pairs crash.
>
> Filing it because the function is live, exported, and silently wrong for a symbolic
> input rather than raising something meaningful; and because the same helper is what
> older versions *did* reach (on torch 2.3.1 this aborts a real SDXL/VAE decoder compile
> under dynamic shapes, which is how I ran into it).
>
> ### Repro
>
> ```python
> import torch
> from torch._inductor.lowering import register_lowering, upsample_nearestnd
>
> with torch.library._scoped_library("demo", "FRAGMENT") as lib:
>     lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
>     lib.impl("ups2d", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta")
>     lib.impl("ups2d", lambda x, h, w: torch.nn.functional.interpolate(
>         x, size=(int(h), int(w)), mode="nearest"), "CPU")
>     register_lowering(torch.ops.demo.ups2d)(
>         lambda x, h, w: upsample_nearestnd(x, [h, w], (None, None), n=2))
>
>     def fn(x, sizes):
>         return torch.ops.demo.ups2d(x, sizes[0], sizes[1])
>
>     x = torch.randn(1, 3, 32, 32)
>     ref = torch.randn(1, 3, 100, 70)
>     torch._dynamo.maybe_mark_dynamic(ref, 2)
>     torch._dynamo.maybe_mark_dynamic(ref, 3)
>     print(torch.compile(fn, dynamic=True)(x, ref.shape[-2:]).shape)
> ```
>
> ### Expected
>
> Lowers and matches eager. Guarding the output size would also work but specializes on
> it, costing a recompile per distinct output size.
>
> ### Versions
>
> torch 2.13.0+cu130 / A100; also present on 2.11.0 and on `main`.

---

## Issue C — `ops.constant` accepts a symbolic value and fails ~a dozen frames later

**Title:** `[inductor] ops.constant silently accepts a symbolic value; the failure surfaces far from the lowering that caused it`

**Body:**

> ### 🐛 Describe the bug
>
> `ops.constant` takes a concrete leaf value. Nothing checks that at the call, so a
> lowering that derives one from sizes it never made concrete gets no error at the
> mistake — the value flows on and dies while tracing, typically as
>
> ```
> NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
> ```
>
> raised from `torch/fx/proxy.py`, a dozen frames below the lowering that broke the
> contract, naming neither the op nor the fix. (On torch 2.3.1 the same mistake landed
> in `IndexPropagation` as `TypeError: Cannot convert expression to float` from
> `sympy/core/expr.py` — equally uninformative about the cause.)
>
> Debugging it means reading the whole lowering to find which value was never guarded.
>
> ### This is an existing convention, not a proposed new rule
>
> Several lowerings already dispatch on exactly this by hand — e.g. in `lowering.py`:
>
> ```python
> # Use index_expr for sympy expressions (e.g. from .item()), constant otherwise
> if isinstance(value, sympy.Basic):
>     value_expr = ops.index_expr(value, dtype)
> else:
>     value_expr = ops.constant(value, dtype)
> ```
>
> in `_full` and again in the addcmul/addcdiv helpers. The contract is real and already
> hand-enforced at several call sites; it just is not enforced centrally, so a lowering
> that forgets gets a bad error instead of a good one.
>
> ### Repro
>
> ```python
> import sympy, torch
> from torch._inductor.virtualized import ops
> ops.constant(32 * sympy.Symbol("s0", integer=True, positive=True), torch.float32)
> ```
>
> ### Expected
>
> An error at the call that names the op, shows the offending expression and its free
> symbols, and states both ways out: make the inputs concrete
> (`V.graph.sizevars.guard_int`, which specializes), or keep it dynamic and use
> `ops.index_expr`, which accepts symbolic expressions.
>
> `sympy.Integer` / `sympy.Float` are `sympy.Expr` but concrete and must keep working
> (`is_number`).
>
> ### Versions
>
> torch 2.13.0+cu130 / A100.

---

## Issue D — `upsample_nearest2d_backward` crashes on a symbolic `input_size`

**Title:** `[inductor] upsample_nearest2d_backward fails to lower with a symbolic input_size ('FloorDiv' cannot be interpreted as an integer)`

**Body:**

> ### 🐛 Describe the bug
>
> `upsample_nearest2d_backward` guards the grad tensor's spatial sizes but not
> `input_size`:
>
> ```python
> inp_h = V.graph.sizevars.guard_int(inp_h)
> inp_w = V.graph.sizevars.guard_int(inp_w)
> *_batch, out_h, out_w = input_size          # not guarded
> ```
>
> `out_h`/`out_w` are then used as divisors — `inp_h % out_h` is evaluated eagerly, and
> `ceildiv(inp_h, out_h)` becomes a `sympy.FloorDiv` — and that quotient becomes a
> `range()` bound inside `_adaptive_pooling_fn`, so lowering aborts:
>
> ```
> TypeError: 'FloorDiv' object cannot be interpreted as an integer
> ```
>
> ### Reachability
>
> Not reachable from autograd today, though not for the obvious reason: the forward is
> decomposed to `arange/mul/_unsafe_index` before autograd runs, so the backward graph
> contains `_unsafe_index_put` and `aten.upsample_nearest2d_backward` is never emitted.
> (The backward op is *not* itself in `torch._decomp.decomposition_table`.) Measured 0
> hits on this lowering across 8 real autograd configurations — nearest and
> nearest-exact, `size=` and `scale_factor=`, static and dynamic. Reachable by calling
> the aten op directly, which is what the repro does.
>
> ### Repro
>
> ```python
> import torch
>
> def fn(grad, ref):
>     return torch.ops.aten.upsample_nearest2d_backward.default(
>         grad, [grad.shape[-2], grad.shape[-1]],
>         [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]])
>
> grad = torch.randn(1, 3, 64, 64)
> ref = torch.randn(1, 3, 32, 32)
> torch._dynamo.maybe_mark_dynamic(ref, 2)
> torch._dynamo.maybe_mark_dynamic(ref, 3)
> print(torch.compile(fn, dynamic=True)(grad, ref).shape)
> ```
>
> Fails for 4/4 of `(64,64)->(32,32)`, `(63,63)->(32,32)`, `(96,96)->(32,32)`,
> `(100,70)->(50,35)` — covering both the exactly-divisible path that routes to
> `avg_pool2d` and the general adaptive-pooling path.
>
> ### Expected
>
> Lowers and matches eager. Unlike the forward case the division cannot be deferred into
> the kernel: the quotient is a Python loop bound over which the pooling window is
> unrolled, so it has to be concrete at lowering time — i.e. guard it, the same way the
> input sizes above are guarded.
>
> ### Versions
>
> torch 2.13.0+cu130 / A100.

---

## Issue E (optional, lower value) — `adaptive_avg_pool2d` under dynamic shapes

Found incidentally while proving the no-op property; **not** touched by any of the PRs
and confirmed identical with all three patches reverted, so it is pre-existing:

```
InductorError: LoweringException: TypeError: cannot determine truth value of
Relational: (((2*s38 + 62)//s38))*(((2*s47 + 62)//s47)) > 25
```

from `_adaptive_avg_pool2d` in `lowering.py`. Reproduces for `dynamic=True` on
`adaptive_avg_pool2d` with `(64,64)->(32,32)`, `(63,61)->(7,7)`, `(100,70)->(50,35)`;
`dynamic=False` is bit-exact for all three. Same family as B/D (a size-derived symbolic
value used where a concrete one is required), different function.

File it separately, or leave it out of this batch — filing five issues at once from a
new account reads worse than filing the strong ones. **Recommendation: file A, B, C, D;
hold E** until the first ones are triaged.

---

## Filing order and PR mapping

| # | Issue | PR to link | Strength |
|---|---|---|---|
| 1 | **A** wrong pixels, `dynamic=True`, CUDA | *(new — not yet written)* | strongest: silent wrong results from ordinary code |
| 2 | **B** `upsample_nearestnd` symbolic output size | PR 1 (`lowering.py` +30/−4, +3 tests) | fix written, GPU-verified |
| 3 | **C** `ops.constant` contract | PR 2 (`virtualized.py` +23, +2 tests) | fix written; a contract change, so expect debate |
| 4 | **D** backward `input_size` | PR 3 (`lowering.py` +7, +1 test) | fix written, smallest and cleanest |
| 5 | E `adaptive_avg_pool2d` | none | hold |

**Three separate PRs, not one.** This is my judgment, not a quoted rule — "one concern
per PR" does **not** appear in `CONTRIBUTING.md` or the wiki (checked 2026-08-30). B is a
lowering fix, C is a cross-cutting contract change, D is an unrelated function. Bundling
them means one objection stalls all three. D is the best first PR: 7 lines, one test,
no design argument.

**Ordering caveat:** PR 2 (Issue C) enforces the contract that PR 1 (Issue B) is
currently the only in-tree violator of. If C lands first, B's crash changes message but
still crashes. They are independent but worth mentioning in each PR's description so a
reviewer isn't surprised.

## Pre-send checklist

- [ ] Sign the CLA
- [ ] Re-probe hardware (`nvidia-smi -L`) and re-run every number; this container's GPU changes between sessions
- [ ] Re-run the prior-art searches above — they age
- [ ] Rebase onto current `main` (HEAD moved twice in one session previously)
- [ ] `lintrunner -a` on the changed files
- [ ] Re-run the guard matrix: each test must FAIL with only its own fix reverted
- [ ] Leave Reviewers empty; do not @-mention maintainers
- [ ] No AI-generated fix explanation in the issue body
- [ ] For each PR: state the reachability limits honestly (B and D are not ATen-reachable today) — overclaiming is the fastest way to lose review
