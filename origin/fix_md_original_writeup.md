# TorchInductor `index_propagation.py` crash — deep analysis & interview prep

> Working notes for the "persistence through uncertainty" story in `CV/CV.md` / SoP.
> Source file under analysis: `env/lib/python3.11/site-packages/torch/_inductor/index_propagation.py`
> (torch **2.3.1**), local copy at `fix_torch_compile/index_propagation.py`.
> The patched lines are **261–264** (the `try/except` around `propagate_sympy`).

> **STATUS 2026-08-25 — the artifact is built; this file is now the analysis, not the deliverable.**
> A standalone, public-safe reproduction + fix + test suite lives in
> **`optimizer-image/genai-sd-optimization/fix_torch_dynamic_project/`** (its own README; read that
> repo's CLAUDE.md status section first). What changed since these notes were written:
> - **§5's four open items are all RESOLVED** — the ZenAI environment was found, the bug reproduced
>   live, and it now reproduces **standalone on CPU in seconds**. The missing ingredient was
>   `torch.inference_mode()` (it excludes `DispatchKey.Autograd`, so the Autograd-keyed decomposition
>   never fires and the raw op reaches the buggy lowering). `no_grad` does *not* do this.
> - **§4's caveat was wrong and is corrected below** — `fallback()` does **not** fix this bug.
> - **The bug is already fixed in torch ≥ 2.4** (verified on 2.4.0/2.7.0/2.11.0/2.13.0). Never claim
>   an open PyTorch bug. A **regression test** PR is written and validated in that repo's
>   `upstream/`, pending send.

---

## 0. Payoff — the number this bug unlocked

The `torch.compile` fallback fix is a shared ingredient in the acceleration method
across models:

| Model / config                                                            | Hardware   | End-to-end inference | Speedup |
|---------------------------------------------------------------------------|------------|----------------------|---------|
| **FLUX.1-schnell (8-step)** — baseline (eager)                            | 1× H100    | **2.02 s**           | 1.0×    |
| FLUX.1-schnell — context parallel + `torch.compile` + first-block cache   | 2× H100    | **0.86 s**           | **≈2.35×** |
| **SDXL (DreamShaper-XL v2 Turbo, 8-step, 1024×1024)** — baseline (eager)   | 1× H100    | **1.26 s**           | 1.0×    |
| SDXL — CFG parallel + `torch.compile`                                     | 2× H100    | **0.72 s**           | **≈1.75×** |

Quality held at **≥95%** (FLUX) and **≥99%** (SDXL) of the baseline image (same prompt /
seed / steps), so these are real latency wins, not quality-for-speed trades. The compile crash documented
below had to be fixed first — without it `torch.compile` couldn't run on the model at
all, and the `torch.compile` leg of the method was blocked. Benchmarks:
`flux_schnell_8step.py`, `sdxl_8step.py`.

---

## 1. Where this code sits in the stack

`index_propagation.py` lives inside **TorchInductor**, the default compilation
backend that `torch.compile` lowers to (Dynamo → AOTAutograd → Inductor → Triton/C++).

When Inductor lowers your model to a kernel, it must generate not only the *math*
ops but also the **indexing arithmetic** — how each element maps to a memory offset
in `buf0[...]`. Inductor's codegen is written against an abstract **"ops" protocol**
(`ops.constant`, `ops.index_expr`, `ops.add`, `ops.mul`, `ops.indirect_indexing`,
`ops.load`, …).

`IndexPropagation` is an **ops-handler wrapper** that sits *in front of* the real
handler and performs compile-time symbolic folding: it lifts these ops into **SymPy
expressions** so that, e.g., an indirect index computed from `arange` collapses into
a plain static index (docstring example: `ops.load("buf0", x*2)` instead of a chain
of `index_expr → constant → mul → indirect_indexing`).

**One-sentence framing (good for a professor):** it's *abstract interpretation /
partial evaluation over the indexing sublanguage* — Inductor interprets the ops with
SymPy values instead of real values to fold indexing at compile time.

So the intuition "torch.compile wraps all operations into a sympy object" is correct:
that wrapping happens here.

---

## 2. The control flow that breaks

Every op routes through `__getattr__ → inner` (lines 249–265):

1. `SymPyOps` has no method for this op → `fallback` (hand it to the real handler).
2. Not all `IndexPropVar` args are symbolic → `fallback`.
3. Otherwise → `propagate_sympy(name, args, kwargs)` — try to build the SymPy expr.

`propagate_sympy` (227–247):

```
new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)   # BUILD the sympy expr
is_valid_expr = new_expr is not NotImplemented and (
    isinstance(new_expr.expr, sympy.Number) or new_expr.expr.is_integer)
if not is_valid_expr:
    return self.fallback(name, args, kwargs)                  # graceful giveup
return IndexPropVar.new_symbolic(new_expr)
```

**The structural flaw:** the validity guard runs *after* `new_expr` is already built.
The code assumes *constructing* the SymPy expression is always safe, and that only the
*result* might be unusable (float / non-integer). It never anticipated that the
**construction step itself could raise**.

---

## 3. What actually raises

The exception escapes the `getattr(SymPyOps, name)(...)` call — i.e. inside a
`SymPyOps` static method doing raw SymPy algebra (`x.expr * y.expr`, `FloorDiv(...)`,
`ModularIndexing(...)`, `sympy.Min/Max(...)`, `Where(...)`), or during SymPy's own
coercion when it tries to `sympify` an operand it doesn't recognize. A
"cannot convert … to expression"-type message is SymPy refusing to coerce some operand
into an `Expr`.

Strong hints from the file itself:

- The docstring says *"minimum and maximum cannot be translated to SymPy expressions
  yet"* — yet `SymPyOps.minimum/maximum` **do** call `sympy.Min/Max`. Mismatch.
- `to_dtype` and `is_valid_expr` deliberately let **float constants** leak into the
  symbolic path (*"Inductor doesn't handle floating point in sympy expressions well"*),
  while the integer-oriented constructors (`FloorDiv`, `ModularIndexing`, and the
  relational inside `Where`/`Min`/`Max`) can choke when a float or an already-non-`Expr`
  operand shows up.

**Most likely root trigger:** a specific op+dtype combination in the model — very
plausibly a **float operand feeding min/max/floordiv/mod** — that SymPy could not build
into a valid `Expr`. Possibly aggravated by **SymPy version skew**: Inductor 2.3.1 was
pinned to that era's coercion behavior, and small SymPy changes shift which operands
raise vs. return `NotImplemented`.

---

## 4. Why the fix works — and the honest caveat

The patch (lines 261–264) turns a **hard compile-time crash into graceful
degradation**: if symbolic folding of the op throws, stop folding it and re-materialize
it as a plain `index_expr`. That is exactly what the `is_valid_expr → fallback` branch
was *meant* to guarantee but couldn't — it only handled bad *results*, not construction
failures.

**Caveat to own — CORRECTED 2026-08-25 (the earlier version of this note was wrong):**
- The recovery that *looks* cleaner is `return self.fallback(name, args, kwargs)`, which unwraps the
  args and routes the **original** op through the real handler. This file previously called that
  "the more general and defensible recovery."
- **It was tested, and it does not work.** It raises
  `NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>`. The crash happens while
  *constructing* the constant, so the argument is still a `sympy.Mul`; `fallback()` hands it to a
  handler that expects a concrete number and the same type-contract violation resurfaces one layer
  down with a worse message.
- **`propagate_sympy("index_expr", ...)` is required, not a shortcut** — `index_expr` is precisely
  the op whose contract accepts a symbolic expression. So the narrow-looking fix is the correct one
  and the general-looking one is wrong.
- **Say this out loud in an interview.** "Just use the official fallback" is the first thing a
  reviewer suggests; having actually run it is the differentiator. Evidence:
  `test_fallback_alternative_is_insufficient` in
  `optimizer-image/genai-sd-optimization/fix_torch_dynamic_project/tests/`.
- Remaining honest caveat: `except Exception` is broad, so an unrelated failure inside
  `propagate_sympy` would be absorbed into the index path. `except TypeError` is the better
  blast-radius choice for a production deployment.

---

## 4b. How I actually localized it (the real debugging method)

This is worth telling because it shows *method*, not luck. The crash surfaced with a
cryptic SymPy message and no pointer to my own code, so:

- I **traced down through many Inductor files** to find where the ops actually get
  interpreted, and landed on `index_propagation.py`.
- I **instrumented the hot path** — printed the result of
  `return self.propagate_sympy(name, args, kwargs)` (and the `name`/args going in) — to
  watch the **stream of ops** as compilation ran: `ops.constant`, `ops.index_expr`,
  `ops.add`, `ops.mul`, `ops.indirect_indexing`, …
- Watching that sequence made the **pattern** visible: the ops fold fine up to a
  particular op, and the crash lands on one specific op in the chain. That's how I
  confirmed the failure is *inside expression construction for one op*, not a random
  global error — and where I placed the `try/except`.

**Why this reads well to a professor:** printing the op stream is exactly the right move
when you don't have a mental model yet — you turn an opaque symbolic pipeline into an
observable trace, let the *data* show you the pattern, then localize. Frame it as
"I made the invisible pipeline observable" rather than "I added print statements."

**One honest note:** `print`-tracing is the pragmatic first tool; the next-level answer
is "and once localized I could have set a conditional breakpoint on that op / logged the
operand dtype to confirm the float hypothesis." Mentioning that shows you know the
ladder of debugging tools, not just the bottom rung.

---

## 4c. Companion finding — `torch.compile`'s speedup collapses as shape variety grows

**Separate experiment, same project.** The crash above was about whether `torch.compile` could run
*at all*. This is about whether it stays *worth running* once the input shapes stop being fixed —
the question that actually decides if a compiled path survives in production, where users submit
arbitrary resolutions and batch sizes.

**What I measured.** Swept the number of *distinct* input shapes the compiled model sees at
inference and watched end-to-end latency:

| Distinct shapes seen | Behaviour |
|---|---|
| **< ~16** | `torch.compile` is efficient — the speedup holds |
| **increasing past that** | speedup **degrades progressively** as shape count rises |
| **~100** | **graph breaks / falls back — speed equals no compilation at all** |

**Why this happens (the mechanism to state in an interview).** Inductor specializes generated
kernels on shape. Every genuinely new shape is a **guard miss → recompilation**, so compile time is
paid repeatedly instead of once, and the code-cache advantage erodes. Dynamo also has a
**recompilation ceiling** per code object: past it, it stops specializing and falls back, so the
compiled path silently stops being a compiled path. The failure is **quiet** — no error, no warning
in the normal log, just the speedup disappearing. That is what makes it dangerous: a benchmark run
on one fixed resolution reports a large win that production never sees.

**Why this is the more interesting half of the story.** The crash fix was persistence; this is
**characterization** — it says *when* the technique works and *where its boundary is*, which is what
turns "I used `torch.compile`" into "I know what `torch.compile` costs." It also has a real design
implication: **bucket or pad shapes to a small fixed set** (keeping the count under the ceiling)
rather than compiling per-shape — a placement/scheduling decision the compiler cannot make for you,
because only the application knows which shapes are worth serving.

**Thematic link to the rest of the portfolio.** Same shape as the other findings: the limiting
factor was a **default nobody was accounting for** — here, the implicit assumption that shapes are
static. Cross-reference `multinode_management_unicamo.md` §5.1 (more GPUs made a pipeline slower)
and the Ray thread-default finding in `ray_learning/`.

> ⚠️ **Measurement status — read before quoting.** The **thresholds are remembered, not currently
> backed by a saved sweep**: `< 16` efficient, degrading in between, `~100` falling back to
> uncompiled speed. No log or CSV in this folder records them, and `flux_schnell_8step.py` /
> `sdxl_8step.py` are fixed-shape benchmarks.
> - **Safe to say now:** the qualitative shape of the curve — "the speedup holds for a small number
>   of distinct shapes, degrades as variety grows, and by ~100 shapes falls back to roughly
>   uncompiled speed." Present the numbers as approximate, which is how they are written above.
> - **Do not** present these as measured datapoints, quote a per-shape latency, or name a precise
>   crossover — the honest framing is "approximately," and if pressed, "I did not keep the sweep log."
> - **To make it bulletproof** (cheap, and does not need the original ZenAI environment — any
>   `torch.compile`-able model on one GPU will reproduce the phenomenon): sweep N ∈ {1, 2, 4, 8, 16,
>   32, 64, 100} distinct shapes, record end-to-end latency per N plus `torch._dynamo.utils`
>   recompile counts, and set `TORCH_LOGS="recompiles,graph_breaks"` to capture the guard misses and
>   the fallback directly. That converts this from a remembered threshold into a curve with a
>   located ceiling — and it is the single cheapest experiment in the portfolio, since it needs
>   neither 8 GPUs nor the lost ZenAI setup.
> - Note the torch version matters: dynamic-shape handling changed substantially after 2.3.1, so
>   state the version alongside the claim.

---

## 5. RESOLVED 2026-08-25 — all four items answered, nothing left pending

These four were blocked on "the lost ZenAI environment." **That environment was found**, the bug was
reproduced live, and a **standalone CPU reproduction** now exists. Working artifact + evidence:
`optimizer-image/genai-sd-optimization/fix_torch_dynamic_project/` (README §2/§3/§5, tests green in
both patch states). Do **not** re-open these as TODOs — unchecked boxes read as unfinished.

- [x] **Full traceback under 2.3.1** — captured, and asserted by test rather than pasted:
      `lowering.py:3528 fn` → `lowering.py:3520 scale_fn` → `virtualized.py:261` →
      `index_propagation.py:262 inner` → `:238 propagate_sympy` → `:62 constant` →
      `sympy/core/expr.py:375 __float__` → `TypeError: Cannot convert expression to float`.
      `test_failure_goes_through_the_claimed_frames` pins the three load-bearing frames.
- [x] **Triggering op and operand** — `ops.constant(scale, torch.float32)` inside `scale_fn`, reached
      from `upsample_nearestnd` (`F.interpolate(..., mode="nearest")` / `aten.upsample_nearest2d`).
      The operand is **not** a stray float: it is a `sympy.Expr` (e.g. `32/s0`), because `o_sizes`
      is never concretised while `i_sizes` is. The earlier "a float sneaks in" hypothesis was wrong.
- [x] **Version pair** — `torch 2.3.1+cu121`, `sympy 1.14.0`, python 3.11.15. Reproduces on **CPU**
      and on H100.
- [x] **Does `fallback()` also fix it? NO — and this is the interesting answer.** See §4's corrected
      caveat below. It raises `NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>`.
      `index_expr` is *required*, not merely pragmatic.

**Also settled: the bug is already fixed upstream (torch ≥ 2.4).** Verified on 2.4.0/2.7.0/2.11.0/
2.13.0 — all compile it bit-exact. Two independent upstream fixes: the decomposition gained
`py_impl(DispatchKey.CompositeImplicitAutograd)` (primary — it now fires under `inference_mode`, so
the buggy lowering is unreachable), and `SymPyOps.constant` dropped its `sympy.Float(float(value))`
coercion. **The caller-side defect is still in `main` (72e0517), latent.** Never write "fixed a bug
in PyTorch" — the accurate claim is in that repo's CLAUDE.md.

---

## 6. How to answer when a professor asks

### The 30-second version (say this first)
> "`torch.compile` lowers through Inductor, which folds indexing math at compile time by
> lifting ops into SymPy expressions. On my model, one op couldn't be built into a valid
> SymPy expression and it crashed — but the crash happened *while constructing* the
> expression, whereas Inductor's guard only checked the expression *after* it was built.
> I traced it into `index_propagation.py`, saw the missing failure path, and added a
> fallback so an unrepresentable op degrades to a plain index expression instead of
> aborting the whole compile."

### If they push for the root cause
> "SymPy couldn't coerce the operand into an `Expr` — most likely a float feeding an
> integer-oriented constructor like `FloorDiv`/`ModularIndexing` or a `Min`/`Max`, which
> the file's own docstring even flags as not fully supported. I confirmed the failing op
> by printing the op stream (`constant`, `index_expr`, `add`, `mul`, `indirect_indexing`,
> …) as it folded and watching which op the crash landed on. There's also a plausible
> torch-2.3.1-vs-SymPy version-skew component."

### If they probe the fix quality (the differentiator)
> "The pragmatic fix re-routes to the `index_expr` path on exception, which resolved my
> case. The more principled fix mirrors the existing giveup branch — `fallback(name,
> args, kwargs)` — routing the original op through the real handler. Same idea: never let
> a compile-time symbolic-folding failure be fatal; treat it as 'can't fold, emit
> directly.'"

### If they ask what you learned about `torch.compile` itself (the strongest follow-up — see §4c)
> "That its benefit is conditional on shape stability, which production violates. I swept the
> number of distinct input shapes: under roughly 16 the speedup holds, then it degrades as variety
> grows, and by about 100 shapes it falls back to uncompiled speed. Every new shape is a guard miss
> and a recompilation, and Dynamo has a recompile ceiling per code object — past it, it stops
> specializing. The dangerous part is that it's silent: no error, just the win quietly disappearing,
> so a fixed-resolution benchmark reports a speedup production never sees. The practical response is
> to bucket or pad shapes into a small fixed set rather than compile per-shape — and that's a
> decision only the application can make, since the compiler doesn't know which shapes matter."
>
> **If pressed on rigor:** "Those thresholds are approximate — I observed the curve while getting the
> compiled path into production and didn't keep the sweep log. It's cheap to reproduce: sweep N
> shapes with `TORCH_LOGS=recompiles,graph_breaks` and the recompile counters. I'd want to re-measure
> before putting exact numbers on it, and note that dynamic-shape handling changed after 2.3.1."

### The meta-point (this is what they're actually testing)
Don't sell it as "I patched a PyTorch bug." Sell the **process**:
1. A black-box symptom (compile crash, cryptic SymPy message) with no obvious link to my code.
2. Built the mental model of the stack: Dynamo → AOTAutograd → **Inductor** → indexing → SymPy folding.
3. **Made the pipeline observable** — traced through many Inductor files and printed the op
   stream through `propagate_sympy` to let the data reveal which op fails.
4. Localized the failure to one function and understood *why the guard didn't catch it*.
5. Chose a minimal, semantics-preserving fix and can articulate its limits vs. the ideal.

That arc — **persistence through uncertainty + reading unfamiliar systems code + a fix
whose tradeoffs I understand** — is the story. The bug is just the vehicle.

### Traps to avoid
- Don't overclaim ("I fixed PyTorch"): it's a **local workaround** in a vendored file,
  not an upstreamed patch. **And do not offer "I could upstream it" as the follow-up** —
  as of 2026-08-25 that line is falsified: the bug is already fixed in torch ≥ 2.4
  (verified on 2.4.0/2.7.0/2.11.0/2.13.0). The correct follow-up is *"upstream fixed it at
  two layers by 2.4 — a dispatch-key addition and dropping the coercion — but left the
  caller-side contract violation latent, so I wrote the regression test that pins it."*
- Don't call it a "SymPy bug" — it's an **Inductor assumption** (construction can't fail)
  meeting an operand SymPy won't accept.
- The exact op **is** now captured, so state it plainly: `ops.constant(scale, torch.float32)`
  in `scale_fn` inside `upsample_nearestnd`, with `scale` a `sympy.Expr` such as `32/s0`.
  (Calibrated uncertainty is still right for anything *not* in §5's resolved list.)
- Don't say `fallback()` is the cleaner fix — it does not work. See §4.
