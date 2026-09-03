# A TorchInductor dynamic-shape bug, from HTTP 500 to root cause

I was serving SDXL behind an HTTP endpoint. Requests at some resolutions returned
**500 with a `sympy` traceback** — symbolic-algebra errors from a request to generate
an image. This repository is how I got from that to a one-line fix in
`torch/_inductor/index_propagation.py`, and to three patches plus four bug reports
against current PyTorch `main`.

Everything reproduces **on a laptop CPU in seconds**, with `torch==2.3.1` as the only
dependency.

```bash
pip install torch==2.3.1
./demo.sh cpu          # fail -> explain -> fix -> verify, ~2 min
```

**What I want you to take from this repo:** the interesting work was not the patch. It
was that the bug needed *three simultaneous ingredients*, the third of which is
invisible unless you know how ATen dispatch interacts with decompositions — and four
reasonable attempts to reproduce it failed first (§4).

I found the production failure, reduced it, and wrote the original fix while debugging
my own dynamic-shape workload. I later used Claude and Codex as supporting tools: to
challenge measurements, organize the upstream material, and look for adjacent failure
modes. The core bug and fix are my work; assistant-derived follow-ups are labelled as
such instead of being folded into that story.

---

## Honest status, before anything else

I would rather you read these five lines than discover them later.

| | |
|---|---|
| Is this an open bug in current PyTorch? | **No.** torch ≥ 2.4 compiles it (verified on 2.4.0, 2.7.0, 2.11.0, 2.13.0). §3 explains what upstream changed. |
| Was it a wrong-results bug? | **No — a compile-time crash.** Numerics were bit-exact throughout. |
| Is anything merged into PyTorch? | **No. Nothing has been pushed.** The three patches are written, GPU-verified, and staged. |
| Then why does the repo exist? | The deployment was pinned to 2.3.1 and needed the compiled path. The reasoning is the deliverable. |
| Did the same investigation find anything live? | **Yes** — plain `F.interpolate` under `dynamic=True` on CUDA returns **wrong pixels** on current 2.13. A forward prototype works, but a newly isolated CUDA backward inconsistency makes it unsafe to submit alone ([§6](#6-upstream-what-i-found-on-current-main)). |

---

## 1. Why a diffusion server needs the configuration that breaks

Two properties of this workload collide, and the collision is what makes the bug
reachable.

**The compiled path is load-bearing.** Diffusion inference is dozens of sequential
denoising steps through one UNet — hundreds of small kernel launches per step,
dominated by launch overhead and memory traffic rather than arithmetic. Fusing that
graph is worth a large constant factor, so losing it was not an option.

**The input resolution belongs to the user.** A ComfyUI graph exposes width and
height. Users ask for 1024×1024, 832×1216, and whatever an uploaded img2img image
happened to be. The server sees a *stream of distinct shapes*.

Those fight, because `torch.compile` specializes on shape:

| | behaviour | why it fails here |
|---|---|---|
| `dynamic=False` (default) | one graph **per distinct shape** | every new resolution stalls a live request on a recompile. Past `cache_size_limit` (**8** on 2.3.1) it stops specializing and silently falls back to **eager** |
| **`dynamic=True`** | **one** graph, shape as a runtime variable | compile once, serve every resolution — the correct choice for a server |

So `dynamic=True` is not an exotic flag; for multi-resolution serving it is the only
configuration with bounded compile cost. **This bug is what happens when you pick it.**

### Why the torch version could not simply be raised

The obvious fix to a bug fixed in 2.4 is to upgrade. That option did not exist: the
serving stack used **OneDiff + nexfort**, and nexfort ships as prebuilt CPython
extensions linked against libtorch — `nexfort/_C_inductor...so` carries **87
undefined `at::`/`c10::` symbols** and `NEEDED` entries for `libtorch.so`,
`libtorch_cpu.so`, `libc10.so`, `libtorch_python.so`. That is a C++ ABI coupling, not
a version range you can widen: OneDiff's own nexfort instructions pin
`torch==2.3.0 torchvision==0.18.0 torchao==0.1`, and nexfort's source gates behaviour
on `torch_version_compare` down to `torch<2.2.0`.

This is the general shape of the problem, and it is why a fix for an
already-fixed-upstream bug was still worth writing: **a compiled accelerator pins your
torch version, and a compiler bug in that version is yours to fix.** Anyone on
onediff, nexfort, or a vendor-pinned wheel is in the same position.

### The exact condition that reaches the bug

A U-shaped decoder upsamples and concatenates with an encoder skip. The idiomatic way
to guarantee the two agree on spatial size is to name the size explicitly:

```python
x = F.interpolate(x, size=(skip.shape[-2], skip.shape[-1]), mode="nearest")
```

Ordinary, correct PyTorch. But there are two ways to write an upsample and only one is
dangerous:

| form | what Inductor does with the scale | safe? |
|---|---|---|
| `F.interpolate(x, scale_factor=2.0)` | takes the `1.0 / scale` path — a concrete float | ✅ |
| `F.interpolate(x, size=(h, w))` | computes `input_size / output_size` — **symbolic if the output size is** | ❌ **the bug** |

In `diffusers`, `Upsample2D.forward` has exactly this branch, and the safe one is the
default:

```python
if output_size is None:
    hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
else:
    hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
```

The dangerous branch fires when the UNet forces the output size, and
`UNet2DConditionModel.forward` decides that as:

```python
if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
    forward_upsample_size = True
...
    upsample_size = down_block_res_samples[-1].shape[2:]   # symbolic under dynamic=True
```

So the trigger is precise: **a latent resolution that is not a multiple of
`2**num_upsamplers`** (8 for a 4-stage SDXL UNet; the latent grid is the image ÷ 8)
makes the UNet forward an explicit `upsample_size` read off a tensor's `.shape` —
symbolic under `dynamic=True`. Whether a request lands on the safe path or the
crashing one is **decided by the user, per request**.

A resolution that divides evenly compiles. One that does not aborts with:

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
TypeError: Cannot convert expression to float
```

> The `diffusers` excerpts establish that the `size=` form occurs in real model code
> under a documented condition. This repo does **not** import diffusers, ComfyUI, or
> any model weights — `repro/unet_like.py` reproduces the failure with a ~40-line
> decoder and random weights, and `repro/ablation.py` verifies the
> `scale_factor=`/`size=` asymmetry directly against Inductor.

---

## 2. The failure, and how to read it

```
  File "torch/_inductor/lowering.py", line 3528, in fn
    [*b, *[scale_fn(i, s, size) for i, s, size in zip(x, inv_scales, i_sizes)]]
  File "torch/_inductor/lowering.py", line 3520, in scale_fn
    x = ops.mul(x, ops.constant(scale, torch.float32))
  File "torch/_inductor/virtualized.py", line 261, in inner
  File "torch/_inductor/index_propagation.py", line 262, in inner
    return self.propagate_sympy(name, args, kwargs)
  File "torch/_inductor/index_propagation.py", line 238, in propagate_sympy
    new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
  File "torch/_inductor/index_propagation.py", line 62, in constant
    expr = sympy.Float(float(value))
  File "sympy/core/expr.py", line 375, in __float__
    raise TypeError("Cannot convert expression to float")
TypeError: Cannot convert expression to float
```

Three properties of this traceback generalize to most compiler bugs:

1. **The exception site is not the defect site.** `sympy` is four frames below the
   mistake and is behaving correctly — it was asked to float() a symbolic expression.
2. **The error text names nothing you searched for.** No "upsample", no "interpolate",
   no "dynamic shape". Searching it finds unrelated sympy questions.
3. **The useful frame is the topmost compiler frame, not the bottom one.**

So the reading rule is **upward, not downward**: find the highest frame still inside
the compiler, then read that function's source. `repro/repro.py` asserts these
specific frames rather than matching the error string, so a similar-looking
`TypeError` cannot count as a reproduction.

> **Primary source.** The capture from the production worker is
> [`origin/production_traceback.txt`](origin/production_traceback.txt) — 711 lines, the
> same error **8 times** across the retry loop, which is how one compile failure became
> a sustained outage rather than one bad request. The frames above were read out of it.
> In this develop tree the file is unredacted and carries the deployment's absolute
> paths; the public cut replaces them with `<site-packages>/` / `<server>/` and leaves
> every frame, line number and source line verbatim
> ([`origin/README.md`](origin/README.md), `PUBLISH.md`).

---

## 3. Root cause

`upsample_nearestnd` (`_inductor/lowering.py:3492`) computes its scale factors like
this:

```python
i_sizes = [V.graph.sizevars.evaluate_static_shape(i) for i in i_sizes]  # concretised
o_sizes = output_size                                                   # NOT concretised

inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]   # sympy.Expr if o is symbolic
for i, scale in enumerate(scales_x):
    if scale is not None:
        inv_scales[i] = 1.0 / scale                      # concrete float — safe path
```

**Input** sizes are forced concrete; **output** sizes are not. With `size=` (so
`scales_x` is all `None`) and a symbolic output size, `inv_scales[i]` stays a
`sympy.Expr` — and flows into `ops.constant(scale, torch.float32)`, which is
contractually a *leaf constant*. `IndexPropagation` routes it to `SymPyOps.constant`,
which calls `float()` on it.

**The defect is a type-contract violation one layer above where it blows up.** That
gap is exactly why the error message is useless.

### The third ingredient: why this is normally unreachable

`aten.upsample_nearest2d` has a decomposition registered on `DispatchKey.Autograd`.
Normally it fires first, rewriting the op into `arange → mul → _to_copy →
_unsafe_index`, so the buggy lowering is **never called**:

```
$ python repro/why_it_survives.py --device cpu

--- grad enabled (default) ---   aten.upsample_nearest2d survives: False   compiles fine
--- torch.no_grad() ---          aten.upsample_nearest2d survives: False   compiles fine
--- torch.inference_mode() ---   aten.upsample_nearest2d survives: True    FAILS with the bug
```

**`inference_mode` excludes `DispatchKey.Autograd`**, so the `py_impl` never fires and
the raw ATen op reaches the lowering intact. **`no_grad` does not** — it only stops
gradient recording and leaves the key in the dispatch set. The two contexts are
interchangeable for almost every purpose; here they are not.

**The protection against this bug was never in the lowering — it was an accident of
dispatch ordering.** And the configuration that removes the accident,
`inference_mode` + dynamic shapes, is precisely what you deploy in production.

---

## 4. The fix, and four attempts that failed first

`propagate_sympy` already has a fallback for values SymPy cannot represent; it was
just unreachable from an exception. One hunk:

```diff
-            return self.propagate_sympy(name, args, kwargs)
+            try:
+                return self.propagate_sympy(name, args, kwargs)
+            except Exception:
+                # A value that cannot be lowered as a float constant (e.g. a symbolic
+                # output size from the upsample_nearestnd lowering) is treated as an
+                # index expression instead of aborting the whole compilation.
+                return self.propagate_sympy("index_expr", args, kwargs)
```

A value that cannot be a float constant becomes a **symbolic index expression**.
Inductor then emits the index arithmetic it would have used anyway — which is why
results are bit-exact rather than merely close.

### Verification

A patch that compiles and returns wrong pixels would be worse than the crash:

```
$ python repro/verify_fix.py --device cpu
in -> out              bit-exact   max |diff|
(32, 32) -> (64, 64)   True        0            exact 2x upsample
(32, 32) -> (100, 70)  True        0            non-integer ratios
(32, 32) -> (17, 17)   True        0            downsample
(7, 13)  -> (29, 31)   True        0            odd/prime sizes
(64, 64) -> (256, 256) True        0            large 4x
PASS: every case is bit-exact against eager.       (7 sizes, one compiled artifact)
```

Bit-exact on CPU **and** CUDA. `repro/unet_like.py` runs the realistic case, where
output is *close* (≈7.5e-08 CPU, ≈1.7e-05 CUDA) rather than exact — that model has
convolutions, and Inductor fuses float arithmetic and picks different conv kernels
than eager. Correct behaviour: upsample indices are *integer* arithmetic, so a genuine
index error appears as a large diff, not a rounding difference.

### Why `index_expr` and not `fallback()` — tested, not assumed

The obvious alternative is Inductor's own escape hatch, `self.fallback(...)`. It looks
more general and more defensible. **It does not work:**

```
except Exception:
    return self.fallback(name, args, kwargs)
  -> BackendCompilerFailed: NotImplementedError: argument of type: <class 'sympy.core.mul.Mul'>
```

The crash happens while *constructing* the constant, so the argument is still a
`sympy.Mul`; `fallback()` hands it to a handler expecting a number and the same
violation resurfaces one layer down. `index_expr` works precisely because its contract
*does* accept symbolic expressions. Worth stating plainly, because "use the official
fallback" is the first thing a reviewer suggests — `test_fallback_alternative_is_insufficient`
pins it.

### The four failed reproductions

Reading the source gave me the mechanism and predicted the trigger (`size=`, not
`scale_factor=`). Reproducing it still took five attempts:

| attempt | result |
|---|---|
| `interpolate(scale_factor=2.0)`, `dynamic=True` | no error — concrete `1.0/scale` path |
| `interpolate(size=(h*2, w*2))`, dynamic and static | no error — backed symbols get resolved |
| `size=(h*3//2, ...)` + `mark_dynamic` | no error |
| unbacked symint via `n.item()` | a *different* bug: `GuardOnDataDependentSymNode` |
| **`size=` from a `mark_dynamic` tensor, under `inference_mode`** | **reproduces** |

The resolution came from asking *why the safe path is normally safe*, not from trying
more shapes. Instrumenting the lowering showed it was never being called at all — the
decomposition was eating the op first. The production worker ran under
`inference_mode`; my attempts did not.

**The transferable lesson: when a bug will not reproduce, the missing ingredient is
often ambient context — dispatch keys, grad mode, config — rather than data.** Four
attempts varied the data. The answer was in the state around it.

`repro/ablation.py` compiles nine variants and shows all three ingredients are
required, and maps the blast radius: `nearest` **and** `nearest-exact`, in **1-D, 2-D
and 3-D**, all fail — the defect is in the shared helper, not one entry point.
`bilinear` has a different lowering and is unaffected.

### Scope, honestly

- **Already fixed in torch ≥ 2.4.** Upstream fixed the *route to* the defect, at two
  layers, and neither is what I would have proposed. **(1) The dispatch registration**
  — from 2.4 the decomposition also carries
  `py_impl(DispatchKey.CompositeImplicitAutograd)`, which *is* active under
  `inference_mode`. This is the one that closes the hole. **(2) The callee** —
  `SymPyOps.constant` dropped its `sympy.Float(float(value))` coercion. Backporting
  only (2) onto 2.3.1 **does not** make the repro pass (measured): the failure
  relocates to `fallback()` and dies with `NotImplementedError`. Only the dispatch
  guard actually prevents it.
- **The arithmetic is still in `main`.** `upsample_nearestnd` still divides unguarded
  `o_sizes`, and reached directly it still crashes — 12/12 shape pairs on 2.11 and
  2.13. Made unreachable through ATen, not repaired. That is what §6 is about.
- **`except Exception` is deliberately broad.** An unrelated failure inside
  `propagate_sympy` would now be absorbed into the index path rather than surfacing.
  `except TypeError` would cover this bug; for a production deployment the narrow form
  is the better blast-radius bet.
- **This is a local mitigation for a pinned version**, not a patch PyTorch should take.

---

## 5. Reproduce it yourself

```bash
python patch/patch.py status        # is the installed torch patched?
python repro/repro.py               # compile the bug            -> FAILS (exit 2)
python repro/ablation.py            # why a naive repro doesn't fail
python repro/why_it_survives.py     # the dispatch mechanism, with graphs
python repro/unet_like.py           # the same bug in a UNet decoder
python patch/patch.py apply         # the fix (backs up first)
python repro/repro.py               # same operation             -> SUCCEEDS (exit 0)
python repro/verify_fix.py          # and is bit-exact vs eager
python tests/run_tests.py           # 15 pass unpatched / 18 patched
python patch/patch.py revert
/other/env/bin/python repro/check_upstream.py    # confirm >= 2.4 is unaffected
```

`patch.py` locates `index_propagation.py` via `torch.__file__`, so it always edits the
torch your interpreter imports — never a hardcoded path. It backs up before its first
edit, is idempotent, and **refuses to touch a file matching neither the pristine nor
the patched form**. Exit codes are scriptable: **2** = bug reproduced, **0** =
compiled correctly, **3** = environment problem.

The suite is **state-aware** — tests needing the opposite patch state skip with a
message naming the command to run, so any single run is green. Beyond "it raises" /
"it doesn't," it pins the *explanation*: that the traceback really passes through
`scale_fn → propagate_sympy → constant`, and that the op survives into the graph under
`inference_mode` but is decomposed away under `no_grad`. The root-cause story cannot
silently rot.

---

## 6. Upstream: what I found on current `main`

Having understood the 2.3.1 defect, I ran the same investigation against `main`. Four
independent findings, first on A100 and H100 and now re-verified on an RTX PRO 6000
Blackwell GPU, torch 2.13.0+cu130.
Narrative in **[`PR.md`](PR.md)**; submission material in **[`upstream/`](upstream/)**;
every measurement in **[`evidence/`](evidence/)**.

### ⭐ A — `F.interpolate(mode="nearest")` returns wrong pixels (no fix, issue only)

The consequential one: **silent wrong results on current PyTorch from ordinary user
code.** Stock 2.13.0+cu130, no patches, no custom op, no `inference_mode`.

| ratio | rows wrong | first disagreement |
|---|---|---|
| 448→192 | 7/192 | out row 27: eager src 62, compiled 63 |
| 384→363 | 2/363 | out row 121: eager 127, compiled 128 |
| 448→368 | 9/368 | out row 23: eager 27, compiled 28 |
| 41→82 | 40/82 | out row 2: eager 1, compiled 0 |

**CUDA only, dynamic only** — CPU exact everywhere, `dynamic=False` exact everywhere.
An off-by-one in the *source index*, not a tolerance issue. The dynamic path computes
the scale on-device as `(ks0 / 192).to(tl.float32)` — an approximate reciprocal — where
the static path folds it to a literal (`tl.full([1], 2.3333333333333335, ...)`).
Modelling ATen's `UpSample.h` exactly over 9 ratios / 3305 coordinates: the emitted
divide disagrees with eager on **45**, `div_rn` on **0**.

It lives in the **decomposition**, a different code path from the lowering the patches
touch. I initially described its divide as per-element; emitted-code inspection proved
that wrong. The scale divide is loop-invariant. That removes one performance objection,
but it does not make the obvious patch safe. Both a decomposition change and a
symbolic-printer prototype repair the forward result while reducing the gradient matrix
from 25/28 to 24/28.

The reason is deeper than Inductor. A new eager-only check reconstructs the exact source
indices selected by CUDA forward and compares `forward(x).sum().backward()` with their
histogram. On Blackwell, native CUDA backward is not the transpose of native forward at
three ULP-sensitive cases (14/448 wrong input gradients for nearest 448→192, 4/384 for
nearest 384→363, and 4/384 for nearest-exact 384→363); every CPU control passes. This
matches the still-open [#97135](https://github.com/pytorch/pytorch/issues/97135), so the
right upstream action is to add the fresh reproduction there and coordinate forward and
backward semantics before proposing the dynamic-forward fix.

⚠️ **Which ratios are visible depends on the emitted divide's form, not just the GPU** —
worth knowing before anyone reports a non-reproduction. Inductor emits `ks0 / 192` when
it can see the output size as a constant and `ks0 / ks1` when it cannot, and only the
second form puts *both* operands through the hardware reciprocal. Whether you get one
or the other turns on ordinary Python scoping: a closure over a module global folds the
size; a closure over a local keeps it symbolic. `37→74` is wrong **only** in the second
form (36/74 vs 0/74), so it is excluded from the report; all four ratios above are wrong
in **both**, on A100 and H100. `evidence/issueA_runtime_divide.py` prints both columns
per ratio, one process each.

### B, C, D — three fixes, written and staged

| | defect | fix |
|---|---|---|
| **B** → PR1 | `upsample_nearestnd` crashes on symbolic output size | defer the divide to the kernel via `ops.div_rn`, `lowering.py` +30/−4 |
| **C** → PR2 | `ops.constant` accepts a symbolic value, fails ~12 frames later in `fx/proxy.py` | reject at the call site, `virtualized.py` +23 |
| **D** → PR3 | `upsample_nearest2d_backward` crashes on symbolic `input_size` | guard it, `lowering.py` +7 |

**PR1 refuses to guard; PR3 guards.** Not inconsistent — structural:

| | forward (PR1) | backward (PR3) |
|---|---|---|
| the quotient becomes | a scalar multiplier | a `range()` bound |
| so it lives in | the kernel **argument** list | the kernel **body** |
| deferrable? | yes — one kernel serves every size | no — the window is unrolled at lowering time |
| measured, 7 sizes | **1** lowering call (vs 7 guarded) | necessarily one per window |

A 2×2 pooling window is 4 unrolled loads; 8×8 is 64. Different kernels, not one kernel
with a different argument.

⚠️ **PRs 1 and 3 are not ATen-reachable today** — the decompositions fire first
(measured: **0** lowering hits across 8 entry points). Both PR bodies say so up front.
Overclaiming reachability is the fastest way to lose a review.

### Evidence

- **6 new tests pass, and each fails with only its own fix reverted.** A test that
  passes with its fix reverted is not a regression test. Full matrix re-run on H100
  2026-08-30.
- **Strict no-op: 63/63 compiled results bit-identical with and without the patches**,
  same GPU, same session — 63 real `F.interpolate`/`nn.Upsample` configurations
  (`size=` and `scale_factor=`, 1-D/2-D/3-D, static and dynamic, forward and autograd).
- **42/42 adversarial configs bit-exact** — mixed symbolic/concrete dims, explicit
  `scales_x` on one dim, ULP-hostile ratios, fp16/bf16, non-contiguous, degenerate
  sizes. Coverage *verified* by instrumenting the predicate, so the deferred branch is
  known to have run.
- **Full dynamic-shapes suite: 2621 tests, 1 failure — pre-existing**, identical with
  all patches reverted.
- **RTX PRO 6000 Blackwell:** the full six-test guard matrix, 42/42 adversarial cases,
  and a same-device 63-case ON/OFF digest all pass. The controlled division benchmark
  resolves no slowdown on this GPU, but does not replace the A100 ≈2% result or imply
  anything about B200. See [`evidence/logs/blackwell_20260903.md`](evidence/logs/blackwell_20260903.md).
- **Codex follow-up:** `TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING=1` fixes ordinary
  integer-tensor division but does not reach the equivalent `IntTrueDiv` symbolic-shape
  path. The generic repro remains wrong at 191/192 values. This is a separate lead, not
  part of the three prepared patches.

### Status

**Nothing filed, nothing pushed.** The gate is procedural: PyTorch will not review a
new contributor's PR without a linked issue labelled `actionable`, so the order is
*file issue → wait for the label → push*. Remaining: sign the CLA, file the issues,
rebase, `lintrunner -a`. `upstream/SUBMIT.md` has the commands and the claim limits.

---

## 7. Two corrections and one trap, kept on purpose

A repo that only records what worked is not evidence of judgment.

**A wrong first diagnosis.** About 1 run in 18 failed with a missing
`triton_.cubin.tmp.pid_*`. I blamed the network filesystem — Inductor caches under
`$HOME`, which is a fleet-wide NFS export here. Plausible, and **wrong**: redirecting
the cache to node-local `/tmp` did not change the rate. The real cause is concurrency
*inside one process* — `compile_threads=32` forking against Triton's unlocked
create-then-`os.replace` (`triton/runtime/cache.py:108`). A race regardless of
filesystem. `repro/__init__.py` sets `TORCHINDUCTOR_COMPILE_THREADS=1` before
importing torch; **30/30 runs clean**, from ~1-in-18. The cheap experiment falsified
the appealing hypothesis.

**A performance claim that was a measurement artifact.** I reported `div_rn` as
0.96–0.98× — *faster* than `truediv`. Wrong: that came from timing each variant once,
in a fixed order, on a GPU idling at 210 MHz against 1410 MHz. The first-measured
variant ate the clock ramp. Re-run with CUDA events, 15 alternations, and an A-vs-A
control first: **≈2% slower** at large shapes. Corrected everywhere it had propagated.
The structural argument that actually answers the B200 objection — the divisor is a
kernel argument, so the divide is loop-invariant — was never in doubt.

**A test that passed for the wrong reason.** My test registered a lowering per loop
iteration under one op name. The FX graph cache keys on the *op name*, so a flag
captured in the lowering's closure is invisible to it and iteration two silently reused
iteration one's kernel. This produced a false *failure* (35.2% of elements wrong,
which looks exactly like a broken fix) and, worse, a false *pass* on the test meant to
pin the `sympy.Integer` path. Found on A100; it would have failed CI.

Six more of these are in [`PR.md`](PR.md) §5–6, including two reachability claims I
nearly shipped wrong.

---

## 8. Layout

```
README.md      this file
PR.md          the upstream story: four findings, three PRs, what was nearly claimed wrongly
demo.sh        fail -> explain -> fix -> verify; restores its starting state

repro/         bug.py (the minimal trigger) · repro.py (exit-coded) · ablation.py
               (9 variants) · why_it_survives.py (the dispatch mechanism) ·
               unet_like.py · verify_fix.py · check_upstream.py (run under ANY torch)
patch/         patch.py apply|revert|status  +  the same hunk for `patch -p1`
tests/         state-aware suite + a pytest shim, so torch stays the only dependency

upstream/      README (send order) · issues.md (four issue bodies + prior art) ·
               SUBMIT*.md (per-PR argument, objections, claim limits) ·
               PR{1,2,3}_BODY.md · patches/ (git am-ready) · report.md (lab notebook)
evidence/      RESULTS_a100.md (§1–§19) + 27 portable scripts + logs/, including the
               runs labelled INVALID and the A-vs-A controls
tools/         state.py — the only sanctioned patch toggle · guard_matrix.sh
origin/        the production traceback this started from (paths redacted, frames verbatim)
```

---

## 9. Environment, and what is *not* claimed

For current/upstream work in this checkout, the canonical runtime is the Conda
environment named `dynamic`:

```bash
conda run -n dynamic python -m pip install -r requirements-upstream.txt
conda run -n dynamic python repro/check_upstream.py
```

Keep the PyTorch 2.3.1 teaching artifact in a separate compatibility
environment. Installing `requirements.txt` into `dynamic` would downgrade the
upstream runtime and make its evidence incomparable. See `AGENTS.md` for the
full experiment and patch-state protocol.

| | |
|---|---|
| the 2.3.1 artifact | torch `2.3.1+cu121`, sympy `1.14.0`, python `3.11.15` |
| the upstream work | torch `2.13.0+cu130`, triton `3.7.1`, python `3.12.14` |
| hardware | 1× A100-SXM4-80GB and 1× H100 80GB HBM3, driver 575.57.08 |
| GPU required? | **No.** The 2.3.1 bug and fix reproduce on CPU in seconds |

**Verified here:** the traceback and its frames; the root cause read from source; the
standalone reproduction on CPU and CUDA; the ingredient ablation (1-D/2-D/3-D,
`nearest` and `nearest-exact`); the dispatch mechanism; the fix compiling and being
bit-exact; `fallback()` failing with `NotImplementedError`; the UNet-shaped case; the
Triton cache race and that serial compilation removes it (30/30, from ~1-in-18); that
2.4.0/2.7.0/2.11.0/2.13.0 are unaffected through ATen. For the upstream work: the
lowering is unreachable through ATen (0 hits, 8 entry points) yet **still crashes**
when reached; the three fixes are a strict no-op (63/63 identical results, same GPU
A/B); 42/42 adversarial bit-exact; 6/6 guard matrix; 2621 tests with 1 pre-existing
failure.

**Original context, not re-verified here:** that this bug surfaced as an HTTP 500 in a
production SDXL worker. That observation started the investigation; the repository does
not contain or depend on that code, and the reproduction stands on its own. The
`diffusers` excerpts in §1 are quoted from that library, which is never imported here.

**Not claimed:** that the 2.3.1 patch is what upstream should adopt — it is a local
mitigation for a pinned version. Nor that anything here is **merged, or even pushed**.
And this was a compile-time **crash**, never a miscompilation: numerics were bit-exact
throughout.

> **In one sentence:** I diagnosed a TorchInductor dynamic-shape crash on a pinned
> torch version, shipped a verified local fix, confirmed upstream had independently
> closed the reachable path at two layers while the caller-side defect stayed latent on
> `main`, and prepared three upstream fixes plus four issue reports — each with a
> regression test that fails when only its own fix is reverted.
