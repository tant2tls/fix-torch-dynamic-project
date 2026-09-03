#!/usr/bin/env python
"""How wide is PR2's blast radius, really?

`OpsWrapper.constant` governs ~111 static call sites. A reviewer's first question
is: how many of those can actually reach the new raise, and what does the check
cost on the hot path?

Two things are measured here, because both are load-bearing and neither is
answerable by reading the diff:

  A. COVERAGE -- instrument `OpsWrapper.constant` and count, over a broad
     compile workload, how many distinct call sites are hit, how many values are
     sympy at all, and how many are sympy-but-concrete (the `is_number` branch
     that must keep working). If sympy values never appear in practice, the
     `isinstance` is nearly free; if sympy-concrete values are common, the
     `is_number` test is the load-bearing half of the predicate, not the
     `isinstance`.

  B. COST -- `ops.constant` is called during lowering, once per IR node, not per
     element. Time a lowering-heavy compile with and without the check to show
     the added `isinstance` does not register.
"""

import os
import tempfile
import time
from collections import Counter

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# ⚠️ THIS PROBE MUST RUN AGAINST A COLD CACHE, and it has to arrange that itself.
#
# `ops.constant` is only called while a graph is being LOWERED. If inductor's FX
# graph cache is warm, compilation is served from cache, no lowering happens, and
# the instrumentation below counts ZERO calls at ZERO call sites -- which reads as
# "the check is never reached" instead of "nothing was measured". That is a silent
# false negative, and it is easy to hit here because the default
# TORCHINDUCTOR_CACHE_DIR on this box points into NFS $HOME, a cache shared with
# every other container in the fleet and warm from earlier sessions.
#
# Observed on A100 2026-08-28: warm cache -> 0 calls / 0 sites; cold cache ->
# 1221 calls / 5 sites / 13 sympy-concrete. Force a private cold dir and disable
# the graph caches, before importing torch.
_cold = tempfile.mkdtemp(prefix="pr2_blast_cold_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cold
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"

import sympy
import torch
import torch.nn.functional as F
from torch._inductor import virtualized
from torch._inductor.virtualized import OpsWrapper

# ---------------------------------------------------------------- A. coverage
stats = Counter()
sites = Counter()
real_constant = OpsWrapper.constant


def counting_constant(self, value, dtype):
    import traceback

    stats["calls"] += 1
    v = OpsWrapper._unwrap(value)
    if isinstance(v, sympy.Expr):
        stats["sympy"] += 1
        stats["sympy_concrete" if v.is_number else "sympy_symbolic"] += 1
    else:
        stats["plain"] += 1
    # attribute to the caller inside _inductor
    for fr in reversed(traceback.extract_stack()[:-1]):
        if "_inductor" in fr.filename and "virtualized.py" not in fr.filename:
            sites[f"{os.path.basename(fr.filename)}:{fr.lineno}"] += 1
            break
    return real_constant(self, value, dtype)


OpsWrapper.constant = counting_constant

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def workload():
    """Deliberately broad: pointwise, reductions, norms, pooling, upsample,
    matmul, indexing, scalar folding -- the ops most likely to build constants."""
    out = []

    def run(f, *a, dynamic=False):
        torch._dynamo.reset()
        out.append(torch.compile(f, dynamic=dynamic)(*a))

    x = torch.randn(8, 16, 32, 32, device=DEV)
    run(lambda t: torch.full_like(t, 3.5) + t * 2.5, x)
    run(lambda t: torch.addcmul(t, t, t, value=0.7), x)
    run(lambda t: torch.addcdiv(t, t, t + 1, value=1.3), x)
    run(lambda t: F.gelu(t) + F.silu(t) + F.hardswish(t), x)
    run(lambda t: F.layer_norm(t, t.shape[-1:]), x)
    run(lambda t: F.softmax(t, -1) + F.log_softmax(t, -1), x)
    run(lambda t: t.mean(-1) + t.var(-1) + t.amax(-1), x)
    run(lambda t: F.avg_pool2d(t, 2) + F.max_pool2d(t, 2), x)
    run(lambda t: F.adaptive_avg_pool2d(t, (8, 8)), x)
    run(lambda t: F.interpolate(t, size=(64, 64), mode="nearest"), x)
    run(lambda t: F.interpolate(t, scale_factor=2.0, mode="nearest"), x)
    run(lambda t: F.interpolate(t, size=(64, 64), mode="bilinear"), x)
    run(lambda t: torch.where(t > 0, t, t * 0.01), x)
    run(lambda t: t.clamp(-1, 1) + t.round() + t.sign(), x)
    run(lambda t: torch.cat([t, t], 1)[:, :16], x)
    run(lambda t: t.cumsum(-1) + t.flip(-1), x)
    run(lambda t: (t.to(torch.float16) * 2).to(torch.float32), x)
    # dynamic shapes: where symbolic values actually flow
    run(lambda t: F.interpolate(t, size=(64, 64), mode="nearest"), x, dynamic=True)
    run(lambda t: F.layer_norm(t, t.shape[-1:]), x, dynamic=True)
    run(lambda t: t.mean(-1) + t.var(-1), x, dynamic=True)
    run(lambda t: F.avg_pool2d(t, 2), x, dynamic=True)
    m = torch.randn(128, 128, device=DEV)
    run(lambda a: (a @ a).relu(), m)
    run(lambda a: (a @ a).relu(), m, dynamic=True)
    return out


t0 = time.perf_counter()
workload()
t_instrumented = time.perf_counter() - t0

print(f"torch {torch.__version__}  device={DEV}")
print(f"cold inductor cache: {_cold}")
print()
print("=== A. what actually reaches ops.constant ===")
tot = stats["calls"]
if tot == 0:
    # Do not let a measurement failure masquerade as the finding "the check is
    # never reached". Zero calls means no graph was lowered -- almost always a
    # warm cache -- not that ops.constant is cold code.
    raise SystemExit(
        "ABORT: 0 ops.constant calls recorded, so NOTHING was measured.\n"
        "A graph must be lowered for ops.constant to run; zero means compilation\n"
        "was served from cache. Expected >1000 calls on a cold cache.\n"
        f"Cache dir used: {_cold}\n"
        "Check that TORCHINDUCTOR_CACHE_DIR/FX_GRAPH_CACHE were not re-set by the\n"
        "environment after this script forced them."
    )
print(f"  total ops.constant calls in this workload : {tot}")
print(f"  distinct inductor call sites hit          : {len(sites)}  (of ~111 static)")
for k in ("plain", "sympy", "sympy_concrete", "sympy_symbolic"):
    v = stats[k]
    print(f"  {k:<41}: {v:>6}  ({100.0 * v / max(tot, 1):.2f}%)")
print()
print("  top call sites:")
for s, c in sites.most_common(8):
    print(f"    {s:<34} {c}")
print()
print("  -> `sympy_concrete` is the population that the `is_number` half of the")
print("     predicate protects. If it is non-zero, an `isinstance`-only check")
print("     would reject working code, which is why both halves are needed.")

# ------------------------------------------------------------------- B. cost
OpsWrapper.constant = real_constant


def timed(label, n=3):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        workload()
        best = min(best, time.perf_counter() - t0)
    print(f"  {label:<44} {best:.2f} s")
    return best


# emulate the pre-PR path: no override, straight to _default
def bare_constant(self, value, dtype):
    return self._default("constant", (OpsWrapper._unwrap(value), dtype), {})


print()
print("=== B. compile-time cost of the check (lowering-time, not per-element) ===")
# ⚠️ INTERLEAVE, and report the SPREAD, not one ratio.
#
# Measuring with-then-without once each gives a number that is mostly ordering
# and cache-warmth artifact. Three identical runs of the earlier version of this
# script produced ratios 0.9864 / 1.0431 / 1.1246 (A100, 2026-08-28) -- straddling
# 1.0, i.e. the true effect is below this harness's resolution. Quoting any single
# one of those as "the cost" (the repo previously quoted 0.98x) overstates the
# precision available. Alternate the two variants and report min/median/max.
ratios = []
for rep in range(3):
    OpsWrapper.constant = real_constant
    w = timed(f"[rep {rep + 1}] with check", n=1)
    OpsWrapper.constant = bare_constant
    wo = timed(f"[rep {rep + 1}] without check", n=1)
    ratios.append(w / wo)
OpsWrapper.constant = real_constant

ratios.sort()
lo, mid, hi = ratios[0], ratios[len(ratios) // 2], ratios[-1]
print()
print(f"  ratio with/without: median {mid:.4f}  (min {lo:.4f}, max {hi:.4f})")
if lo <= 1.0 <= hi:
    print("  -> the spread STRADDLES 1.0: no measurable compile-time cost.")
    print("     Report it that way; do not quote a point ratio.")
else:
    print("  -> the spread excludes 1.0; the direction is resolvable here.")
print("  ops.constant runs once per IR node during lowering, not once per")
print("  element at runtime, so this is compile time and does not touch the")
print("  generated kernel at all.")
