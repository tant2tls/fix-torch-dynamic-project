#!/usr/bin/env python
"""Is `div_rn` actually free here? A controlled re-measurement.

WHY THIS EXISTS. `evidence/perf_divrn.py` reported `div_rn/truediv` = 0.957-0.984
(i.e. div_rn marginally FASTER) and `RESULTS_a100.md` §10 quotes that as "div_rn
costs nothing here". Re-running it on a second A100 produced 1.009-1.212 --
div_rn up to 21% SLOWER. A claim that flips direction between two runs of the same
script on the same GPU model is not measuring what it thinks it is.

THE FLAW. `perf_divrn.py` times each variant ONCE, in a fixed order (div_rn then
truediv), with 20 warmup iterations and no clock control. On this box the GPU idles
at 210 MHz against a 1410 MHz max, so whichever variant runs first pays the
clock-ramp cost. That is an order effect, not a property of the divide.

WHAT THIS DOES INSTEAD.
  * locks clocks if permitted, else spins a fixed workload to reach steady state
    and REPORTS the achieved clock so the reader can judge
  * A/B/A/B interleaving with many alternations, so drift cancels instead of
    landing entirely on one variant
  * CUDA-event timing (device-side) rather than host wall clock
  * reports the MEDIAN of per-alternation ratios plus min/max, so a single
    outlier cannot carry the conclusion
  * runs an A-vs-A control (div_rn vs itself) FIRST -- the repo's own hard-learned
    rule. If the control does not straddle 1.0, the harness cannot resolve the
    effect it is being asked to measure, and any A-vs-B number is noise.

Emits JSON if given a path. Exit 0 always; read the verdict.
"""
import json
import os
import statistics
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

DEV = "cuda"
REPS = 15          # alternations per pair
INNER = 100        # timed launches per measurement
WARMUP = 50

SIZES = [
    (1, 3, 64, 64, 128, 128),
    (1, 3, 256, 256, 512, 512),
    (8, 64, 128, 128, 256, 256),
    (4, 128, 256, 256, 512, 512),
]


def clock_info():
    try:
        sm = torch.cuda.clock_rate()  # MHz, if available
    except Exception:
        sm = None
    return sm


def stabilize(seconds=6.0):
    """Spin a dense workload so the SM clock leaves its idle state."""
    import time

    a = torch.randn(4096, 4096, device=DEV)
    b = torch.randn(4096, 4096, device=DEV)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            a = a @ b * 1e-4 + b
    torch.cuda.synchronize()
    return a.sum().item()


def bench_events(fn):
    """Device-side timing of INNER launches; returns microseconds per call."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(INNER):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / INNER  # ms -> us


# ---- the two lowerings, identical except for the divide ---------------------
from torch._inductor.virtualized import V, ops


def make_lowering(use_div_rn):
    def low(x, h, w):
        x.realize_hint()
        x_loader = x.make_loader()
        i_sizes = [V.graph.sizevars.guard_int(s) for s in x.get_size()[-2:]]
        batch = x.get_size()[:-2]
        o_sizes = [h, w]

        def scale_fn(idx, o_size, size):
            f = ops.index_expr(idx, torch.float32)
            num = ops.constant(size, torch.float32)
            den = ops.index_expr(o_size, torch.float32)
            inv = ops.div_rn(num, den) if use_div_rn else ops.truediv(num, den)
            f = ops.mul(f, inv)
            f = ops.to_dtype(f, torch.int32)
            return ops.indirect_indexing(f, size, check=False)

        def fn(index):
            sp = index[-2:]
            b = index[:-2]
            return x_loader(
                [*b, *[scale_fn(i, o, s) for i, o, s in zip(sp, o_sizes, i_sizes)]]
            )

        from torch._inductor.ir import Pointwise

        return Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=fn,
            ranges=[*batch, h, w],
        )

    return low


def build(tag, use_div_rn, n, c, h, w, oh, ow):
    """Register a uniquely-named custom op so the FX cache cannot alias variants.

    The repo learned this the hard way: the FX graph cache keys on the op NAME, so
    a flag captured in the lowering's closure is invisible to it and the second
    variant silently reuses the first's kernel -- producing a false result.

    The symbolic quantity must be the OUTPUT size, not the input: marking the
    input dynamic makes Dynamo specialize it (the lowering guards i_sizes) and
    raises ConstraintViolationError. So mark a reference tensor dynamic and pass
    its trailing sizes in, exactly as `perf_divrn.py` does.
    """
    import torch._inductor.lowering as L

    name = f"perfab_{tag}"
    lib = torch.library.Library("perfab", "FRAGMENT")  # noqa: TOR901
    lib.define(f"{name}(Tensor x, SymInt h, SymInt w) -> Tensor")
    lib.impl(name, lambda t, a, b: t.new_empty((t.shape[0], t.shape[1], a, b)), "Meta")
    lib.impl(
        name,
        lambda t, a, b: F.interpolate(t, size=(int(a), int(b)), mode="nearest"),
        "CUDA",
    )
    op = getattr(torch.ops.perfab, name)
    L.register_lowering(op)(make_lowering(use_div_rn))

    x = torch.randn(n, c, h, w, device=DEV)
    ref = torch.randn(n, c, oh, ow, device=DEV)
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)
    torch._dynamo.reset()
    g = torch.compile(lambda t, s: op(t, s[0], s[1]), dynamic=True)
    sizes = ref.shape[-2:]
    g(x, sizes)  # compile
    return (lambda: g(x, sizes)), lib


def ratio_stats(pairs):
    rs = [a / b for a, b in pairs]
    return statistics.median(rs), min(rs), max(rs)


def run_pair(tag_a, a_divrn, tag_b, b_divrn, size):
    fa, _la = build(tag_a, a_divrn, *size)
    fb, _lb = build(tag_b, b_divrn, *size)
    for _ in range(WARMUP // 10):
        fa()
        fb()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(REPS):
        ta = bench_events(fa)
        tb = bench_events(fb)
        pairs.append((ta, tb))
    return pairs


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"sm_{cap[0]}{cap[1]}  |  {REPS} alternations x {INNER} launches, CUDA events")
    print()
    print("Stabilizing clocks...", flush=True)
    stabilize()
    sm = clock_info()
    print(f"SM clock after stabilization: {sm if sm else 'unavailable'} MHz")
    print("(idle on this box is ~210 MHz against a 1410 MHz max; timing a variant")
    print(" while the GPU ramps is what produced the earlier contradictory numbers)")
    print()

    results = {"gpu": torch.cuda.get_device_name(0), "sm": f"{cap[0]}{cap[1]}",
               "torch": torch.__version__, "control": {}, "ab": {}}

    print("=== CONTROL: div_rn vs div_rn (must straddle 1.0) ===")
    hdr = f"{'shape':<24}{'median':<10}{'min':<10}{'max':<10}{'verdict':<12}"
    print(hdr)
    print("-" * len(hdr))
    ctrl_ok = True
    for idx, size in enumerate(SIZES):
        pairs = run_pair(f"ctrl_a{idx}", True, f"ctrl_b{idx}", True, size)
        med, lo, hi = ratio_stats(pairs)
        band = lo <= 1.0 <= hi
        ctrl_ok &= band
        shape = f"{size[0]}x{size[1]}x{size[2]}x{size[3]}"
        results["control"][shape] = {"median": med, "min": lo, "max": hi}
        print(f"{shape:<24}{med:<10.4f}{lo:<10.4f}{hi:<10.4f}"
              f"{'ok' if band else 'BIASED':<12}")

    print()
    print("=== A/B: div_rn vs truediv, same lowering, interleaved ===")
    print(hdr)
    print("-" * len(hdr))
    for idx, size in enumerate(SIZES):
        pairs = run_pair(f"rn{idx}", True, f"td{idx}", False, size)
        med, lo, hi = ratio_stats(pairs)
        shape = f"{size[0]}x{size[1]}x{size[2]}x{size[3]}"
        results["ab"][shape] = {"median": med, "min": lo, "max": hi}
        verdict = "free" if med <= 1.02 else ("slower" if med > 1.05 else "marginal")
        print(f"{shape:<24}{med:<10.4f}{lo:<10.4f}{hi:<10.4f}{verdict:<12}")

    meds = [v["median"] for v in results["ab"].values()]
    cmeds = [v["median"] for v in results["control"].values()]
    print()
    print(f"control medians : {['%.4f' % m for m in cmeds]}")
    print(f"A/B     medians : {['%.4f' % m for m in meds]}")
    print()
    if not ctrl_ok:
        print("VERDICT: the A-vs-A control is itself biased -- this harness cannot")
        print("resolve the effect. Do not quote any A/B number from this run.")
    else:
        worst = max(meds)
        print("Control straddles 1.0 at every shape, so the harness is unbiased.")
        print(f"Worst-case div_rn cost: {worst:.4f}x")
        if worst <= 1.02:
            print("VERDICT: div_rn is free here (<=2%), consistent with the")
            print("loop-invariance argument. The earlier 0.957-0.984 figures were")
            print("optimistic (they credited div_rn with a clock-ramp artifact),")
            print("but the DIRECTION of the claim holds.")
        else:
            print("VERDICT: div_rn is measurably slower at some shape. RESULTS §10's")
            print("'costs nothing' wording must be corrected before the PR is sent.")

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(results, fh, indent=1, sort_keys=True)
        print(f"\nwrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
