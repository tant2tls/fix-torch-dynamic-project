#!/usr/bin/env python
"""Validate the one-line Issue A fix broadly.

    scale = isize / osize                       # SymInt/SymInt -> approximate
    scale = torch.sym_float(isize) / osize      # -> correctly-rounded fp32

`_compute_upsample_nearest_indices` in torch/_decomp/decompositions.py.

Checks, for both stock and fixed, across a wide grid:
  * compiled == eager, bit-exact, dynamic input
  * static compile unaffected (must stay exact)
  * CPU unaffected (it was already exact)
  * scale_factor= path unaffected
  * nearest and nearest-exact, 1-D/2-D/3-D
"""
import itertools
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch
import torch.nn.functional as F
import torch._decomp.decompositions as D

RATIOS = [
    (448, 192), (384, 363), (37, 74), (32, 64), (32, 100), (63, 31),
    (100, 70), (1000, 999), (255, 256), (17, 5), (5, 17), (8, 8),
    (127, 16), (96, 32), (64, 8), (33, 99),
]
MODES = ["nearest", "nearest-exact"]


def make(use_fix):
    def f(input, output_size, scales, exact=False):
        ind = []
        n = len(output_size)
        off = 0.5 if exact else 0.0
        for d in range(n):
            osize = output_size[d]
            isize = input.shape[-n + d]
            if scales[d] is not None and scales[d] > 0:
                scale = isize / (isize * scales[d])
            elif use_fix:
                scale = torch.sym_float(isize) / osize
            else:
                scale = isize / osize
            oi = torch.arange(osize, dtype=torch.float32, device=input.device)
            inp = ((oi + off) * scale).to(torch.int64)
            for _ in range(n - 1 - d):
                inp = inp.unsqueeze(-1)
            ind.append(inp)
        return ind

    return f


def rows(t):
    return t[0, 0, :, 0].to(torch.int64).tolist()


def sweep(label, use_fix):
    D._compute_upsample_nearest_indices = make(use_fix)
    bad = []
    n = 0
    # --- 2-D, dynamic input, size= ---
    for mode, (i, o) in itertools.product(MODES, RATIOS):
        for dev in ["cuda", "cpu"]:
            for dyn in [True, False]:
                torch._dynamo.reset()
                x = (
                    torch.arange(float(i), device=dev)
                    .view(1, 1, i, 1)
                    .expand(1, 1, i, 4)
                    .contiguous()
                )
                if dyn:
                    torch._dynamo.mark_dynamic(x, 2)
                f = lambda t: F.interpolate(t, size=(o, 4), mode=mode)
                try:
                    e, g = rows(f(x)), rows(torch.compile(f, dynamic=dyn)(x))
                except Exception as ex:
                    bad.append((mode, i, o, dev, dyn, type(ex).__name__))
                    n += 1
                    continue
                n += 1
                if e != g:
                    w = sum(1 for a, b in zip(e, g) if a != b)
                    bad.append((mode, i, o, dev, dyn, f"{w}/{o}"))
    # --- scale_factor= path (must be untouched) ---
    for mode, sf in itertools.product(MODES, [2.0, 1.5, 0.5, 3.0]):
        torch._dynamo.reset()
        x = torch.randn(2, 3, 32, 32, device="cuda")
        torch._dynamo.mark_dynamic(x, 2)
        f = lambda t: F.interpolate(t, scale_factor=sf, mode=mode)
        n += 1
        if not torch.equal(torch.compile(f, dynamic=True)(x), f(x)):
            bad.append((mode, "sf", sf, "cuda", True, "differs"))
    # --- 1-D and 3-D ---
    for mode in MODES:
        torch._dynamo.reset()
        x1 = torch.randn(2, 3, 40, device="cuda")
        torch._dynamo.mark_dynamic(x1, 2)
        f1 = lambda t: F.interpolate(t, size=(90,), mode=mode)
        n += 1
        if not torch.equal(torch.compile(f1, dynamic=True)(x1), f1(x1)):
            bad.append((mode, "1d", 90, "cuda", True, "differs"))
        torch._dynamo.reset()
        x3 = torch.randn(1, 2, 8, 10, 12, device="cuda")
        torch._dynamo.mark_dynamic(x3, 2)
        f3 = lambda t: F.interpolate(t, size=(16, 15, 25), mode=mode)
        n += 1
        if not torch.equal(torch.compile(f3, dynamic=True)(x3), f3(x3)):
            bad.append((mode, "3d", 0, "cuda", True, "differs"))
    print(f"\n=== {label} ===")
    print(f"  configs checked : {n}")
    print(f"  disagreeing     : {len(bad)}")
    for b in bad[:14]:
        print("     ", b)
    return len(bad), n


print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
b, nb = sweep("STOCK  (scale = isize / osize)", False)
a, na = sweep("FIXED  (scale = torch.sym_float(isize) / osize)", True)
print()
print(f"stock: {b}/{nb} disagree   fixed: {a}/{na} disagree")
