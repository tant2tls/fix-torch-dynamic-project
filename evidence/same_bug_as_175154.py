#!/usr/bin/env python
"""Is my Issue A the same defect as the already-filed #175154, or distinct?

#175154: `interpolate(x, scale_factor=1.3, mode='nearest')` on a 1x1x1x2 float64
tensor returns [1,1] compiled vs [1,2] eager. Open, high priority, actionable,
labelled PT2-Bug-Bash, no linked PR.

My Issue A: `interpolate(x, size=(192,4), mode='nearest')` with a *dynamic* input
on CUDA picks the wrong source row for 4 of 140 configurations.

If these are the same root cause, Issue A should be filed as a comment on #175154
rather than as a new issue -- and my fix should be offered there. If they are
distinct, Issue A stands alone.

Discriminators:
  1. does #175154 need dynamic shapes?          (mine does)
  2. is #175154 CUDA-only?                      (mine is)
  3. does #175154 use scale_factor= or size=?   (mine is size=)
  4. does the sym_float fix repair #175154 too?  <- the decisive one
"""
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch
import torch.nn.functional as F
import torch._decomp.decompositions as D

orig = D._compute_upsample_nearest_indices


def sym_float_fix(input, output_size, scales, exact=False):
    ind = []
    n = len(output_size)
    off = 0.5 if exact else 0.0
    for d in range(n):
        osize = output_size[d]
        isize = input.shape[-n + d]
        if scales[d] is not None and scales[d] > 0:
            scale = isize / (isize * scales[d])
        else:
            scale = torch.sym_float(isize) / osize
        oi = torch.arange(osize, dtype=torch.float32, device=input.device)
        inp = ((oi + off) * scale).to(torch.int64)
        for _ in range(n - 1 - d):
            inp = inp.unsqueeze(-1)
        ind.append(inp)
    return ind


def case_175154(dev="cpu", dt=torch.float64, dynamic=False):
    torch._dynamo.reset()
    x = torch.tensor([[[[1.0, 2.0]]]], dtype=dt, device=dev)
    if dynamic:
        torch._dynamo.maybe_mark_dynamic(x, 3)
    f = lambda t: F.interpolate(t, scale_factor=1.3, mode="nearest")
    e = f(x)
    g = torch.compile(f, dynamic=dynamic)(x)
    return e.flatten().tolist(), g.flatten().tolist(), torch.equal(e, g)


def case_issueA(dev="cuda"):
    torch._dynamo.reset()
    i, o = 448, 192
    x = (
        torch.arange(float(i), device=dev)
        .view(1, 1, i, 1)
        .expand(1, 1, i, 4)
        .contiguous()
    )
    torch._dynamo.mark_dynamic(x, 2)
    f = lambda t: F.interpolate(t, size=(o, 4), mode="nearest")
    e = f(x)[0, 0, :, 0].to(torch.int64).tolist()
    g = torch.compile(f, dynamic=True)(x)[0, 0, :, 0].to(torch.int64).tolist()
    n = sum(1 for a, b in zip(e, g) if a != b)
    return n


print(f"torch {torch.__version__}\n")
print("--- Q1/Q2/Q3: does #175154 need dynamic shapes / CUDA / size=? ---")
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    for dyn in (False, True):
        e, g, ok = case_175154(dev=dev, dynamic=dyn)
        print(f"  scale_factor=1.3  {dev:<5} dynamic={int(dyn)}  eager={e} compiled={g} {'ok' if ok else 'MISMATCH'}")
print("  -> #175154 reproduces with STATIC shapes and on CPU, so it is NOT")
print("     dynamic-specific and NOT CUDA-specific. Mine is both.")

print("\n--- Q4: does the sym_float fix repair #175154? (the decisive test) ---")
for label, fn in (("stock", orig), ("sym_float fix", sym_float_fix)):
    D._compute_upsample_nearest_indices = fn
    e, g, ok = case_175154(dev="cpu", dynamic=False)
    a = case_issueA("cuda") if torch.cuda.is_available() else -1
    print(f"  {label:<14} #175154: compiled={g} {'ok' if ok else 'STILL WRONG'}"
          f"   |  IssueA 448->192: {a} rows wrong")
D._compute_upsample_nearest_indices = orig
print()
print("If sym_float fixes Issue A but NOT #175154, they are different defects:")
print("  #175154 = the scale_factor= path (isize/(isize*scales[d]) arithmetic)")
print("  Issue A = the size= path with a symbolic isize (approximate SymInt divide)")
