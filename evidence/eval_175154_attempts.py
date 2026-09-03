#!/usr/bin/env python
"""Evaluate the three failed attempts at #175154 against BOTH forward and grad.

History (all closed unmerged):
  #175177  scale = isize/osize when isize==osize or osize==2*isize  -> stale-closed,
           reviewer wanted a test + noted the branch is duplicated
  #178513  torch.floor(...) instead of truncation + clamp to isize-1  -> stale-closed,
           CI NEVER RAN, zero review
  #184848  native fast-path shortcuts + 1.0/scales[d]  -> CI failed 23x; reviewer:
           "forward now matches eager, but the grad no longer does"
  ours     sym_float(isize)/osize  -> fixes our Issue A forward, BREAKS grad

So the discriminating test is: forward parity AND gradient parity, together.
#178513's two-liner is the one nobody evaluated. If it fixes #175154 without
touching gradients, it is the strongest candidate in the set -- and it is already
written, CLA-signed, and stale-closed rather than rejected.
"""
import itertools
import os

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch
import torch.nn.functional as F
import torch._decomp.decompositions as D

ORIG = D._compute_upsample_nearest_indices


def variant_stock(input, output_size, scales, exact=False):
    return ORIG(input, output_size, scales, exact=exact)


def _build(scale_fn, index_fn):
    def f(input, output_size, scales, exact=False):
        ind = []
        n = len(output_size)
        off = 0.5 if exact else 0.0
        for d in range(n):
            osize = output_size[d]
            isize = input.shape[-n + d]
            scale = scale_fn(isize, osize, scales[d])
            oi = torch.arange(osize, dtype=torch.float32, device=input.device)
            inp = index_fn(oi, off, scale, isize)
            for _ in range(n - 1 - d):
                inp = inp.unsqueeze(-1)
            ind.append(inp)
        return ind

    return f


def sc_default(i, o, s):
    return i / (i * s) if (s is not None and s > 0) else i / o


def sc_symfloat(i, o, s):
    return i / (i * s) if (s is not None and s > 0) else torch.sym_float(i) / o


def sc_175177(i, o, s):
    # the duplicated-branch version
    if (i == o) or (o == 2 * i):
        return i / o
    if s is not None and s > 0:
        return i / (i * s)
    return i / o


def ix_trunc(oi, off, scale, isize):
    return ((oi + off) * scale).to(torch.int64)


def ix_178513(oi, off, scale, isize):
    v = torch.floor((oi + off) * scale).to(torch.int64)
    return torch.min(v, torch.as_tensor(isize - 1, device=v.device, dtype=v.dtype))


VARIANTS = [
    ("stock", variant_stock),
    ("#175177 branch", _build(sc_175177, ix_trunc)),
    ("#178513 floor+clamp", _build(sc_default, ix_178513)),
    ("ours sym_float", _build(sc_symfloat, ix_trunc)),
    ("#178513 + sym_float", _build(sc_symfloat, ix_178513)),
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def check_175154():
    """The filed repro: scale_factor=1.3 on a 1x1x1x2 float64 tensor."""
    torch._dynamo.reset()
    x = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
    f = lambda t: F.interpolate(t, scale_factor=1.3, mode="nearest")
    return torch.equal(torch.compile(f)(x), f(x))


def check_issueA():
    """Our finding: dynamic input, size=, CUDA, ULP-sensitive ratio."""
    if DEV != "cuda":
        return None
    torch._dynamo.reset()
    i, o = 448, 192
    x = (
        torch.arange(float(i), device=DEV)
        .view(1, 1, i, 1)
        .expand(1, 1, i, 4)
        .contiguous()
    )
    torch._dynamo.maybe_mark_dynamic(x, 2)
    f = lambda t: F.interpolate(t, size=(o, 4), mode="nearest")
    try:
        e = f(x)[0, 0, :, 0].to(torch.int64).tolist()
        g = torch.compile(f, dynamic=True)(x)[0, 0, :, 0].to(torch.int64).tolist()
    except Exception as ex:
        return f"ERR:{type(ex).__name__}"
    return sum(1 for a, b in zip(e, g) if a != b)


def check_grads():
    """The question that killed #184848."""
    ok = tot = 0
    for mode, dyn, spec, hw in itertools.product(
        ["nearest", "nearest-exact"],
        [False, True],
        ["size", "sf"],
        [((32, 32), (64, 64)), ((33, 31), (99, 62)), ((64, 64), (32, 32)),
         ((448, 448), (192, 192))],
    ):
        in_hw, out_hw = hw
        if spec == "sf":
            sf = out_hw[0] / in_hw[0]
            if out_hw[1] / in_hw[1] != sf:
                continue
        torch._dynamo.reset()
        torch.manual_seed(0)
        x = torch.randn(2, 3, *in_hw, device=DEV, requires_grad=True)
        if spec == "size":
            f = lambda t, o=out_hw, m=mode: F.interpolate(t, size=o, mode=m).sum()
        else:
            f = lambda t, s=sf, m=mode: F.interpolate(t, scale_factor=s, mode=m).sum()
        try:
            torch.compile(f, dynamic=dyn)(x).backward()
            g = x.grad.clone()
            x.grad = None
            f(x).backward()
            tot += 1
            ok += torch.equal(g, x.grad)
        except Exception:
            tot += 1
    return ok, tot


def check_noop_static():
    """Existing static callers must not change."""
    bad = 0
    n = 0
    for mode, (i, o) in itertools.product(
        ["nearest", "nearest-exact"],
        [(32, 64), (32, 100), (63, 31), (100, 70), (448, 192), (1, 8), (8, 1)],
    ):
        torch._dynamo.reset()
        x = (
            torch.arange(float(i), device=DEV)
            .view(1, 1, i, 1)
            .expand(1, 1, i, 4)
            .contiguous()
        )
        f = lambda t, o=o, m=mode: F.interpolate(t, size=(o, 4), mode=m)
        n += 1
        if not torch.equal(torch.compile(f)(x), f(x)):
            bad += 1
    return bad, n


print(f"torch {torch.__version__}  device={DEV}\n")
hdr = f"{'variant':<22}{'#175154':<10}{'IssueA':<10}{'grads':<10}{'static no-op':<14}"
print(hdr)
print("-" * len(hdr))
for name, fn in VARIANTS:
    D._compute_upsample_nearest_indices = fn
    a = "FIXED" if check_175154() else "wrong"
    b = check_issueA()
    bs = "n/a" if b is None else ("exact" if b == 0 else (b if isinstance(b, str) else f"{b} wrong"))
    ok, tot = check_grads()
    nb, nn = check_noop_static()
    print(f"{name:<22}{a:<10}{bs:<10}{f'{ok}/{tot}':<10}{f'{nn - nb}/{nn} same':<14}")
D._compute_upsample_nearest_indices = ORIG
print()
print("A candidate must be: #175154 FIXED, IssueA exact, grads at the stock count,")
print("and static no-op unchanged. Anything that lowers the grad count repeats")
print("#184848's failure.")
