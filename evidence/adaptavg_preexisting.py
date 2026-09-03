#!/usr/bin/env python
"""Is the adaptive_avg_pool2d dynamic-shape failure pre-existing?

    InductorError: LoweringException: TypeError: cannot determine truth value of
    Relational: (((2*s38 + 62)//s38))*(((2*s47 + 62)//s47)) > 25

Raised from `_adaptive_avg_pool2d` (lowering.py). None of the three PRs touch
that function, so it should reproduce identically with them reverted. Verify
rather than assume -- PR3 routes into avg_pool2d, so "adjacent code" is not a
safe argument on its own.
"""
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

CASES = [((64, 64), (32, 32)), ((63, 61), (7, 7)), ((100, 70), (50, 35))]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0) if DEV=='cuda' else 'cpu'}")
for dyn in (False, True):
    for hw, osz in CASES:
        torch._dynamo.reset()
        torch.manual_seed(0)
        x = torch.randn(2, 3, *hw, device=DEV)
        f = lambda t, o=osz: F.adaptive_avg_pool2d(t, o)
        try:
            got = torch.compile(f, dynamic=dyn)(x)
            ok = torch.equal(got, f(x))
            res = "bit-exact" if ok else "MISMATCH"
        except Exception as e:
            msg = str(e).replace("\n", " ")
            head = "cannot determine truth value of Relational"
            res = f"RAISED {type(e).__name__}" + (f" [{head}]" if head in msg else f" {msg[:70]}")
        print(f"  dynamic={int(dyn)}  in={hw!s:<10} out={osz!s:<10} {res}")
