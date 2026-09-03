import os, time
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, sympy
from torch._inductor.virtualized import OpsWrapper

# cost of the proposed check, per call
s = sympy.Symbol("s0", integer=True)
vals = [0.5, 1, True, 2.0, 3]
N = 200_000
def check(value):
    return isinstance(value, sympy.Expr) and not value.is_number

t0=time.perf_counter()
for _ in range(N):
    for v in vals: check(v)
t1=time.perf_counter()
per = (t1-t0)/(N*len(vals))
print(f"isinstance-based check: {per*1e9:.1f} ns per ops.constant call")
print(f"at 10,000 ops.constant calls per compile: {per*10_000*1e3:.4f} ms total")
# and the sympy.Expr path
t0=time.perf_counter()
for _ in range(N): check(32*s)
t1=time.perf_counter()
print(f"symbolic path (is_number on a Mul): {(t1-t0)/N*1e9:.1f} ns")
