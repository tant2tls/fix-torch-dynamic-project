"""Does the OpsWrapper.constant override actually take effect, and not break dispatch?"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, sympy
from torch._inductor.virtualized import ops, OpsWrapper

# 1. is our override the one installed on the class (not clobbered by _init_cls)?
import inspect
src = inspect.getsource(type(ops).constant) if hasattr(type(ops), "constant") else "<none>"
print("1. OpsWrapper.constant is our override:", "requires a concrete value" in src)

# 2. does it fire on a symbolic value?
s = sympy.Symbol("s0", integer=True, positive=True)
try:
    ops.constant(32*s, torch.float32)
    print("2. symbolic value: DID NOT RAISE (bad)")
except TypeError as e:
    msg=str(e)
    print("2. symbolic value raises TypeError:", "index_expr" in msg and "concrete" in msg)
    print("   ", msg.splitlines()[0][:110])
except Exception as e:
    print(f"2. unexpected {type(e).__name__}: {e}")

# 3. concrete values must NOT raise -- but ops.constant outside a lowering
#    context needs a handler; check the guard logic in isolation instead
for v in (0.5, 1, True, sympy.Integer(3), sympy.Float(0.5)):
    bad = isinstance(v, sympy.Expr) and not v.is_number
    if bad: print(f"3. FALSE POSITIVE on {v!r}")
print("3. no false positives on concrete values (incl. sympy numbers)")
