import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
import torch, sympy, unittest
from torch._inductor.virtualized import ops

class T(unittest.TestCase):
    def test_reject(self):
        s = sympy.Symbol("s0", integer=True, positive=True)
        with self.assertRaisesRegex(TypeError, "requires a concrete value"):
            ops.constant(32 * s, torch.float32)
        with self.assertRaisesRegex(TypeError, "ops.index_expr"):
            ops.constant(32 * s, torch.float32)
    def test_accept(self):
        def fn(x):
            return (torch.full_like(x,3.5) + torch.add(x,x,alpha=2.5)
                    + torch.addcmul(x,x,x,value=0.7))
        x=torch.randn(32,32)
        self.assertTrue(torch.allclose(torch.compile(fn,dynamic=True)(x), fn(x), atol=1e-6))
unittest.main(verbosity=2)
