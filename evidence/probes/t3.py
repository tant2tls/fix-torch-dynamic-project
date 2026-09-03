import os, unittest, inspect
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch
import torch._inductor.lowering as LL
HAS_GPU=torch.cuda.is_available(); GPU_TYPE="cuda"
state = "patched" if "range()` bound" in inspect.getsource(LL.upsample_nearest2d_backward) or "there is no way to defer a" in inspect.getsource(LL.upsample_nearest2d_backward) else "unpatched"

class T(unittest.TestCase):
    def test_upsample_nearest2d_backward_symbolic_input_size(self):
        def fn(grad, ref):
            return torch.ops.aten.upsample_nearest2d_backward.default(
                grad, [grad.shape[-2], grad.shape[-1]],
                [ref.shape[0], ref.shape[1], ref.shape[-2], ref.shape[-1]])
        for device in ["cpu"] + ([GPU_TYPE] if HAS_GPU else []):
            for grad_hw, ref_hw in [((64,64),(32,32)),((63,63),(32,32)),((96,96),(32,32)),((100,70),(50,35))]:
                torch._dynamo.reset()
                grad=torch.randn(1,3,*grad_hw,device=device); ref=torch.randn(1,3,*ref_hw,device=device)
                torch._dynamo.maybe_mark_dynamic(ref,2); torch._dynamo.maybe_mark_dynamic(ref,3)
                self.assertEqual(fn(grad,ref).tolist(), torch.compile(fn,dynamic=True)(grad,ref).tolist())
print(f"lowering state: {state}")
r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(T))
nf=len(r.failures)+len(r.errors)
print("VERDICT:", ("as expected" if (nf==0 if state=="patched" else nf>0) else "NOT as expected"))
