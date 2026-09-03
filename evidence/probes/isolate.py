import os, sys
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
name=sys.argv[1]; dyn=sys.argv[2]=="dyn"
W2=torch.randn(4,3,3,3)
cases = {
 "avg_pool2d": (lambda x: F.avg_pool2d(x,2), (2,8,32,32)),
 "adaptive_avg": (lambda x: F.adaptive_avg_pool2d(x,(7,7)), (2,8,32,32)),
 "softmax": (lambda x: x.softmax(-1), (32,128)),
 "cumsum": (lambda x: x.cumsum(-1), (32,64)),
 "upsample_bilinear": (lambda x: F.interpolate(x,size=(31,33),mode="bilinear"), (2,3,16,16)),
 "addcmul": (lambda x: torch.addcmul(x,x,x,value=0.7), (64,64)),
 "addcdiv": (lambda x: torch.addcdiv(x,x,x.abs()+1,value=0.7), (64,64)),
 "full_like": (lambda x: torch.full_like(x,3.5)+x, (64,64)),
 "floor_divide": (lambda x: torch.floor_divide(x, 3.0), (64,64)),
 "add_alpha": (lambda x: torch.add(x,x,alpha=2.5), (64,64)),
 "conv2d": (lambda x: F.conv2d(x, W2), (2,3,16,16)),
 "logsumexp": (lambda x: x.logsumexp(-1), (32,64)),
}
fn,shape=cases[name]
x=torch.randn(*shape)
if dyn: torch._dynamo.maybe_mark_dynamic(x,0)
e=fn(x); g=torch.compile(fn,dynamic=dyn)(x)
print(f"{name} {'dyn' if dyn else 'static'}: allclose={torch.allclose(g,e,atol=1e-5,rtol=1e-5)}")
