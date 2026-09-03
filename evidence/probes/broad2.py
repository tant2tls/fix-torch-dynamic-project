"""Blast radius of the OpsWrapper.constant override across many lowerings."""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
DEV="cpu"
W = torch.randn(4,3,3,3)          # hoisted, so eager and compiled agree
cases = {
 "avg_pool2d": (lambda x: F.avg_pool2d(x,2), (2,8,32,32)),
 "max_pool2d": (lambda x: F.max_pool2d(x,2), (2,8,32,32)),
 "adaptive_avg": (lambda x: F.adaptive_avg_pool2d(x,(7,7)), (2,8,32,32)),
 "clamp": (lambda x: x.clamp(-0.5,0.5), (64,64)),
 "pad_constant": (lambda x: F.pad(x,(2,2,2,2),value=0.3), (2,3,16,16)),
 "masked_fill": (lambda x: x.masked_fill(x>0,-1e9), (64,64)),
 "softmax": (lambda x: x.softmax(-1), (32,128)),
 "gelu": (lambda x: F.gelu(x), (64,64)),
 "layernorm": (lambda x: F.layer_norm(x,(64,)), (32,64)),
 "where": (lambda x: torch.where(x>0,x,torch.zeros_like(x)), (64,64)),
 "div_scalar": (lambda x: x/3.0, (64,64)),
 "pow": (lambda x: x.abs()**2.5, (64,64)),
 "cumsum": (lambda x: x.cumsum(-1), (32,64)),
 "upsample_nearest_sf": (lambda x: F.interpolate(x,scale_factor=2.0,mode="nearest"), (2,3,16,16)),
 "upsample_nearest_size": (lambda x: F.interpolate(x,size=(31,33),mode="nearest"), (2,3,16,16)),
 "upsample_bilinear": (lambda x: F.interpolate(x,size=(31,33),mode="bilinear"), (2,3,16,16)),
 "hardtanh": (lambda x: F.hardtanh(x), (64,64)),
 "logsumexp": (lambda x: x.logsumexp(-1), (32,64)),
 "norm": (lambda x: x.norm(dim=-1), (32,64)),

 "tril": (lambda x: x.tril(), (64,64)),
 "addcmul": (lambda x: torch.addcmul(x,x,x,value=0.7), (64,64)),
 "addcdiv": (lambda x: torch.addcdiv(x,x,x.abs()+1,value=0.7), (64,64)),
 "full_like": (lambda x: torch.full_like(x,3.5)+x, (64,64)),
 "floor_divide": (lambda x: torch.floor_divide(x, 3.0), (64,64)),
 "add_alpha": (lambda x: torch.add(x,x,alpha=2.5), (64,64)),
}
for label, dyn in (("static",False),("dynamic",True)):
    ok=bad=0
    for name,(fn,shape) in cases.items():
        torch._dynamo.reset()
        x=torch.randn(*shape)
        if dyn: torch._dynamo.maybe_mark_dynamic(x,0)
        try:
            e=fn(x); g=torch.compile(fn,dynamic=dyn)(x)
            if torch.allclose(g,e,atol=1e-5,rtol=1e-5): ok+=1
            else: bad+=1; print(f"  {label} MISMATCH {name}")
        except Exception as ex:
            bad+=1; print(f"  {label} ERROR {name}: {type(ex).__name__}: {str(ex).splitlines()[-1][:90]}")
    print(f"{label} sweep: {ok} ok, {bad} bad, of {len(cases)}")
