"""Is the SHIPPED patch byte-identical on paths that compile today? (real installed torch)

Hash the generated code for concrete-size / scale_factor= paths, with the patch applied and
reverted, and compare. This is the "strict no-op" claim in the PR body.
"""
import os, sys, hashlib, re, json
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

CASES=[]
for mode in ("nearest","nearest-exact"):
    CASES += [
        (f"{mode}|size64",   lambda x,m=mode: F.interpolate(x,size=(64,64),mode=m)),
        (f"{mode}|sf2",      lambda x,m=mode: F.interpolate(x,scale_factor=2.0,mode=m)),
        (f"{mode}|sf1.5",    lambda x,m=mode: F.interpolate(x,scale_factor=1.5,mode=m)),
        (f"{mode}|sf0.5",    lambda x,m=mode: F.interpolate(x,scale_factor=0.5,mode=m)),
        (f"{mode}|size100x70",lambda x,m=mode: F.interpolate(x,size=(100,70),mode=m)),
        (f"{mode}|size17",   lambda x,m=mode: F.interpolate(x,size=(17,17),mode=m)),
        (f"{mode}|3d_sf2",   lambda x,m=mode: F.interpolate(x[:, :, :1].expand(-1,-1,4,-1,-1).contiguous() if x.dim()==4 else x, scale_factor=2.0, mode=m)),
    ]

def norm(blob):
    blob=re.sub(r"# AOT ID: \[[^\]]*\]","# AOT ID: [N]",blob)
    blob=re.sub(r"\b(triton|cpp)_(\w*?)fused_\w+",r"\1_fused_N",blob)
    blob=re.sub(r"[0-9a-z]{2}/c[0-9a-z]{20,}","HP",blob)
    blob=re.sub(r"\bc[0-9a-z]{20,}\b","H",blob)
    blob=re.sub(r"async_compile\.triton\('[^']*'","async_compile.triton('K'",blob)
    return blob

out={}
for name, fn in CASES:
    if "3d" in name: continue
    torch._dynamo.reset()
    x=torch.randn(2,3,32,32,device="cuda")
    try:
        _,code=run_and_get_code(torch.compile(fn,dynamic=False),x)
        out[name]=hashlib.sha256(norm("\n".join(code)).encode()).hexdigest()
    except Exception as e:
        out[name]=f"ERR:{type(e).__name__}"
print(json.dumps(out))
