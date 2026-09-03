"""THE open verification: does ops.div_rn emit tl.div_rn on Triton, and is it bit-exact?"""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
from torch._inductor.lowering import register_lowering, upsample_nearestnd, lowerings
from torch._inductor.utils import run_and_get_code
print("torch", torch.__version__, "| gpu:", torch.cuda.get_device_name(0))

def build(exact=False):
    lib = torch.library._scoped_library("tv","FRAGMENT"); l=lib.__enter__()
    l.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
    mode = "nearest-exact" if exact else "nearest"
    ref = lambda x,h,w: F.interpolate(x,size=(int(h),int(w)),mode=mode)
    l.impl("ups2d", lambda x,h,w: x.new_empty((x.shape[0],x.shape[1],h,w)), "Meta")
    l.impl("ups2d", ref, "CPU"); l.impl("ups2d", ref, "CUDA")
    register_lowering(torch.ops.tv.ups2d)(
        lambda x,h,w: upsample_nearestnd(x,[h,w],(None,None),n=2,exact=exact))
    return lib, ref

# --- 1. generated Triton source contains div_rn ---
lib, ref = build()
torch._dynamo.reset()
x=torch.randn(1,3,32,32,device="cuda"); rt=torch.randn(1,3,64,64,device="cuda")
torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
_, code = run_and_get_code(torch.compile(lambda a,s: torch.ops.tv.ups2d(a,s[0],s[1]), dynamic=True), x, rt.shape[-2:])
blob="\n".join(code)
print("1. generated Triton contains 'div_rn':", "div_rn" in blob)
for line in blob.splitlines():
    if "div_rn" in line: print("     ", line.strip()[:120])
lib.__exit__(None,None,None)
for k in [k for k in list(lowerings) if "tv" in str(k)]: lowerings.pop(k,None)

# --- 2. bit-exact on the ULP-sensitive pairs, on GPU ---
PAIRS=[((32,32),(64,64)),((32,32),(100,70)),((32,32),(17,17)),((448,448),(192,192)),
       ((384,384),(363,363)),((256,256),(82,82)),((10,10),(100,100)),((32,32),(65,33)),
       ((1,1),(9,9)),((13,13),(40,39)),((3,4),(97,53)),((7,5),(23,11)),((64,64),(21,85)),
       ((512,512),(300,300)),((224,224),(97,97))]
ok=wrong=crash=0; bad=[]
for exact in (False,True):
    for in_hw,out_hw in PAIRS:
        lib, ref = build(exact)
        torch._dynamo.reset()
        x=torch.randn(1,3,*in_hw,device="cuda"); rt=torch.randn(1,3,*out_hw,device="cuda")
        torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
        exp=ref(x,*out_hw)
        try:
            got=torch.compile(lambda a,s: torch.ops.tv.ups2d(a,s[0],s[1]),dynamic=True)(x,rt.shape[-2:])
            if torch.equal(got,exp): ok+=1
            else:
                wrong+=1; bad.append(f"{in_hw}->{out_hw}{'/exact' if exact else ''}: {int((got!=exp).sum())}/{exp.numel()}")
        except Exception as e:
            crash+=1; bad.append(f"{in_hw}->{out_hw}: {type(e).__name__}")
        lib.__exit__(None,None,None)
        for k in [k for k in list(lowerings) if "tv" in str(k)]: lowerings.pop(k,None)
print(f"2. GPU bit-exact vs eager: {ok}/{2*len(PAIRS)}  wrong={wrong} crashed={crash}")
for b in bad[:5]: print("     !", b)
