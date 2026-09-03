"""Reviewer objections, tested rather than argued."""
import os
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE","0")
import torch, torch.nn.functional as F
from torch._inductor.lowering import register_lowering, upsample_nearestnd, lowerings
from torch._inductor import config

def harness(exact=False):
    lib=torch.library._scoped_library("st","FRAGMENT"); l=lib.__enter__()
    l.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
    mode="nearest-exact" if exact else "nearest"
    ref=lambda x,h,w: F.interpolate(x,size=(int(h),int(w)),mode=mode)
    l.impl("ups2d", lambda x,h,w: x.new_empty((x.shape[0],x.shape[1],h,w)),"Meta")
    l.impl("ups2d", ref,"CPU"); l.impl("ups2d", ref,"CUDA")
    register_lowering(torch.ops.st.ups2d)(
        lambda x,h,w: upsample_nearestnd(x,[h,w],(None,None),n=2,exact=exact))
    return lib, ref
def cleanup(lib):
    lib.__exit__(None,None,None)
    for k in [k for k in list(lowerings) if "st" in str(k)]: lowerings.pop(k,None)

def one(label, out_hw=(192,192), in_hw=(448,448), **cfg):
    lib, ref = harness()
    try:
        torch._dynamo.reset()
        x=torch.randn(1,3,*in_hw,device="cuda"); rt=torch.randn(1,3,*out_hw,device="cuda")
        torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
        exp=ref(x,*out_hw)
        f=lambda a,s: torch.ops.st.ups2d(a,s[0],s[1])
        if cfg:
            with config.patch(**cfg):
                got=torch.compile(f,dynamic=True)(x,rt.shape[-2:])
        else:
            got=torch.compile(f,dynamic=True)(x,rt.shape[-2:])
        print(f"  {label}: bit-exact={torch.equal(got,exp)}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[0][:85]}")
    finally:
        cleanup(lib)

print("1. does the fix survive non-default inductor configs?")
one("default")
one("cpp_wrapper=True", cpp_wrapper=True)
one("triton.cudagraphs=True", **{"triton.cudagraphs":True})
one("max_autotune=True", max_autotune=True)
one("eager_numerics.division_rounding=True", **{"eager_numerics.division_rounding":True})
one("eager_numerics.division_rounding=False", **{"eager_numerics.division_rounding":False})
one("fallback_random=True", fallback_random=True)

print("2. dtypes other than float32")
for dt in (torch.float16, torch.bfloat16, torch.float64, torch.int32):
    lib, ref = harness()
    try:
        torch._dynamo.reset()
        x=torch.randn(1,3,448,448,device="cuda").to(dt) if dt.is_floating_point else torch.randint(0,100,(1,3,448,448),device="cuda",dtype=dt)
        rt=torch.randn(1,3,192,192,device="cuda")
        torch._dynamo.maybe_mark_dynamic(rt,2); torch._dynamo.maybe_mark_dynamic(rt,3)
        exp=ref(x,192,192)
        got=torch.compile(lambda a,s: torch.ops.st.ups2d(a,s[0],s[1]),dynamic=True)(x,rt.shape[-2:])
        print(f"  {dt}: bit-exact={torch.equal(got,exp)}")
    except Exception as e:
        print(f"  {dt}: {type(e).__name__}: {str(e).splitlines()[0][:70]}")
    finally: cleanup(lib)

print("3. 1-D and 3-D nearest (n=1, n=3), symbolic")
for n in (1,3):
    lib=torch.library._scoped_library("st3","FRAGMENT"); l=lib.__enter__()
    if n==1:
        l.define("u(Tensor x, SymInt a) -> Tensor")
        ref=lambda x,a: F.interpolate(x,size=(int(a),),mode="nearest")
        l.impl("u", lambda x,a: x.new_empty((x.shape[0],x.shape[1],a)),"Meta")
    else:
        l.define("u(Tensor x, SymInt a, SymInt b, SymInt c) -> Tensor")
        ref=lambda x,a,b,c: F.interpolate(x,size=(int(a),int(b),int(c)),mode="nearest")
        l.impl("u", lambda x,a,b,c: x.new_empty((x.shape[0],x.shape[1],a,b,c)),"Meta")
    l.impl("u", ref,"CUDA"); l.impl("u", ref,"CPU")
    if n==1:
        register_lowering(torch.ops.st3.u)(lambda x,a: upsample_nearestnd(x,[a],(None,),n=1))
    else:
        register_lowering(torch.ops.st3.u)(lambda x,a,b,c: upsample_nearestnd(x,[a,b,c],(None,None,None),n=3))
    try:
        torch._dynamo.reset()
        if n==1:
            x=torch.randn(1,3,448,device="cuda"); rt=torch.randn(1,3,192,device="cuda")
            torch._dynamo.maybe_mark_dynamic(rt,2)
            exp=ref(x,192); got=torch.compile(lambda a,s: torch.ops.st3.u(a,s[0]),dynamic=True)(x,rt.shape[-1:])
        else:
            x=torch.randn(1,3,32,64,64,device="cuda"); rt=torch.randn(1,3,17,192,192,device="cuda")
            for d in (2,3,4): torch._dynamo.maybe_mark_dynamic(rt,d)
            exp=ref(x,17,192,192); got=torch.compile(lambda a,s: torch.ops.st3.u(a,s[0],s[1],s[2]),dynamic=True)(x,rt.shape[-3:])
        print(f"  n={n}: bit-exact={torch.equal(got,exp)}")
    except Exception as e:
        print(f"  n={n}: {type(e).__name__}: {str(e).splitlines()[0][:70]}")
    finally:
        lib.__exit__(None,None,None)
        for k in [k for k in list(lowerings) if "st3" in str(k)]: lowerings.pop(k,None)
