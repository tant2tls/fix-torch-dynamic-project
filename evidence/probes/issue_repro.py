import torch
import torch.nn.functional as F
from torch._inductor.lowering import register_lowering, upsample_nearestnd

with torch.library._scoped_library("demo", "FRAGMENT") as lib:
    lib.define("ups2d(Tensor x, SymInt h, SymInt w) -> Tensor")
    ref = lambda x, h, w: F.interpolate(x, size=(int(h), int(w)), mode="nearest")
    lib.impl("ups2d", lambda x, h, w: x.new_empty((x.shape[0], x.shape[1], h, w)), "Meta")
    lib.impl("ups2d", ref, "CPU")
    register_lowering(torch.ops.demo.ups2d)(
        lambda x, h, w: upsample_nearestnd(x, [h, w], (None, None), n=2)
    )

    x = torch.randn(1, 3, 32, 32)
    sizes = torch.randn(1, 3, 64, 64)
    torch._dynamo.maybe_mark_dynamic(sizes, 2)
    torch._dynamo.maybe_mark_dynamic(sizes, 3)
    torch.compile(
        lambda a, s: torch.ops.demo.ups2d(a, s[0], s[1]), dynamic=True
    )(x, sizes.shape[-2:])
