"""The minimal trigger for the TorchInductor `ops.constant` bug in torch 2.3.1.

This module holds *only* the code needed to hit the bug, so that every other
script in this repo (repro, patch verification, tests) exercises literally the
same trigger.

The trigger has three ingredients. Removing any one of them makes the bug
disappear -- which is why it took four failed attempts to reproduce (see
README.md "How this was localized").

  1. `mode="nearest"`   -> routes to the `upsample_nearestnd` lowering
  2. `size=` (not `scale_factor=`) computed from a *dynamically shaped tensor*
                        -> leaves the output size as a symbolic sympy expression
  3. `torch.inference_mode()`
                        -> disables the Autograd-keyed decomposition that would
                           otherwise rewrite the op before Inductor ever sees it

Ingredient 3 is the non-obvious one and the reason a "reasonable" standalone
repro does not fail.
"""

import torch
import torch.nn.functional as F

# The version this bug and its patch are specific to. torch >= 2.4 rewrote
# index_propagation.py, so neither the failure nor the fix applies there.
TARGET_TORCH_VERSION = "2.3.1"


def upsample_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour upsample `x` to `ref`'s spatial size.

    This is the shape of code every UNet / VAE decoder contains: upsample a
    feature map to match a skip connection whose size is only known at runtime.
    Because the size comes from `ref.shape`, it is symbolic under dynamic
    shapes -- ingredient 2.
    """
    return F.interpolate(x, size=(ref.shape[-2], ref.shape[-1]), mode="nearest")


def make_inputs(device: str = "cpu", in_hw=(32, 32), out_hw=(64, 64)):
    """Build (x, ref) where `ref`'s spatial dims are marked dynamic.

    `mark_dynamic` is what forces the compiler to keep `ref`'s H/W symbolic
    instead of burning in the concrete 64x64. In the inference server where this
    bug was first observed, the same effect came from enabling dynamic shapes in
    the serving stack's compile configuration.
    """
    x = torch.randn(1, 3, *in_hw, device=device)
    ref = torch.randn(1, 3, *out_hw, device=device)
    torch._dynamo.mark_dynamic(ref, 2)  # height stays symbolic
    torch._dynamo.mark_dynamic(ref, 3)  # width  stays symbolic
    return x, ref


def compile_and_run(device: str = "cpu", in_hw=(32, 32), out_hw=(64, 64)):
    """Compile `upsample_to_match` and run it. Raises on an unpatched torch 2.3.1.

    Returns the compiled output tensor. Caller is responsible for catching
    `torch._dynamo.exc.BackendCompilerFailed`.
    """
    torch._dynamo.reset()  # never reuse a cached compile between checks
    x, ref = make_inputs(device, in_hw, out_hw)
    compiled = torch.compile(upsample_to_match)

    # Ingredient 3: inference_mode bypasses the DispatchKey.Autograd
    # decomposition, so the raw aten op survives into Inductor's lowering.
    with torch.inference_mode():
        return compiled(x, ref)


def eager_reference(device: str = "cpu", in_hw=(32, 32), out_hw=(64, 64)):
    """The uncompiled answer, for numerical comparison against the patched build."""
    x, ref = make_inputs(device, in_hw, out_hw)
    with torch.inference_mode():
        return upsample_to_match(x, ref)


def pick_device(requested: str = "auto") -> str:
    """Resolve the device to run on. The bug reproduces on CPU *and* CUDA."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def version_warning() -> str:
    """Return a warning string if the running torch is not the target version."""
    actual = torch.__version__.split("+")[0]
    if actual != TARGET_TORCH_VERSION:
        return (
            f"WARNING: this repo targets torch {TARGET_TORCH_VERSION}, but the "
            f"active interpreter has torch {torch.__version__}.\n"
            f"         torch >= 2.4 rewrote _inductor/index_propagation.py: the "
            f"bug is already gone and this patch does not apply."
        )
    return ""
