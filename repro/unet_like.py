#!/usr/bin/env python3
"""A UNet-shaped decoder: the realistic setting where this bug actually bit.

The minimal repro in `bug.py` is deliberately tiny. This script shows the same
failure in the structure that produced it in production -- a decoder that
upsamples a feature map to match a skip connection whose spatial size is only
known at runtime. No diffusers, no ComfyUI, no model weights: just the shape of
the computation.

    python repro/unet_like.py [--device cpu|cuda|auto]

Exit codes match repro.py: 0 = compiled correctly, 2 = the bug reproduced.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repro  # noqa: F401  -- sets TORCHINDUCTOR_CACHE_DIR before torch

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from repro.bug import pick_device, version_warning  # noqa: E402


class UpBlock(nn.Module):
    """Upsample to the skip connection's size, concatenate, convolve.

    `F.interpolate(..., size=skip.shape[-2:])` is the idiomatic way to write
    this -- and it is the exact pattern that fails, because `skip.shape` is
    symbolic once the input resolution is dynamic.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(skip.shape[-2], skip.shape[-1]), mode="nearest")
        return F.silu(self.conv(torch.cat([x, skip], dim=1)))


class TinyUNet(nn.Module):
    """Two down blocks, two up blocks. Structurally a diffusion UNet decoder."""

    def __init__(self, ch: int = 16):
        super().__init__()
        self.in_conv = nn.Conv2d(3, ch, 3, padding=1)
        self.down1 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)
        self.up1 = UpBlock(ch * 4, ch * 2, ch * 2)
        self.up2 = UpBlock(ch * 2, ch, ch)
        self.out_conv = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = F.silu(self.in_conv(x))     # full resolution
        s1 = F.silu(self.down1(s0))      # 1/2
        h = F.silu(self.down2(s1))       # 1/4
        h = self.up1(h, s1)              # 1/4 -> 1/2, matched to s1
        h = self.up2(h, s0)              # 1/2 -> 1/1, matched to s0
        return self.out_conv(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    warn = version_warning()
    if warn:
        print(warn + "\n")
    dev = pick_device(args.device)

    print("=" * 78)
    print(f"UNet-shaped decoder under dynamic shapes "
          f"(torch {torch.__version__}, {dev})")
    print("=" * 78)
    print("A decoder that upsamples to match skip connections -- the production")
    print("pattern. Serving variable image sizes means dynamic shapes.\n")

    torch.manual_seed(0)
    model = TinyUNet().to(dev).eval()

    torch._dynamo.reset()
    compiled = torch.compile(model)

    # Serving at multiple resolutions is what makes the shapes dynamic: the
    # second distinct size triggers a dynamic recompile, at which point the
    # output size becomes symbolic.
    sizes = [(64, 64), (96, 80), (128, 128)]
    print(f"Serving resolutions: {sizes}\n")

    # Unlike the pure-interpolate repro, this model contains convolutions, so
    # compiled output is NOT expected to be bit-identical: Inductor fuses and
    # reorders float arithmetic, and picks different conv kernels than eager.
    # A few ULPs of fp32 drift is correct behaviour; the upsample indices
    # themselves are integer and must be exact, which is what a large diff
    # would reveal.
    TOL = 1e-4
    worst = 0.0
    try:
        with torch.inference_mode():
            for hw in sizes:
                x = torch.randn(1, 3, *hw, device=dev)
                torch._dynamo.mark_dynamic(x, 2)
                torch._dynamo.mark_dynamic(x, 3)
                out = compiled(x)
                expected = model(x)
                maxdiff = (out.float() - expected.float()).abs().max().item()
                worst = max(worst, maxdiff)
                print(f"  {str(hw):<12} -> out {tuple(out.shape)}  "
                      f"max|diff| vs eager = {maxdiff:.3g}  "
                      f"{'OK' if maxdiff <= TOL else 'TOO LARGE'}")
    except Exception as exc:  # noqa: BLE001
        if "Cannot convert expression to float" in str(exc):
            print("\nRESULT: compilation FAILED -- the bug reproduced in a realistic")
            print("        UNet decoder, not just a synthetic one-liner.\n")
            print("        TypeError: Cannot convert expression to float\n")
            print("        This is what a production inference server sees: it")
            print("        serves one resolution fine, then dies on the second.")
            print("\n        Fix it: python patch/patch.py apply")
            return 2
        print(f"\nUnexpected {type(exc).__name__}: {str(exc)[:300]}")
        return 3

    if worst > TOL:
        print(f"\nRESULT: compiled, but max|diff| {worst:.3g} exceeds the {TOL:g}")
        print("        tolerance. That is more than fp32 fusion drift -- investigate.")
        return 1

    print(f"\nRESULT: compiled and correct at every resolution (worst "
          f"max|diff| {worst:.3g},")
    print(f"        within the {TOL:g} fp32 tolerance), with one compiled artifact")
    print("        serving all three sizes.\n")
    print("        Note: not bit-exact here, unlike repro/verify_fix.py. This model")
    print("        has convolutions, and Inductor fuses/reorders float arithmetic")
    print("        and selects different conv kernels than eager. The upsample")
    print("        indices are integer arithmetic and remain exact -- a real index")
    print("        error would show up as a large diff, not a few ULPs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
