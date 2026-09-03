#!/usr/bin/env python3
"""Check whether CUDA nearest-neighbor backward transposes its own forward map.

For a one-dimensional spatial signal, the gradient of ``forward(x).sum()`` is
exactly the number of output positions that selected each input position.  We
recover the selection map from eager forward itself, so this test does not use
an external definition of nearest-neighbor rounding.

The check is intentionally eager-only.  A failure here explains why repairing
only TorchInductor's dynamic forward can create a compiled-vs-eager gradient
difference: eager CUDA backward is already inconsistent with eager CUDA
forward at rounding-sensitive ratios.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


RATIOS = [
    (448, 192),
    (384, 363),
    (37, 74),
    (32, 64),
    (64, 32),
]
MODES = ("nearest", "nearest-exact")


def check(device: str, mode: str, input_size: int, output_size: int) -> tuple[int, float]:
    # Values encode source positions, making the eager forward map observable.
    x = (
        torch.arange(input_size, device=device, dtype=torch.float32)
        .view(1, 1, input_size)
        .requires_grad_()
    )
    y = F.interpolate(x, size=output_size, mode=mode)
    source = y.detach().view(-1).to(torch.int64)
    expected = torch.bincount(source, minlength=input_size).to(torch.float32)
    y.sum().backward()
    actual = x.grad
    if actual is None:
        raise AssertionError("autograd did not populate the input gradient")
    actual = actual.view(-1)
    wrong = int(torch.count_nonzero(actual != expected).item())
    max_abs = float((actual - expected).abs().max().item())
    return wrong, max_abs


def main() -> int:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    print(f"torch={torch.__version__}")
    if torch.cuda.is_available():
        print(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"capability={torch.cuda.get_device_capability(0)}"
        )

    failures: list[tuple[str, str, int, int, int, float]] = []
    print("device  mode           ratio       wrong   max_abs")
    for device in devices:
        for mode in MODES:
            for input_size, output_size in RATIOS:
                wrong, max_abs = check(device, mode, input_size, output_size)
                print(
                    f"{device:<7} {mode:<14} {input_size:>4}->{output_size:<4} "
                    f"{wrong:>5}/{input_size:<4} {max_abs:g}"
                )
                if wrong:
                    failures.append(
                        (device, mode, input_size, output_size, wrong, max_abs)
                    )

    cpu_failures = [failure for failure in failures if failure[0] == "cpu"]
    if cpu_failures:
        raise AssertionError(f"unexpected CPU inconsistencies: {cpu_failures}")

    if torch.cuda.is_available():
        expected_sensitive = {
            ("nearest", 448, 192),
            ("nearest", 384, 363),
            ("nearest-exact", 384, 363),
        }
        observed = {(mode, i, o) for _, mode, i, o, _, _ in failures}
        if not expected_sensitive.issubset(observed):
            raise AssertionError(
                "known CUDA-sensitive ratios became consistent; re-evaluate the finding"
            )

    print("\nExpected gradient is the exact transpose of the observed eager forward map.")
    print(f"inconsistent cases={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
