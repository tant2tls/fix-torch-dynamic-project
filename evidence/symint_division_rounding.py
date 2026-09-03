#!/usr/bin/env python3
"""Show that eager-numerics division rounding does not cover SymInt division.

The positive control is ordinary integer-tensor true division. With
TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING=1, Inductor emits div_rn and matches
eager. The shape-only case divides two dynamic tensor extents. That expression
is printed through TritonPrinter._print_IntTrueDiv instead, so the same flag
does not affect it.

The parent process runs two fresh child processes because Inductor reads the
environment flag during import. Each child gets its own cold cache.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def division_line(codes: list[str]) -> str:
    for line in "\n".join(codes).splitlines():
        stripped = line.strip()
        if " / " in stripped or "div_rn" in stripped:
            return stripped
    return "<no division found>"


def bit_mismatches(expected, actual) -> list[int]:
    import torch

    return (
        torch.nonzero(
            expected.contiguous().view(torch.int32)
            != actual.contiguous().view(torch.int32)
        )
        .flatten()
        .cpu()
        .tolist()
    )


def child() -> int:
    import torch
    import torch.nn.functional as F
    from torch._inductor.utils import run_and_get_code

    flag = os.environ["TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING"]
    print(f"flag={flag} torch={torch.__version__}")
    print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")

    # Positive control: the flag does cover tensor integer true division.
    numerators = torch.tensor(
        [448, 384, 37, 41, 74, 101], device="cuda", dtype=torch.int64
    )
    denominators = torch.tensor(
        [192, 363, 74, 82, 148, 202], device="cuda", dtype=torch.int64
    )

    def tensor_div(a, b):
        return a / b

    expected = tensor_div(numerators, denominators)
    actual, codes = run_and_get_code(
        torch.compile(tensor_div, fullgraph=True), numerators, denominators
    )
    tensor_bad = bit_mismatches(expected, actual)
    print(f"tensor_int_div wrong={len(tensor_bad)}/6 indices={tensor_bad}")
    print(f"tensor_int_div code={division_line(codes)}")

    # Minimal generic repro: no interpolate and no custom operator.
    def shape_div(x, y):
        scale = x.shape[0] / y.shape[0]
        return (
            torch.arange(y.shape[0], device=x.device, dtype=torch.float32) * scale
        )

    x = torch.empty(448, device="cuda")
    y = torch.empty(192, device="cuda")
    expected = shape_div(x, y)
    actual, codes = run_and_get_code(
        torch.compile(shape_div, dynamic=True, fullgraph=True), x, y
    )
    shape_bad = bit_mismatches(expected, actual)
    print(f"symint_div wrong={len(shape_bad)}/192 first={shape_bad[:8]}")
    print(f"symint_div code={division_line(codes)}")

    # User-visible symptom: nearest interpolation selects wrong source rows.
    image = (
        torch.arange(448, device="cuda", dtype=torch.float32)
        .view(1, 1, 448, 1)
        .expand(1, 1, 448, 4)
        .contiguous()
    )

    def resize(t):
        return F.interpolate(t, size=(192, 4), mode="nearest")

    expected = resize(image)
    actual = torch.compile(resize, dynamic=True, fullgraph=True)(image)
    expected_rows = expected[0, 0, :, 0].to(torch.int64)
    actual_rows = actual[0, 0, :, 0].to(torch.int64)
    row_bad = (
        torch.nonzero(expected_rows != actual_rows).flatten().cpu().tolist()
    )
    print(f"interpolate wrong_rows={len(row_bad)}/192 first={row_bad[:8]}")

    if flag == "0":
        assert tensor_bad, "positive control unexpectedly exact with flag off"
    else:
        assert not tensor_bad, "flag failed its ordinary tensor-division control"
    assert shape_bad, "SymInt division appears fixed; re-evaluate this finding"
    assert row_bad, "interpolate symptom appears fixed; re-evaluate this finding"
    return 0


def parent() -> int:
    for flag in ("0", "1"):
        print(f"\n=== division_rounding={flag} ===", flush=True)
        env = os.environ.copy()
        env["TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING"] = flag
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        env["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
            prefix=f"torchinductor_symint_div_{flag}_"
        )
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            env=env,
            check=False,
            text=True,
        )
        if result.returncode:
            return result.returncode

    print("\nFinding: the flag fixes the tensor-division control but does not")
    print("reach the symbolic shape-division path or its interpolate symptom.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child() if args.child else parent())
