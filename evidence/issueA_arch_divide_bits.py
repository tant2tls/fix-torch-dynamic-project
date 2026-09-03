#!/usr/bin/env python
"""WHY is Issue A's `37->74` case wrong on A100 but clean on H100?

`37/74` is exactly 0.5 -- a power of two, exactly representable in fp32. A
correctly-rounded divide CANNOT be wrong here, so the architecture split cannot
be explained by "float division is imprecise" in general. Something more specific
is going on, and the error direction says what: for `37->74` the compiled kernel
picks a LOWER source index (eager 1, compiled 0), whereas for the downsample
cases it picks a HIGHER one (eager 62, compiled 63).

A lower index for the 0.5 case means the computed scale landed just BELOW 0.5:
    floor(2 * 0.5)        = floor(1.0)        = 1   <- eager
    floor(2 * 0.49999997) = floor(0.99999994) = 0   <- compiled
That is the signature of an approximate reciprocal: `div.approx.f32` computes
x * rcp(y) rather than dividing, and `rcp(74)` is NOT exactly representable, so
the product can fall a ULP short of 0.5 even though 37/74 is exact.

This probe measures, on THIS device, the exact bit pattern of the divide Triton
emits for each ratio, and reports whether it is exactly equal to the correctly
rounded result. That converts "architecture-dependent, cause unknown" into a
located, measured mechanism -- and tells the issue reader which GPUs to expect
it on.

Run with all patches in any state: this exercises the DECOMPOSITION, not the
lowering the PRs touch. Nothing here depends on them.
"""
import os
import struct

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"triton unavailable: {exc}")


def bits(x: float) -> str:
    return "0x" + struct.pack(">f", x).hex()


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


# The ratios the repo cites for Issue A, plus the two clean controls.
RATIOS = [(448, 192), (384, 363), (37, 74), (32, 64), (32, 100)]


@triton.jit
def _truediv(out_ptr, i, o):
    """Exactly what inductor emits today: int/int truediv, then use it."""
    tl.store(out_ptr + 0, (i / o).to(tl.float32))


@triton.jit
def _divrn(out_ptr, i, o):
    """The correctly-rounded divide PR1 uses."""
    tl.store(out_ptr + 0, tl.math.div_rn(i.to(tl.float32), o.to(tl.float32)))


def on_device(kernel, i, o):
    buf = torch.zeros(1, device="cuda", dtype=torch.float32)
    kernel[(1,)](buf, i, o)
    return buf.item()


def main():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"torch {torch.__version__} | triton {triton.__version__}")
    print(f"gpu   {name} | sm_{cap[0]}{cap[1]}")
    print()
    print("For each ratio: the scale each divide produces on THIS device,")
    print("versus the correctly-rounded fp32 value of i/o.")
    print()
    hdr = (
        f"{'i->o':<12}{'exact fp32':<14}{'truediv':<14}{'div_rn':<14}"
        f"{'truediv==rn':<13}{'exactly repr':<13}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for i, o in RATIOS:
        want = f32(i / o)
        got_t = on_device(_truediv, i, o)
        got_r = on_device(_divrn, i, o)
        # is i/o exactly representable in fp32 at all?
        exact_repr = (i / o) == want
        same = got_t == got_r
        rows.append((i, o, want, got_t, got_r, same, exact_repr))
        print(
            f"{f'{i}->{o}':<12}{bits(want):<14}{bits(got_t):<14}{bits(got_r):<14}"
            f"{str(same):<13}{str(exact_repr):<13}"
        )

    print()
    print("Which output coordinates does each divide send to the wrong source pixel?")
    print("(model of ATen's UpSample.h: src = min(floor(d * scale), i-1))")
    print()
    hdr2 = f"{'i->o':<12}{'truediv wrong':<16}{'div_rn wrong':<16}{'first truediv miss':<28}"
    print(hdr2)
    print("-" * len(hdr2))
    for i, o, want, got_t, got_r, _, _ in rows:
        bad_t, bad_r, first = 0, 0, ""
        for d in range(o):
            # ATen UpSample.h: floorf(dst_index * scale) -- the multiply is fp32.
            # Doing it in float64 here would attribute the MULTIPLY's rounding to
            # the DIVIDE and invent disagreements the GPU does not show.
            ref = min(int(f32(d * want)), i - 1)
            vt = min(int(f32(d * got_t)), i - 1)
            vr = min(int(f32(d * got_r)), i - 1)
            if vt != ref:
                bad_t += 1
                if not first:
                    first = f"row {d}: want {ref}, got {vt}"
            if vr != ref:
                bad_r += 1
        print(f"{f'{i}->{o}':<12}{f'{bad_t}/{o}':<16}{f'{bad_r}/{o}':<16}{first:<28}")

    print()
    print("Reading it:")
    print("  * a ratio where truediv != div_rn in BITS is one where this SM's")
    print("    divide is not correctly rounded -- that is the mechanism.")
    print("  * `37->74` is exactly representable (0.5), so ANY bit difference")
    print("    there is purely the approximate reciprocal, not float precision.")
    print("  * div_rn wrong == 0 everywhere is the claim PR1's design rests on.")


if __name__ == "__main__":
    main()
