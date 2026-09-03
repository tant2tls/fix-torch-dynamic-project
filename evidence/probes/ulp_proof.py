"""Prove, without a GPU, that an approximate-reciprocal fp32 divide breaks the
floored index -- and that a correctly-rounded fp32 divide does not.

Triton's default `/` on fp32 lowers to div.full (approximate reciprocal:
compute 1/y then multiply). Model that in numpy and compare against the
correctly-rounded IEEE divide that div_rn / C++ / eager all produce.
"""
import numpy as np

f32 = np.float32
bad = 0
examples = []
# eager: floor(idx * float32(i/o))  -- correctly rounded divide
# triton default: floor(idx * float32(i * float32(1/o)))  -- approx reciprocal
for i in range(1, 513):
    for o in range(1, 513):
        s_rn  = f32(i) / f32(o)                    # correctly rounded
        s_apx = f32(f32(i) * f32(f32(1.0) / f32(o)))  # approx reciprocal model
        if s_rn == s_apx:
            continue
        for idx in range(o):
            a = int(np.floor(f32(idx) * s_rn))
            b = int(np.floor(f32(idx) * s_apx))
            if a != b:
                bad += 1
                if len(examples) < 6:
                    examples.append(f"{i}->{o} idx={idx}: rn={a} approx={b} "
                                    f"(scale {s_rn.view(np.uint32):#x} vs {s_apx.view(np.uint32):#x})")
                break
print(f"(i,o) pairs in 1..512 where approx-reciprocal flips a floored index: {bad}")
for e in examples: print("   ", e)

# and confirm the correctly-rounded fp32 divide MATCHES eager exactly (it is the
# same operation), i.e. div_rn introduces no error of its own
mismatch = 0
for i in range(1, 513):
    for o in range(1, 513):
        # ATen compute_scales_value: static_cast<float>(i) / o
        if (f32(i) / f32(o)) != f32(np.float64(i) / np.float64(o)):
            # narrowing a correctly-rounded f64 divide can differ from a f32
            # divide in rare double-rounding cases; count them
            mismatch += 1
print(f"f32-divide vs f64-divide-then-narrow disagreements (double rounding): {mismatch}")
