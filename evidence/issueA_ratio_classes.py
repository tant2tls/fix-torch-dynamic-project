#!/usr/bin/env python
"""Which ratios reproduce Issue A on ANY GPU, not just the one under my desk?

WHY. The four ratios in the issue draft were found by accident, and one of them
(`37->74`) is **A100-wrong / H100-clean**. Leading a correctness report with a
case a Hopper maintainer cannot reproduce is how it gets closed. So: choose the
filing ratios on purpose, and state the mechanism.

THE REFERENCE MUST BE EAGER, NOT EXACT RATIONAL. Issue A is "compiled disagrees
with eager". Eager (`ATen/native/UpSample.h`) is itself fp32:

    scale   = f32(i) / f32(o)          # correctly rounded fp32 divide
    src(d)  = min(int(f32(d * scale)), i - 1)

An earlier version of this script compared against exact rational `(d*i)//o` and
therefore measured a different, non-bug quantity (eager's own departure from
rational math, which compiled and eager agree about). It reported `448->192` as
`0/192` -- inverted from the real GPU result. Fixed: `eager_src()` below is the
only reference used.

WHAT VARIES ACROSS GPUs. Only one thing: the value the SM's `i / o` returns.
Triton lowers int/int truediv to an approximate divide (`x * rcp(y)`), which is
not correctly rounded; how it is wrong is SM-dependent. So:

    ratio is buggy on a device  <=>  that device's truediv scale, run through
                                     the SAME floor, moves at least one row

CROSS-ARCH PREDICTION WITHOUT THE OTHER GPU. I have one A100 this session, so I
cannot measure Hopper. But I can bound it: perturb the correctly-rounded scale by
+/-1 ULP and re-floor. A ratio whose rows move under BOTH perturbation directions
is wrong on any device whose divide errs by >=1 ULP either way -- that is the
architecture-robust set. A ratio that moves under only one direction reproduces
only on devices erring that way, which is exactly the `37->74` situation.

Emits `logs/issueA_ratio_classes.json`; prints the filing shortlist.
"""
import json
import math
import os
import struct
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"triton unavailable: {exc}")


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bits(x):
    return "0x" + struct.pack(">f", x).hex()


def ulp_step(x, direction):
    """The next representable fp32 after x, toward +/-inf.

    NOTE: `math.nextafter` steps in float64; rounding that back to fp32 lands on
    the SAME float32, making the perturbation a silent no-op. (That bug made this
    script report 0 ULP-robust ratios while 178 were observably miscompiling.)
    Step the fp32 BIT PATTERN instead. Assumes x > 0, which every scale here is.
    """
    u = struct.unpack(">I", struct.pack(">f", x))[0]
    u += 1 if direction > 0 else -1
    return struct.unpack(">f", struct.pack(">I", u))[0]


def eager_src(i, o, scale):
    """ATen UpSample.h with a given scale; the multiply is fp32."""
    return [min(int(f32(d * scale)), i - 1) for d in range(o)]


@triton.jit
def _scales(out_t, out_r, I, O):
    tl.store(out_t + 0, (I / O).to(tl.float32))  # what inductor emits today
    tl.store(out_r + 0, tl.math.div_rn(I.to(tl.float32), O.to(tl.float32)))


def main():
    hi = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"torch {torch.__version__} | triton {triton.__version__}")
    print(f"gpu   {name} | sm_{cap[0]}{cap[1]}")
    print()

    cands = []
    for i in (37, 63, 100, 255, 256, 363, 384, 448, 500):
        for o in range(2, hi + 1):
            if o != i:
                cands.append((i, o))
    for pair in [(448, 192), (384, 363), (37, 74), (32, 64), (32, 100)]:
        cands.append(pair)
    cands = sorted(set(cands))
    print(f"{len(cands)} candidate ratios, reference = eager fp32")

    t = torch.zeros(1, device="cuda", dtype=torch.float32)
    r = torch.zeros(1, device="cuda", dtype=torch.float32)

    classes = {"robust": [], "one_sided": [], "clean_here": []}
    for i, o in cands:
        _scales[(1,)](t, r, i, o)
        s_true, s_rn = t.item(), r.item()
        s_eager = f32(i / o)  # eager's correctly-rounded fp32 scale

        ref = eager_src(i, o, s_eager)
        # 1) does THIS device's approximate divide move a row?
        here = eager_src(i, o, s_true)
        bad_here = [(d, a, b) for d, (a, b) in enumerate(zip(ref, here)) if a != b]
        # 2) ULP robustness: would a device erring +1 / -1 ULP move a row?
        up = eager_src(i, o, ulp_step(s_eager, +1))
        dn = eager_src(i, o, ulp_step(s_eager, -1))
        bad_up = sum(1 for a, b in zip(ref, up) if a != b)
        bad_dn = sum(1 for a, b in zip(ref, dn) if a != b)

        rec = {
            "i": i,
            "o": o,
            "rows": o,
            "eager_bits": bits(s_eager),
            "truediv_bits": bits(s_true),
            "divrn_bits": bits(s_rn),
            "divide_correctly_rounded_here": s_true == s_rn,
            "scale_exactly_representable": (i / o) == s_eager,
            "wrong_here": len(bad_here),
            "first_wrong_here": bad_here[0] if bad_here else None,
            "wrong_if_plus1ulp": bad_up,
            "wrong_if_minus1ulp": bad_dn,
        }
        if bad_up and bad_dn:
            classes["robust"].append(rec)
        elif bad_up or bad_dn:
            classes["one_sided"].append(rec)
        else:
            classes["clean_here"].append(rec)

    nb, no_, nc = (len(classes[k]) for k in ("robust", "one_sided", "clean_here"))
    print()
    print(f"  ULP-ROBUST  (a row moves under BOTH +1 and -1 ULP) : {nb}")
    print(f"  ONE-SIDED   (only one direction moves a row)       : {no_}")
    print(f"  ULP-IMMUNE  (neither direction moves a row)        : {nc}")
    print()

    print("=== Observed on THIS device (truediv vs eager) ===")
    obs = [rc for rc in classes["robust"] + classes["one_sided"] + classes["clean_here"]
           if rc["wrong_here"]]
    print(f"{len(obs)} of {len(cands)} ratios currently miscompile here")
    print()

    print("=== FILING SHORTLIST: ULP-robust AND wrong on this device ===")
    print("These move a source pixel under a divide error of either sign, so a")
    print("maintainer on any SM whose divide is inexact for the ratio sees it.")
    print()
    hdr = f"{'i->o':<13}{'wrong here':<12}{'+1ULP':<8}{'-1ULP':<8}{'first (d,eager,compiled)':<26}"
    print(hdr)
    print("-" * len(hdr))
    short = sorted(
        (rc for rc in classes["robust"] if rc["wrong_here"]),
        key=lambda rc: -rc["wrong_here"],
    )
    for rc in short[:15]:
        print(
            f"{str(rc['i']) + '->' + str(rc['o']):<13}"
            f"{str(rc['wrong_here']) + '/' + str(rc['rows']):<12}"
            f"{rc['wrong_if_plus1ulp']:<8}{rc['wrong_if_minus1ulp']:<8}"
            f"{str(rc['first_wrong_here']):<26}"
        )

    print()
    print("=== The repo's cited ratios, reclassified against eager ===")
    hdr2 = f"{'i->o':<13}{'class':<12}{'wrong here':<12}{'+1ULP':<8}{'-1ULP':<8}{'exact repr':<11}"
    print(hdr2)
    print("-" * len(hdr2))
    for i, o in [(448, 192), (384, 363), (37, 74), (32, 64)]:
        for cls, recs in classes.items():
            for rc in recs:
                if (rc["i"], rc["o"]) == (i, o):
                    lbl = {"robust": "ULP-ROBUST", "one_sided": "ONE-SIDED",
                           "clean_here": "ULP-IMMUNE"}[cls]
                    print(
                        f"{str(i) + '->' + str(o):<13}{lbl:<12}"
                        f"{str(rc['wrong_here']) + '/' + str(o):<12}"
                        f"{rc['wrong_if_plus1ulp']:<8}{rc['wrong_if_minus1ulp']:<8}"
                        f"{str(rc['scale_exactly_representable']):<11}"
                    )

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs",
                       "issueA_ratio_classes.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "gpu": name,
                "sm": f"{cap[0]}{cap[1]}",
                "torch": torch.__version__,
                "triton": triton.__version__,
                "reference": "eager fp32 (ATen UpSample.h)",
                "counts": {k: len(v) for k, v in classes.items()},
                "miscompiling_here": len(obs),
                "shortlist": short[:40],
                "one_sided_wrong_here": [
                    rc for rc in classes["one_sided"] if rc["wrong_here"]
                ][:40],
            },
            fh,
            indent=1,
        )
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
