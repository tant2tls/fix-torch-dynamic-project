#!/usr/bin/env python
"""Falsification test: does the ULP model PREDICT real F.interpolate mismatches?

`issueA_ratio_classes.py` builds a host-side model of Issue A and classifies
ratios by whether a +/-1 ULP error in the scale moves a source pixel. On the four
ratios the repo already knew, the model's counts match the GPU exactly -- but that
is a fit, not a prediction. Fitting four known points proves nothing.

So: take ratios the repo has NEVER run, have the model commit to an exact
wrong-row count in advance, then run genuine `torch.compile(F.interpolate)` on the
GPU and compare. A model that predicts unseen ratios exactly is a tool; one that
only explains known ones is a story.

Everything here is stock behaviour of the DECOMPOSITION, so the three PR patches
are irrelevant to the result (verify with `tools/state.py status` if unsure).

Exit 0 if every prediction lands, 1 otherwise.
"""
import os
import struct
import sys

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"triton unavailable: {exc}")


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bits(x):
    return "0x" + struct.pack(">f", x).hex()


@triton.jit
def _scale_truediv(out, I, O):
    tl.store(out + 0, (I / O).to(tl.float32))


def device_truediv(i, o, buf):
    _scale_truediv[(1,)](buf, i, o)
    return buf.item()


def eager_src(i, o, scale):
    return [min(int(f32(d * scale)), i - 1) for d in range(o)]


def predict(i, o, buf):
    """Model: eager uses correctly-rounded fp32; inductor uses the SM's truediv."""
    s_eager = f32(i / o)
    s_true = device_truediv(i, o, buf)
    ref = eager_src(i, o, s_eager)
    got = eager_src(i, o, s_true)
    bad = [(d, a, b) for d, (a, b) in enumerate(zip(ref, got)) if a != b]
    return bad, s_eager, s_true


def measure(i, o, mode="nearest"):
    """Real torch.compile vs eager, value == source row index."""
    torch._dynamo.reset()
    x = (
        torch.arange(i, device="cuda", dtype=torch.float32)
        .view(1, 1, i, 1)
        .expand(1, 1, i, 4)
        .contiguous()
    )
    f = lambda t: F.interpolate(t, size=(o, 4), mode=mode)
    want = f(x)[0, 0, :, 0].to(torch.int64).tolist()
    got = torch.compile(f, dynamic=True)(x)[0, 0, :, 0].to(torch.int64).tolist()
    return [(d, a, b) for d, (a, b) in enumerate(zip(want, got)) if a != b]


# Ratios the repo has never run. Chosen from the model's shortlist across the
# whole range of predicted counts, INCLUDING predicted-zero cases -- a model that
# only ever predicts "wrong" is not falsifiable.
UNSEEN = [
    (448, 368),  # model: 9 wrong, ULP-robust
    (448, 384),  # model: 7 wrong
    (500, 120),  # model: 6 wrong
    (100, 370),  # model: 4 wrong
    (363, 319),  # model: 3 wrong
    (384, 246),  # model: 3 wrong
    (448, 96),   # model: 3 wrong
    (256, 128),  # model: expect 0 -- exact power-of-two downsample
    (64, 256),   # model: expect 0 -- exact upsample
    (255, 256),  # model: expect 0 -- near-unity, was clean in aten_ref_probe
    (1000, 999), # model: expect 0
    (100, 70),   # model: expect 0
]


def main():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"torch {torch.__version__} | triton {triton.__version__}")
    print(f"gpu   {name} | sm_{cap[0]}{cap[1]}")
    print()
    print("Model commits FIRST, then the GPU runs. Ratios never tested in this repo.")
    print()
    hdr = (
        f"{'i->o':<12}{'predicted':<12}{'measured':<12}{'match':<8}"
        f"{'eager scale':<14}{'truediv':<14}"
    )
    print(hdr)
    print("-" * len(hdr))

    buf = torch.zeros(1, device="cuda", dtype=torch.float32)
    ok = True
    detail = []
    for i, o in UNSEEN:
        pred, s_e, s_t = predict(i, o, buf)
        meas = measure(i, o)
        same_count = len(pred) == len(meas)
        # stronger: the exact rows and the exact substituted pixels must agree
        same_rows = pred == meas
        good = same_count and same_rows
        ok &= good
        tag = "yes" if good else ("count" if same_count else "NO")
        print(
            f"{str(i) + '->' + str(o):<12}"
            f"{str(len(pred)) + '/' + str(o):<12}"
            f"{str(len(meas)) + '/' + str(o):<12}"
            f"{tag:<8}{bits(s_e):<14}{bits(s_t):<14}"
        )
        detail.append((i, o, pred, meas))

    print()
    for i, o, pred, meas in detail:
        if pred != meas:
            print(f"  MISMATCH {i}->{o}")
            print(f"    predicted first 3: {pred[:3]}")
            print(f"    measured  first 3: {meas[:3]}")

    nonzero = sum(1 for _, _, p, _ in detail if p)
    zero = len(detail) - nonzero
    print()
    print(f"{nonzero} ratios predicted wrong, {zero} predicted clean.")
    if ok:
        print("ALL PREDICTIONS EXACT (row indices and substituted pixels).")
        print("The host-side ULP model is a predictor, not a post-hoc fit --")
        print("so it can be used to choose architecture-robust filing ratios.")
    else:
        print("At least one prediction missed. Do NOT use the model to choose")
        print("filing ratios until the discrepancy is understood.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
