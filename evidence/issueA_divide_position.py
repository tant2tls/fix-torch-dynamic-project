#!/usr/bin/env python
"""Is Issue A's divide per-element, or loop-invariant like PR1's?

This decides whether Issue A is fixable cheaply or is a genuine repeat of
#164144. My earlier note said "there the divide is per-element in a memory-bound
gather, so #164144 applies" -- but the Triton I captured for 37->74 showed

    tmp0 = (ks0 / 74).to(tl.float32)     # <- ks0 only?
    tmp1 = (x1).to(tl.float32)
    tmp2 = tmp1 * tmp0

which does NOT reference xindex. If that holds generally, the note is wrong and
the #164144 objection does not transfer to the decomposition either.

Checks the emitted kernel for the plain F.interpolate path (no patches involved)
across several dynamic configurations, and classifies every divide by whether its
operands depend on the program index.
"""
import os
import re

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code

DEV = "cuda"
IDX = ("xindex", "x0", "x1", "x2", "x3", "xoffset")


def kernel_body(src):
    out, inside = [], False
    for line in src.splitlines():
        if re.match(r"\s*def triton_", line):
            inside = True
        if inside:
            out.append(line.strip())
        if inside and "tl.store" in line:
            break
    return out


def classify(body):
    """Return (divides, verdict) -- resolve each divide's operands transitively."""
    defs = {}
    for ln in body:
        m = re.match(r"(tmp\d+|x\d+) = (.+)", ln)
        if m:
            defs[m.group(1)] = m.group(2)

    def depends_on_index(expr, depth=0):
        if depth > 12:
            return False
        if any(re.search(rf"\b{t}\b", expr) for t in IDX):
            return True
        for name in re.findall(r"\b(tmp\d+|x\d+)\b", expr):
            if name in defs and depends_on_index(defs[name], depth + 1):
                return True
        return False

    divides = []
    for name, expr in defs.items():
        # a real division: '/' between operands, or an explicit div call
        if re.search(r"[^/]/[^/]", expr) or "div_rn" in expr or "truediv" in expr:
            divides.append((name, expr, depends_on_index(expr)))
    return divides


CASES = [
    ("dyn input, static out 37->74", (1, 1, 37, 4), (74, 4), True),
    ("dyn input, static out 448->192", (1, 1, 448, 4), (192, 4), True),
    ("dyn both 64->128", (1, 1, 64, 4), (128, 4), True),
]

print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}\n")
for label, in_shape, out_hw, dyn in CASES:
    torch._dynamo.reset()
    x = torch.randn(*in_shape, device=DEV)
    if dyn:
        torch._dynamo.mark_dynamic(x, 2)
    f = lambda t: F.interpolate(t, size=out_hw, mode="nearest")
    _, codes = run_and_get_code(torch.compile(f, dynamic=dyn), x)
    body = kernel_body("\n".join(codes))
    divs = classify(body)
    print(f"--- {label} ---")
    if not divs:
        print("    no divide in the kernel body (scale folded to a literal)")
    for name, expr, per_elem in divs:
        tag = "PER-ELEMENT" if per_elem else "LOOP-INVARIANT"
        print(f"    {name} = {expr[:64]:<64} {tag}")
    print()

print("If every divide is LOOP-INVARIANT, then the claim 'Issue A's divide is")
print("per-element, so #164144 applies' is WRONG, and Issue A is as cheap to fix")
print("as PR1 -- which would make it the highest-value PR in the set, not an")
print("issue-only report.")
