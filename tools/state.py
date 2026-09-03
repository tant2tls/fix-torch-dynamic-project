#!/usr/bin/env python
"""Toggle each of the three PRs on/off in the INSTALLED torch, independently.

Exact-string replacement in both directions, so it is idempotent and it refuses
to run against a file matching neither form. This exists because A100 numbers are
only meaningful if the state they were measured in is unambiguous.

    python state.py status
    python state.py set pr1=on pr2=off pr3=on
"""

import os
import subprocess
import sys

TORCH = os.path.dirname(
    subprocess.check_output(
        [sys.executable, "-c", "import torch,os;print(os.path.dirname(torch.__file__))"],
        text=True,
    ).strip()
    + "/x"
)
LOWERING = os.path.join(TORCH, "_inductor", "lowering.py")
VIRTUALIZED = os.path.join(TORCH, "_inductor", "virtualized.py")

# ---------------------------------------------------------------- PR1: forward
PR1_A_OFF = "    inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]\n"
PR1_A_ON = """\
    # `ops.constant` requires a concrete value, so `i / o` can only be folded
    # here when `o` is concrete. `i_sizes` is guarded above, but `o_sizes` is not,
    # so under dynamic shapes `o` may still be symbolic. `None` marks those dims,
    # and `scale_fn` divides by `o_size` in the kernel instead. Guarding `o` here
    # would specialize on the output size, costing a recompile per distinct size.
    inv_scales = [
        None if isinstance(o, sympy.Expr) and not o.is_number else i / o
        for i, o in zip(i_sizes, o_sizes)
    ]
"""

PR1_B_OFF = """\
    def scale_fn(x, scale, size):
        # Nearest Exact: input_index = round(scale * (output_index + 0.5) - 0.5)
        #                            = floor(scale * (output_index + 0.5))
        # Nearest: input_index = floor(scale * output_index)
        x = ops.index_expr(x, torch.float32)
        if exact:
            x = ops.add(x, ops.constant(0.5, torch.float32))
        x = ops.mul(x, ops.constant(scale, torch.float32))
        x = ops.to_dtype(x, torch.int32)
        return ops.indirect_indexing(x, size, check=False)

    def fn(idx):
        x = idx[-n:]
        b = idx[:-n]
        return x_loader(
            [*b, *[scale_fn(i, s, size) for i, s, size in zip(x, inv_scales, i_sizes)]]
        )
"""
PR1_B_ON = """\
    def scale_fn(x, inv_scale, o_size, size):
        # Nearest Exact: input_index = round(scale * (output_index + 0.5) - 0.5)
        #                            = floor(scale * (output_index + 0.5))
        # Nearest: input_index = floor(scale * output_index)
        x = ops.index_expr(x, torch.float32)
        if exact:
            x = ops.add(x, ops.constant(0.5, torch.float32))
        if inv_scale is None:
            # Divide at runtime. Eager computes the scale as a float32 division
            # (`compute_scales_value` in ATen/native/UpSample.h), so use div_rn
            # to get the same correctly-rounded result -- plain `/` lowers to
            # Triton's approximate div.full, and being one ulp off changes the
            # floored index below.
            inv_scale = ops.div_rn(
                ops.constant(size, torch.float32),
                ops.index_expr(o_size, torch.float32),
            )
        else:
            inv_scale = ops.constant(inv_scale, torch.float32)
        x = ops.mul(x, inv_scale)
        x = ops.to_dtype(x, torch.int32)
        return ops.indirect_indexing(x, size, check=False)

    def fn(idx):
        x = idx[-n:]
        b = idx[:-n]
        return x_loader(
            [
                *b,
                *[
                    scale_fn(i, s, o_size, size)
                    for i, s, o_size, size in zip(x, inv_scales, o_sizes, i_sizes)
                ],
            ]
        )
"""

# --------------------------------------------------------- PR2: ops.constant
PR2_OFF = """\
    @staticmethod
    def _unwrap(x):
"""
PR2_ON = '''\
    # pyrefly: ignore [bad-override]
    def constant(self, value: Any, dtype: torch.dtype) -> Any:
        # `ops.constant` takes a concrete leaf value. A lowering that derives one
        # from sizes it never made concrete can pass a symbolic expression here,
        # which no backend can codegen; the failure then surfaces well downstream
        # (e.g. "NotImplementedError: argument of type: <class
        # 'sympy.core.mul.Mul'>" from fx.proxy while tracing) with nothing naming
        # the op or the fix. Several lowerings already dispatch on this by hand,
        # e.g. `_full` and the addcmul/addcdiv helpers in lowering.py.
        value = OpsWrapper._unwrap(value)
        if isinstance(value, sympy.Expr) and not value.is_number:
            raise TypeError(
                f"ops.constant() requires a concrete value, got the symbolic "
                f"expression {value!r} (free symbols: "
                f"{sorted(map(str, value.free_symbols))}). Either make its "
                f"inputs concrete (e.g. V.graph.sizevars.guard_int), which "
                f"specializes on them, or keep it dynamic and pass it to "
                f"ops.index_expr(), which accepts symbolic expressions."
            )
        return self._default("constant", (value, dtype), {})

    @staticmethod
    def _unwrap(x):
'''

PR2_IMPORT_OFF = "from torch.utils._ordered_set import OrderedSet\n"
PR2_IMPORT_ON = "import sympy\n\nfrom torch.utils._ordered_set import OrderedSet\n"

# -------------------------------------------------------------- PR3: backward
PR3_OFF = "    *_batch, out_h, out_w = input_size\n"
PR3_ON = """\
    *_batch, out_h, out_w = input_size
    # `h_kernel_max` / `w_kernel_max` below become `range()` bounds in
    # `_adaptive_pooling_fn`, and `inp_h % out_h` is evaluated eagerly here, so
    # these must be concrete. Guard them the same way the input sizes are
    # guarded above; unlike the forward lowering there is no way to defer a
    # Python loop bound to the kernel.
    out_h = V.graph.sizevars.guard_int(out_h)
    out_w = V.graph.sizevars.guard_int(out_w)
"""

# (name, file, [(off_text, on_text), ...])
PRS = {
    "pr1": (LOWERING, [(PR1_A_OFF, PR1_A_ON), (PR1_B_OFF, PR1_B_ON)]),
    "pr2": (VIRTUALIZED, [(PR2_IMPORT_OFF, PR2_IMPORT_ON), (PR2_OFF, PR2_ON)]),
    "pr3": (LOWERING, [(PR3_OFF, PR3_ON)]),
}


def _read(p):
    with open(p) as f:
        return f.read()


def _write(p, s):
    with open(p, "w") as f:
        f.write(s)


def detect(name):
    path, pairs = PRS[name]
    src = _read(path)
    states = []
    for off, on in pairs:
        n_on, n_off = src.count(on), src.count(off)
        # `on` text contains `off` text for PR2/PR3 (append-style), so check on first
        if n_on >= 1:
            states.append("on")
        elif n_off >= 1:
            states.append("off")
        else:
            states.append("MISSING")
    if all(s == "on" for s in states):
        return "on"
    if all(s == "off" for s in states):
        return "off"
    return "INCONSISTENT" + str(states)


def setpr(name, want):
    path, pairs = PRS[name]
    cur = detect(name)
    if cur == want:
        return f"{name}: already {want}"
    if cur.startswith("MISSING") or cur.startswith("INCONSISTENT"):
        raise SystemExit(f"REFUSING: {name} is in state {cur} in {path}")
    src = _read(path)
    for off, on in pairs:
        frm, to = (off, on) if want == "on" else (on, off)
        if src.count(frm) != 1:
            raise SystemExit(
                f"REFUSING: {name} anchor appears {src.count(frm)}x (want 1) in {path}"
            )
        src = src.replace(frm, to)
    _write(path, src)
    got = detect(name)
    if got != want:
        raise SystemExit(f"FAILED: {name} -> {got}, wanted {want}")
    return f"{name}: {cur} -> {want}"


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "status":
        for n in ("pr1", "pr2", "pr3"):
            print(f"{n} = {detect(n)}")
        print(f"torch dir: {TORCH}")
    elif sys.argv[1] == "set":
        for arg in sys.argv[2:]:
            n, _, v = arg.partition("=")
            print(setpr(n, v))
        # invalidate compiled caches so a state flip cannot be masked
        for n in ("pr1", "pr2", "pr3"):
            print(f"  {n} = {detect(n)}")
    else:
        raise SystemExit(__doc__)
