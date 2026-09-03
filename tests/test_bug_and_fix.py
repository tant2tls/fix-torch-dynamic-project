"""Tests that assert the failure AND the fix.

These tests are state-aware: they read whether the installed torch is patched
and assert the behaviour appropriate to that state. Run them twice --
unpatched and patched -- to check both halves:

    python patch/patch.py revert && pytest -v
    python patch/patch.py apply  && pytest -v

Tests that require the opposite state skip rather than fail, so a single run is
always green and the skip reasons say what to do.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "patch"))

import repro  # noqa: F401  -- sets TORCHINDUCTOR_CACHE_DIR before torch

import patch as patch_mod  # noqa: E402
from repro.bug import (  # noqa: E402
    TARGET_TORCH_VERSION,
    compile_and_run,
    eager_reference,
    pick_device,
    upsample_to_match,
)

MARKER = "Cannot convert expression to float"
DEVICE = pick_device("auto")

pytestmark = pytest.mark.skipif(
    torch.__version__.split("+")[0] != TARGET_TORCH_VERSION,
    reason=f"needs torch {TARGET_TORCH_VERSION}; >= 2.4 rewrote index_propagation.py",
)


def current_state() -> str:
    return patch_mod.state(patch_mod.target_path())


needs_pristine = pytest.mark.skipif(
    current_state() != "pristine",
    reason="needs the unpatched file: run `python patch/patch.py revert`",
)
needs_patched = pytest.mark.skipif(
    current_state() != "patched",
    reason="needs the fix installed: run `python patch/patch.py apply`",
)


@pytest.fixture(autouse=True)
def _fresh_compile():
    """Never let one test reuse another's compiled artifact."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# --------------------------------------------------------------------------
# The patch tool itself
# --------------------------------------------------------------------------

def test_target_file_is_locatable():
    p = patch_mod.target_path()
    assert p.is_file()
    assert p.name == "index_propagation.py"


def test_state_is_recognized():
    """An unrecognized file would mean the patch tool cannot be trusted."""
    assert current_state() in ("pristine", "patched"), (
        "installed index_propagation.py matches neither expected form; "
        "restore a clean torch 2.3.1"
    )


def test_pristine_and_patched_text_differ():
    assert patch_mod.PRISTINE != patch_mod.PATCHED
    assert "try:" not in patch_mod.PRISTINE
    assert "index_expr" in patch_mod.PATCHED


# --------------------------------------------------------------------------
# The bug, when unpatched
# --------------------------------------------------------------------------

@needs_pristine
def test_bug_reproduces_when_unpatched():
    with pytest.raises(Exception) as ei:
        compile_and_run(DEVICE)
    assert MARKER in str(ei.value)


@needs_pristine
def test_failure_goes_through_the_claimed_frames():
    """Guard the root-cause story, not just the error string."""
    import traceback

    with pytest.raises(Exception) as ei:
        compile_and_run(DEVICE)

    frames = [(Path(f.filename).name, f.name)
              for f in traceback.extract_tb(ei.value.__traceback__)]
    for expected in [("lowering.py", "scale_fn"),
                     ("index_propagation.py", "propagate_sympy"),
                     ("index_propagation.py", "constant")]:
        assert expected in frames, f"missing frame {expected}"


@needs_pristine
@pytest.mark.parametrize("mode", ["nearest", "nearest-exact"])
def test_all_nearest_modes_affected(mode):
    x = torch.randn(1, 3, 32, 32, device=DEVICE)
    ref = torch.randn(1, 3, 64, 64, device=DEVICE)
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)
    fn = torch.compile(
        lambda a, r: F.interpolate(a, size=(r.shape[-2], r.shape[-1]), mode=mode))
    with pytest.raises(Exception) as ei, torch.inference_mode():
        fn(x, ref)
    assert MARKER in str(ei.value)


# --------------------------------------------------------------------------
# Ingredient ablation: each of these must NOT fail, patched or not
# --------------------------------------------------------------------------

def test_bilinear_is_unaffected():
    x = torch.randn(1, 3, 32, 32, device=DEVICE)
    ref = torch.randn(1, 3, 64, 64, device=DEVICE)
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)
    fn = torch.compile(
        lambda a, r: F.interpolate(a, size=(r.shape[-2], r.shape[-1]), mode="bilinear"))
    with torch.inference_mode():
        assert fn(x, ref).shape == (1, 3, 64, 64)


def test_scale_factor_path_is_unaffected():
    """scale_factor= takes the concrete `1.0 / scale` branch."""
    x = torch.randn(1, 3, 32, 32, device=DEVICE)
    ref = torch.randn(1, 3, 64, 64, device=DEVICE)
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)
    fn = torch.compile(lambda a, r: F.interpolate(a, scale_factor=2.0, mode="nearest"))
    with torch.inference_mode():
        assert fn(x, ref).shape == (1, 3, 64, 64)


def test_static_shapes_are_unaffected():
    """Without mark_dynamic the output size is a concrete int."""
    x = torch.randn(1, 3, 32, 32, device=DEVICE)
    ref = torch.randn(1, 3, 64, 64, device=DEVICE)
    fn = torch.compile(upsample_to_match)
    with torch.inference_mode():
        assert fn(x, ref).shape == (1, 3, 64, 64)


@pytest.mark.parametrize("ctx_name", ["no_grad", "grad_enabled"])
def test_non_inference_mode_is_unaffected(ctx_name):
    """The Autograd-keyed decomposition removes the op before Inductor."""
    import contextlib

    ctx = torch.no_grad() if ctx_name == "no_grad" else contextlib.nullcontext()
    x = torch.randn(1, 3, 32, 32, device=DEVICE)
    ref = torch.randn(1, 3, 64, 64, device=DEVICE)
    torch._dynamo.mark_dynamic(ref, 2)
    torch._dynamo.mark_dynamic(ref, 3)
    fn = torch.compile(upsample_to_match)
    with ctx:
        assert fn(x, ref).shape == (1, 3, 64, 64)


def test_decomposition_is_what_hides_the_bug():
    """Assert the mechanism directly: the op survives only under inference_mode."""
    import contextlib

    from torch._dynamo.backends.common import aot_autograd

    def survives(ctx):
        seen = {}

        def fw(gm, ex):
            seen["ops"] = [str(n.target) for n in gm.graph.nodes
                           if n.op == "call_function"]
            from torch._inductor.compile_fx import compile_fx
            return compile_fx(gm, ex)

        torch._dynamo.reset()
        fn = torch.compile(upsample_to_match,
                           backend=lambda gm, ex: aot_autograd(fw_compiler=fw)(gm, ex))
        x = torch.randn(1, 3, 32, 32, device=DEVICE)
        ref = torch.randn(1, 3, 64, 64, device=DEVICE)
        torch._dynamo.mark_dynamic(ref, 2)
        torch._dynamo.mark_dynamic(ref, 3)
        try:
            with ctx:
                fn(x, ref)
        except Exception:
            pass
        return any("upsample_nearest" in o for o in seen.get("ops", []))

    assert survives(torch.inference_mode()) is True
    assert survives(torch.no_grad()) is False
    assert survives(contextlib.nullcontext()) is False


@needs_pristine
def test_fallback_alternative_is_insufficient():
    """The 'cleaner' recovery, `fallback()`, does NOT fix this bug.

    Justifies choosing `index_expr` (README section 4). The alternative is
    monkey-patched in-process, so the installed torch is never modified.
    """
    from torch._inductor.index_propagation import IndexPropagation

    original = IndexPropagation.__getattr__

    def with_fallback_recovery(self, name):
        inner = original(self, name)

        def wrapped(*args, **kwargs):
            try:
                return inner(*args, **kwargs)
            except Exception:
                # The recovery that looks more general -- and is wrong here.
                return self.fallback(name, args, kwargs)

        return wrapped

    IndexPropagation.__getattr__ = with_fallback_recovery
    try:
        with pytest.raises(Exception) as ei:
            compile_and_run(DEVICE)
        msg = str(ei.value)
        # It fails differently: the sympy expression reaches a handler that
        # wants a number, so the contract violation resurfaces one layer down.
        assert MARKER not in msg, "expected a different failure, not the original"
        assert "NotImplementedError" in msg or "sympy" in msg.lower(), (
            f"expected a sympy/NotImplementedError-shaped failure, got: {msg[:200]}"
        )
    finally:
        IndexPropagation.__getattr__ = original


@needs_pristine
def test_index_expr_recovery_does_fix_it():
    """The chosen recovery works, verified without editing installed torch."""
    from torch._inductor.index_propagation import IndexPropagation

    original = IndexPropagation.__getattr__

    def with_index_expr_recovery(self, name):
        inner = original(self, name)

        def wrapped(*args, **kwargs):
            try:
                return inner(*args, **kwargs)
            except Exception:
                return self.propagate_sympy("index_expr", args, kwargs)

        return wrapped

    IndexPropagation.__getattr__ = with_index_expr_recovery
    try:
        torch.manual_seed(0)
        got = compile_and_run(DEVICE)
        torch.manual_seed(0)
        expected = eager_reference(DEVICE)
        assert torch.equal(got, expected)
    finally:
        IndexPropagation.__getattr__ = original


# --------------------------------------------------------------------------
# The fix, when patched
# --------------------------------------------------------------------------

@needs_patched
def test_compiles_when_patched():
    assert compile_and_run(DEVICE).shape == (1, 3, 64, 64)


@needs_patched
@pytest.mark.parametrize("out_hw", [(64, 64), (96, 48), (100, 70), (33, 33),
                                    (17, 17), (29, 31)])
def test_patched_output_is_bit_exact(out_hw):
    """Compiling is not enough -- the pixels must match eager exactly."""
    torch.manual_seed(0)
    got = compile_and_run(DEVICE, out_hw=out_hw)
    torch.manual_seed(0)
    expected = eager_reference(DEVICE, out_hw=out_hw)
    assert got.shape == expected.shape
    assert torch.equal(got, expected), (
        f"compiled output differs from eager at {out_hw}: "
        f"max|diff|={(got - expected).abs().max().item()}"
    )


@needs_patched
def test_one_artifact_serves_many_sizes():
    """The point of dynamic=True: recompilation should not be per-size."""
    torch._dynamo.reset()
    fn = torch.compile(upsample_to_match)
    with torch.inference_mode():
        for out_hw in [(64, 64), (80, 96), (128, 128)]:
            x = torch.randn(1, 3, 32, 32, device=DEVICE)
            ref = torch.randn(1, 3, *out_hw, device=DEVICE)
            torch._dynamo.mark_dynamic(ref, 2)
            torch._dynamo.mark_dynamic(ref, 3)
            assert fn(x, ref).shape == (1, 3, *out_hw)


@needs_patched
def test_unet_like_model_compiles_and_is_close():
    """The realistic setting. Conv fusion means close, not bit-exact."""
    from repro.unet_like import TinyUNet

    torch.manual_seed(0)
    model = TinyUNet().to(DEVICE).eval()
    torch._dynamo.reset()
    compiled = torch.compile(model)

    with torch.inference_mode():
        for hw in [(64, 64), (96, 80)]:
            x = torch.randn(1, 3, *hw, device=DEVICE)
            torch._dynamo.mark_dynamic(x, 2)
            torch._dynamo.mark_dynamic(x, 3)
            got, expected = compiled(x), model(x)
            assert got.shape == expected.shape
            assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)
