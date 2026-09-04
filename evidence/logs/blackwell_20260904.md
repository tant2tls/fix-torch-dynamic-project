# Blackwell release-preparation verification — 2026-09-04

This is a fresh verification on one NVIDIA RTX PRO 6000 Blackwell Server
Edition. It extends the earlier A100, H100, and 2026-09-03 Blackwell evidence;
it does not replace those runs.

## Claim boundary

This device reports compute capability 12.0 (`sm_120`), but it is not a B200.
Correctness results may be described as RTX PRO 6000 Blackwell or SM 12.0
results. Performance conclusions from this host must not be generalized to B200.

## Preflight and environment reconstruction

The first required state check was invalid because the `dynamic` environment
contained no PyTorch installation:

```text
$ conda run -n dynamic python tools/state.py status
ModuleNotFoundError: No module named 'torch'
...
CalledProcessError: Command '['/root/miniconda3/envs/dynamic/bin/python',
'-c', 'import torch,os;print(os.path.dirname(torch.__file__))']' returned
non-zero exit status 1.
```

No result was interpreted from that state. The environment was reconstructed
from the repository's canonical lock:

```text
$ conda run -n dynamic python -m pip install -r requirements-upstream.txt
Successfully installed ... numpy-2.4.6 ... pytest-9.1.1 ...
torch-2.13.0+cu130 triton-3.7.1 ...
```

The required state check and live hardware probe then reported:

```text
$ conda run -n dynamic python tools/state.py status
pr1 = off
pr2 = off
pr3 = off
torch dir: /root/miniconda3/envs/dynamic/lib/python3.11/site-packages/torch

python=3.11.16
torch=2.13.0+cu130
torch_git=cf30153c4c131c8164ee7798e5022d810682e2cb
cuda_build=13.0
cuda_available=True
gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition
compute_capability=12.0
device_count=1

$ nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.09, 12.0
```

## Stock-wheel verification — all patches OFF

The original 2.3.1 route remains fixed in the current release wheel:

```text
$ conda run -n dynamic python repro/check_upstream.py
torch 2.13.0+cu130
  UNAFFECTED: compiles, shape (1, 3, 64, 64), bit-exact=True
  decomposition also keyed on CompositeImplicitAutograd: True
  -> so it fires under inference_mode too, and the buggy
     upsample_nearestnd lowering is never reached. (In 2.3.1 the
     decomposition was Autograd-only, which is the whole bug.)
  SymPyOps.constant still coerces with float(): False
  -> and the eager sympy.Float(float(...)) coercion is gone, so a
     symbolic value could not raise there even if it arrived.
  caller still passes a possibly-symbolic scale to ops.constant: True
  -> the type-contract violation itself was never repaired, only
     made unreachable. Latent, not fixed.
```

The current issue repro harness then matched its documented stock behavior:

```text
$ conda run -n dynamic python evidence/verify_issue_repros.py
torch 2.13.0+cu130  cuda=True
gpu   NVIDIA RTX PRO 6000 Blackwell Server Edition

Issue A: wrong pixels, dynamic=True, CUDA
nearest 448->192: 7/192 rows differ; first [(27, 62, 63)]
nearest 384->363: 2/363 rows differ; first [(121, 127, 128)]
nearest 37->74: 36/74 rows differ; first [(2, 1, 0)]
nearest-exact 384->363: 2/363 rows differ; first [(60, 63, 64)]
--> repro RAN as written

Issue B: upsample_nearestnd symbolic output_size
--> repro RAISED InductorError: NotImplementedError: argument of type:
    <class 'sympy.core.mul.Mul'>

Issue C: ops.constant accepts a symbolic value
(no output)
--> repro RAN as written

Issue D: upsample_nearest2d_backward symbolic input_size
--> repro RAISED InductorError: TypeError: 'FloorDiv' object cannot be
    interpreted as an integer
```

The exact Issue A candidate block, including every proposed filing ratio, was
run separately:

```text
nearest 448->192: 7/192 rows differ; first [(27, 62, 63)]
nearest 384->363: 2/363 rows differ; first [(121, 127, 128)]
nearest 448->368: 9/368 rows differ; first [(23, 27, 28)]
nearest 41->82: 40/82 rows differ; first [(2, 1, 0)]
nearest-exact 384->363: 2/363 rows differ; first [(60, 63, 64)]
```

`evidence/mismatch_repro.py` also confirmed that its CPU/static/CUDA controls
remain exact and only CUDA dynamic compilation differs for its sensitive cases.

## Candidate patch guard matrix — CPU and CUDA

The sanctioned matrix toggled only the installed wheel and restored all patches
ON at completion:

```text
$ PY=/root/miniconda3/envs/dynamic/bin/python bash tools/guard_matrix.sh
############ state matrix: does each test guard its fix? ############

--- baseline: all three PRs ON (expect every test OK) ---
  forward test                                   OK
  concrete-sympy test                            OK
  one-graph test                                 OK
  constant-reject                                OK
  constant-accept                                OK
  backward test                                  OK

--- PR1 OFF (forward fix reverted) -> forward/sympy/one-graph must FAIL ---
  forward test        (expect FAILED)            FAILED (errors=1)
  concrete-sympy test (expect OK)                OK
  one-graph test      (expect FAILED)            FAILED (errors=1)
  backward test       (expect OK)                OK

--- PR2 OFF (ops.constant check reverted) -> reject must FAIL ---
  constant-reject     (expect FAILED)            FAILED (failures=1)
  constant-accept     (expect OK)                OK
  forward test        (expect OK)                OK

--- PR3 OFF (backward guard reverted) -> backward must FAIL ---
  backward test       (expect FAILED)            FAILED (errors=1)
  forward test        (expect OK)                OK

--- restore: all ON ---
pr1 = on
pr2 = on
pr3 = on
torch dir: /root/miniconda3/envs/dynamic/lib/python3.11/site-packages/torch
```

The expected failures are the negative controls that prove each test guards its
own change; they do not make the matrix a failed run.

## PR1 adversarial sweep — patches ON

```text
$ conda run -n dynamic python evidence/adversarial_pr1.py
torch 2.13.0+cu130
gpu   NVIDIA RTX PRO 6000 Blackwell Server Edition

===== cpu =====
  ok   both_sym_up                                    bit-exact
  ok   both_sym_down                                  bit-exact
  ok   mixed_dyn_dim0                                 bit-exact
  ok   mixed_dyn_dim1                                 bit-exact
  ok   scale_dim0_only                                bit-exact
  ok   scale_dim1_only                                bit-exact
  ok   scale_both                                     bit-exact
  ok   exact_both_sym                                 bit-exact
  ok   exact_ulp_448_192                              bit-exact
  ok   ulp_448_192                                    bit-exact
  ok   ulp_384_363                                    bit-exact
  ok   ulp_37_74                                      bit-exact
  ok   out_one                                        bit-exact
  ok   in_one                                         bit-exact
  ok   equal_sizes                                    bit-exact
  ok   big_upscale                                    bit-exact
  ok   1d_sym                                         bit-exact
  ok   3d_sym                                         bit-exact
  ok   fp16                                           bit-exact
  ok   bf16                                           bit-exact
  ok   noncontig                                      bit-exact

===== cuda =====
  ok   both_sym_up                                    bit-exact
  ok   both_sym_down                                  bit-exact
  ok   mixed_dyn_dim0                                 bit-exact
  ok   mixed_dyn_dim1                                 bit-exact
  ok   scale_dim0_only                                bit-exact
  ok   scale_dim1_only                                bit-exact
  ok   scale_both                                     bit-exact
  ok   exact_both_sym                                 bit-exact
  ok   exact_ulp_448_192                              bit-exact
  ok   ulp_448_192                                    bit-exact
  ok   ulp_384_363                                    bit-exact
  ok   ulp_37_74                                      bit-exact
  ok   out_one                                        bit-exact
  ok   in_one                                         bit-exact
  ok   equal_sizes                                    bit-exact
  ok   big_upscale                                    bit-exact
  ok   1d_sym                                         bit-exact
  ok   3d_sym                                         bit-exact
  ok   fp16                                           bit-exact
  ok   bf16                                           bit-exact
  ok   noncontig                                      bit-exact

============================================================
42 bit-exact, 0 problems
```

## Independent follow-up checks

The eager-only forward/backward transpose check remains inconsistent in exactly
three CUDA cases, while all CPU controls and the remaining CUDA cases pass:

```text
$ conda run -n dynamic python evidence/eager_nearest_backward_consistency.py
torch=2.13.0+cu130
gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition capability=(12, 0)
device  mode           ratio       wrong   max_abs
cpu     nearest         448->192      0/448  0
cpu     nearest         384->363      0/384  0
cpu     nearest          37->74       0/37   0
cpu     nearest          32->64       0/32   0
cpu     nearest          64->32       0/64   0
cpu     nearest-exact   448->192      0/448  0
cpu     nearest-exact   384->363      0/384  0
cpu     nearest-exact    37->74       0/37   0
cpu     nearest-exact    32->64       0/32   0
cpu     nearest-exact    64->32       0/64   0
cuda    nearest         448->192     14/448  1
cuda    nearest         384->363      4/384  1
cuda    nearest          37->74       0/37   0
cuda    nearest          32->64       0/32   0
cuda    nearest          64->32       0/64   0
cuda    nearest-exact   448->192      0/448  0
cuda    nearest-exact   384->363      4/384  1
cuda    nearest-exact    37->74       0/37   0
cuda    nearest-exact    32->64       0/32   0
cuda    nearest-exact    64->32       0/64   0

Expected gradient is the exact transpose of the observed eager forward map.
inconsistent cases=3
```

The division-rounding flag still does not reach symbolic shape division:

```text
$ conda run -n dynamic python evidence/symint_division_rounding.py
=== division_rounding=0 ===
tensor_int_div wrong=5/6 indices=[0, 1, 2, 3, 4]
tensor_int_div code=tmp4 = (tmp1 / tmp3)
symint_div wrong=191/192 first=[1, 2, 3, 4, 5, 6, 7, 8]
symint_div code=tmp0 = (ks1 / ks0).to(tl.float32)
interpolate wrong_rows=7/192 first=[27, 51, 54, 99, 102, 105, 108]

=== division_rounding=1 ===
tensor_int_div wrong=0/6 indices=[]
tensor_int_div code=tmp4 = triton.language.div_rn(tmp1, tmp3)
symint_div wrong=191/192 first=[1, 2, 3, 4, 5, 6, 7, 8]
symint_div code=tmp0 = (ks1 / ks0).to(tl.float32)
interpolate wrong_rows=7/192 first=[27, 51, 54, 99, 102, 105, 108]

Finding: the flag fixes the tensor-division control but does not
reach the symbolic shape-division path or its interpolate symptom.
```

## Upstream state refreshed on 2026-09-04

GitHub API checks recorded:

- #97135: open; `needs reproduction`, `module: autograd`, `module: nn`,
  `triaged`, `module: interpolation`.
- #175154: open; no `actionable` label.
- #185806 and #159550: closed.
- Fresh duplicate searches found no matching issue for Issues B, C, or D.

Read-only inspection of PyTorch `main` at
`4f5a382575c6e21b1a6541b8ecd697734a2469cd` (committer timestamp
2026-09-04T09:52:31Z) confirmed that:

- `upsample_nearestnd` still leaves `o_sizes` unguarded, computes
  `inv_scales = [i / o ...]`, and sends the scale to `ops.constant`;
- `upsample_nearest2d_backward` still unpacks `input_size` without guarding
  `out_h` and `out_w`; and
- `OpsWrapper` still has no explicit `constant` method enforcing the concrete
  value contract.

This was source inspection only, not a rebase, source build, lint run, or test
against current `main`.

PyTorch's current `CONTRIBUTING.md`, contribution guide, and `AI_POLICY.md`
still require the issue-first, human-reviewed process documented in
`upstream/SUBMIT.md`. Nothing was filed, commented, pushed, or opened during
this run.

## Final installed-wheel state

```text
pr1 = on
pr2 = on
pr3 = on
torch dir: /root/miniconda3/envs/dynamic/lib/python3.11/site-packages/torch
```

This installed-wheel state is convenient for local candidate testing. It is not
evidence that any patch is applied to, submitted to, or merged in PyTorch main.
