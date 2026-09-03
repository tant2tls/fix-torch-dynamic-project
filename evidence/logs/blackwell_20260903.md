# Blackwell verification — 2026-09-03

This is a fresh verification on an RTX PRO 6000 Blackwell Server Edition. It
extends the earlier A100 and H100 evidence; it does not replace it.

## Claim boundary

This GPU is compute capability 12.0, but it is not a B200. Correctness results
can be described as Blackwell/SM 12.0 results. Performance results must be
reported as RTX PRO 6000 results and must not be generalized to B200.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| compute capability | 12.0 (sm_120) |
| memory | 97,887 MiB reported by nvidia-smi |
| driver | 580.126.09 |
| Python | 3.11.16, Conda environment dynamic |
| PyTorch | 2.13.0+cu130 |
| PyTorch git | cf30153c4c131c8164ee7798e5022d810682e2cb |
| Triton | 3.7.1 |
| NumPy | 2.4.6 |

The installed wheel reports sm_120 in torch.cuda.get_arch_list() and sees the
device as capability (12, 0).

## Stock behavior: all candidate patches OFF

repro/check_upstream.py passes on CPU: the original PyTorch 2.3.1 reachable
crash path compiles on 2.13.0 and is bit-exact.

evidence/verify_issue_repros.py gives:

- Issue A: wrong pixels on CUDA at 448→192 (7/192 rows), 384→363
  (2/363), 37→74 (36/74), and nearest-exact 384→363 (2/363).
- Issue B: direct forward lowering raises the documented Mul argument
  NotImplementedError.
- Issue C: the permissive symbolic ops.constant call runs as written.
- Issue D: direct backward lowering raises the documented FloorDiv TypeError.

evidence/mismatch_repro.py separates the device and compilation modes:

| path | result |
|---|---|
| CPU, static | exact for 6/6 cases |
| CPU, dynamic | exact for 6/6 cases |
| CUDA, static | exact for 6/6 cases |
| CUDA, dynamic | wrong for the four ULP-sensitive cases; exact for 32→64 and 32→100 |

This confirms that Issue A is dynamic-CUDA-specific on this host. The 37→74
case matches the earlier A100 result, not the literal-divisor H100 result; it
remains unsuitable as the headline repro because its result depends on how the
divisor survives lowering.

## Candidate patches: CPU and CUDA

The second, dependency-complete run of tools/guard_matrix.sh behaves exactly
as designed:

| state | forward | concrete SymPy | one graph | constant reject | constant accept | backward |
|---|---|---|---|---|---|---|
| all ON | OK | OK | OK | OK | OK | OK |
| PR1 OFF | fails | OK | fails | — | — | OK |
| PR2 OFF | OK | — | — | fails | OK | — |
| PR3 OFF | OK | — | — | — | — | fails |

The first attempted matrix run was invalid because the fresh environment lacked
PyTorch's expecttest test dependency; all cases stopped during the same import.
After installing the documented test dependencies, the matrix above passed. The
environment was restored to PR1=ON, PR2=ON, PR3=ON.

evidence/adversarial_pr1.py also passes 42/42 configurations: 21 CPU and 21
CUDA, all bit-exact.

## Same-hardware no-op comparison

Artifacts:

- noop_blackwell_20260903_ON.json
- noop_blackwell_20260903_OFF.json

The two 63-case digests have identical case sets:

| comparison | differences |
|---|---:|
| eager result hash | 0/63 |
| compiled result hash | 0/63 |
| normalized generated-code hash | 0/63 |

Each state has the same four compiled-vs-eager mismatches, all belonging to
Issue A. Therefore the three candidate patches do not cause or alter Issue A on
this GPU.

## Controlled division benchmark

Artifact: perf_blackwell_20260903.json.

The harness uses CUDA events, 15 A/B/A/B alternations, 100 launches per sample,
and an A-vs-A control first. Every control interval straddles 1.0.

| shape | median div_rn / truediv |
|---|---:|
| 1×3×64×64 | 0.9816 |
| 1×3×256×256 | 0.9724 |
| 8×64×128×128 | 1.0008 |
| 4×128×256×256 | 1.0000 |

No slowdown is resolved on this RTX PRO 6000 run; the worst median is 1.0008×.
The two small cases are noisy even in the control, so they cannot support a
speedup claim. This result does not override the controlled A100 result
(approximately 2% slower at large shapes) and says nothing about B200.

## Independent follow-up: a SymInt eager-numerics gap

evidence/symint_division_rounding.py tests the existing
TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING compatibility flag with a positive
control.

With the flag OFF, ordinary integer-tensor division differs from eager in 5/6
chosen values. With the flag ON, it emits triton.language.div_rn and becomes
exact. The same flag does not reach an equivalent division of two dynamic shape
values:

| flag | integer-tensor division | generic SymInt division | interpolate 448→192 |
|---|---|---|---|
| OFF | 5/6 wrong | 191/192 wrong | 7/192 wrong rows |
| ON | 0/6 wrong | 191/192 wrong | 7/192 wrong rows |

The generic repro contains no interpolation or custom operator. Its generated
line stays (ks1 / ks0).to(tl.float32) in both states.

The code-path explanation is visible in both the 2.13 wheel and current
PyTorch main: ordinary tensor division goes through TritonOverrides.truediv,
where the flag is checked; symbolic shape division is represented as
IntTrueDiv and printed by TritonPrinter._print_IntTrueDiv, which emits plain
division and never consults the flag. This makes Issue A one user-visible
symptom of a broader symbolic-scalar numerics gap.

This is a new investigation lead, not a prepared upstream fix. Before proposing
a patch, it needs a prior-art search by exact mechanism, validation on a current
main build, large-shape and gradient tests, and a performance review of the
symbolic-scalar printer path.

## Independent follow-up: eager CUDA backward is not the transpose

`evidence/eager_nearest_backward_consistency.py` derives the expected gradient
from the source indices selected by eager forward itself. For `y = resize(x)`,
`y.sum().backward()` must count how many outputs selected each input. The test
therefore avoids choosing an external definition of nearest-neighbor rounding.

| device | mode | ratio | wrong input gradients | max abs |
|---|---|---:|---:|---:|
| CPU | nearest | 448→192 | 0/448 | 0 |
| CUDA | nearest | 448→192 | 14/448 | 1 |
| CUDA | nearest | 384→363 | 4/384 | 1 |
| CUDA | nearest-exact | 384→363 | 4/384 | 1 |
| CPU/CUDA controls | both modes | 37→74, 32→64, 64→32 | 0 | 0 |

All ten CPU cases pass. The remaining stable CUDA cases pass. These failures
are exact count differences of one, not floating-point accumulation noise.

Current CUDA source explains the boundary mismatch. Forward computes one
float32 `input_size / output_size` scale and maps each output with
`floorf(output_index * scale)`. Backward independently computes a float32
`output_size / input_size` scale and uses `ceilf(input_index * scale)` to form
the range of output gradients to sum. Independently rounded reciprocal scales
are not guaranteed to describe inverse partitions at ULP-sensitive boundaries.

This eager bug predates and blocks the dynamic-forward repair. It matches the
still-open PyTorch issue #97135, which is currently labelled `needs reproduction`.
The appropriate external step is a concise, human-authored reproduction comment
there, followed by maintainer agreement on coordinated forward/backward semantics
before preparing code.

A symbolic-printer prototype can make the dynamic compiled forward exact, but it
also exposes one additional compiled-vs-eager gradient mismatch. It is useful as
a localization experiment, not an upstream-ready fix and not part of PRs 1–3.
