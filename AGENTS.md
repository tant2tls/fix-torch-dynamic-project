# Project guidance

This repository documents and validates TorchInductor dynamic-shape bugs and
candidate upstream PyTorch fixes. Treat reproducibility and claim accuracy as
part of the implementation, not as cleanup after it.

## Canonical environments

- Use the Conda environment `dynamic` for current/upstream work. Invoke it as
  `conda run -n dynamic ...`; do not assume the caller activated it.
- `dynamic` is reconstructed with `requirements-upstream.txt` and is expected
  to contain PyTorch 2.13.0 with CUDA 13.0 support.
- The historical teaching artifact under `repro/`, `patch/`, and `tests/`
  requires PyTorch 2.3.1. Do not downgrade `dynamic` to run it; use a separate
  compatibility environment built from `requirements.txt`.
- CPU experiments for upstream work run in `dynamic` with `--device cpu` or the
  script's CPU branch. A CUDA-enabled wheel running on CPU is acceptable unless
  the experiment explicitly compares wheel builds.

Before every result, record the actual environment. At minimum capture Python,
PyTorch, PyTorch git revision, CUDA build, driver, GPU name, and compute
capability. Hardware must be probed in the current session; never inherit the
GPU identity from an earlier log.

## Required reading

Read `CLAUDE.md` for the complete investigation state and failure history. Use:

- `README.md` for the professor-facing narrative.
- `upstream/README.md` and `upstream/SUBMIT*.md` for PR preparation.
- `evidence/README.md` before selecting an experiment.
- `PUBLISH.md` before making a public/profile cut.

Do not remove or sanitize `origin/` or `notebook/` during development. The
public cut is a separate, deliberate step.

## Experiment discipline

1. Run `conda run -n dynamic python tools/state.py status` before interpreting
   any upstream result. `tools/state.py` is the only sanctioned patch toggle.
2. Never toggle patches while a suite is running. Use a separate immutable
   environment for long comparisons.
3. Use a fresh Inductor cache for lowering instrumentation. A warm cache can
   turn a real measurement into zero lowering calls.
4. Run an A-vs-A control before A-vs-B performance or generated-code studies.
5. Compare result hashes for correctness. Raw generated-code hashes are noisy
   unless the harness explicitly normalizes them.
6. Match extracted PyTorch tests to the installed wheel's git revision, not to
   an unrelated `main` checkout.
7. Preserve raw outputs, including invalid runs, and label why an invalid run
   cannot support a claim.

For the current Blackwell host, write new measurements to a separately dated
log. Do not rewrite the existing A100/H100 evidence. Report it as Blackwell
SM 12.0 evidence and do not silently generalize it to B200: RTX PRO 6000
Blackwell and B200 share an architecture generation but are different products
with different performance characteristics.

## Verification order

For a fresh upstream environment:

```bash
conda run -n dynamic python tools/state.py status
conda run -n dynamic python repro/check_upstream.py
conda run -n dynamic python evidence/verify_issue_repros.py
```

`evidence/verify_issue_repros.py` requires all three candidate patches OFF.
The full guard matrix mutates the installed PyTorch package and restores all
patches ON at the end; run it only when that state transition is intended:

```bash
PY=/root/miniconda3/envs/dynamic/bin/python bash tools/guard_matrix.sh
```

Treat the stored patches as research artifacts, not branches ready to upload.
First file the human-authored issue and wait for a maintainer to label it
`actionable`. Only then prepare a fresh one-concern branch against current
PyTorch `main`, run focused tests plus lint/type/pre-commit checks in a real
PyTorch checkout, and re-run the guard matrix on CPU and CUDA. The planned
order is PR3, PR1, PR2.

## Claim boundaries

- The PyTorch 2.3.1 incident was a compile-time crash, not wrong results, and
  its reachable path is already fixed in PyTorch 2.4 and newer.
- Issue A is a separate live CUDA wrong-results finding on the tested current
  release. It has forward prototypes but no upstream-ready fix.
- Candidate PR1 and PR3 lowerings are not ATen-reachable today because
  decompositions run first. State that limitation explicitly.
- Nothing is merged or submitted until the upstream links prove otherwise.
- Attach hardware, software version, patch state, command, and raw log to every
  quantitative statement. Never claim B200 performance from A100, H100, or RTX
  PRO 6000 Blackwell data.

Do not paste AI-generated fix explanations into PyTorch issue bodies. The human
author must write the issue in their own words, understand every proposed code
change, and personally approve the exact content before any external action.
Never post raw or lightly reviewed assistant output and never autonomously file,
comment, push, or open a PR. If AI-generated material is used on GitHub, follow
PyTorch's current policy: disclose and contain it, and add human commentary that
explains its relevance.
