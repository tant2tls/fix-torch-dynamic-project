# PUBLISH.md — the public cut

## ✅ EXECUTED 2026-08-30 — `../../../public_repo/inductor-upsample-dynamic-shapes`

The cut described below **has been made**, git-initialized (2 commits on `main`), and
zipped to `public_repo/inductor-upsample-dynamic-shapes.zip` (1.01 MB, 310 entries,
history included). Verified after extraction: leak scan clean, `repro.py` exit 2,
`tests/run_tests.py` 15/0 unpatched and 18/0 patched, `verify_fix.py` bit-exact on all
7 sizes. Re-run the extraction test if you regenerate it.

**Two deviations from the plan below, both deliberate:**

1. **`evidence/dev_harnesses/` was INCLUDED, not dropped.** `upstream/report.md` cites
   it ~20 times; excluding it would have left that many dead links for a one-directory
   saving. It leak-scanned clean.
2. **The leak scan was NOT clean on the first pass** — contrary to §2's table below.
   The develop tree scanned clean, but the *cut* had **357 hits** in raw suite logs
   (`evidence/logs/big_dynshapes_ON.log` alone: 302), all absolute venv/stdlib paths in
   pytest output, plus a product name in the traceback and a venv name in
   `RESULTS_a100.md`. All redacted to placeholders. **Lesson: scan the cut, not the
   source** — a per-path verdict table is a hypothesis until the scan runs on the
   assembled tree.

Also added in the cut, not present here: `LICENSE` (MIT), a public `.gitignore`, and
**`PUSHING.md`** (remote setup, commit-authorship check, the pre-upstream checklist, and
the claim-discipline rule).

---

**Everything below is the original plan, kept as the record of what was decided.**

**This is a plan for a future step, not a description of the current state.**

The project is in **develop phase**: everything is tracked and visible, including the
internal-facing material. That is deliberate. An earlier pass hid `origin/` and
`notebook/` behind `.gitignore` and it made the working copy harder to navigate than
the leak was worth. **Do not sanitize as you go** — do it once, on purpose, when the
public repo is actually being created.

---

## 1. What has to happen at that point

The project currently lives **inside an internal repository**, so publication is an
extraction into a fresh git repo, not a `git push` from here.

## 2. Per-path verdict (leak-scanned 2026-08-28)

Scanned for: internal employer/project names, internal repo paths, container
hostnames, product/model checkpoints, absolute NFS paths.

| path | verdict at cut time |
|---|---|
| `README.md`, `PR.md` | ✅ clean |
| `CLAUDE.md` | ✅ clean, but **rewrite or drop it** — it is written for the next Claude session, not for a reader |
| `repro/`, `patch/`, `tests/`, `demo.sh`, `requirements.txt` | ✅ clean — the 2.3.1 artifact; `torch==2.3.1` is the only dependency |
| `upstream/**` | ✅ clean — patches, PR bodies, submit docs, issues, commit messages, `final/`, `report.md` (only container names appear) |
| `evidence/**` | ✅ clean — 21 scripts, `RESULTS_a100.md`, logs, JSON digests, `probes/`, `suites/`. One exception: `evidence/dev_harnesses/` is rough and undocumented — consider dropping rather than shipping it. |
| `tools/` | ✅ clean — `guard_matrix.sh` takes `PY=` from the environment |
| **`origin/`** | ❌ **EXCLUDE.** The 711-line production traceback carries the internal venv path on every frame; the brief and the first write-up name the internal project and a commercial checkpoint. See §3 — do not just delete it. |
| **`notebook/`** | ⚠️ **judgement call.** No hard leaks, but it is a record of superseded drafts with little value to an outside reader. Simplest is to exclude it. |

The only identifying strings in the publishable set are **your own git identity** in
the three patch headers (`From: ...`), which `git am` requires in order to attribute
the commits to you. That is intentional — keep it.

## 3. The `origin/` problem, and how to solve it without lying

`README.md` §0 and §2 rest on the production traceback. Excluding `origin/` removes
the primary source for the repo's opening claim, so **do not leave the claim
unsupported** — do one of these:

- **Redact and keep it (preferred).** Strip the absolute path prefix from every
  frame. The frames themselves — `lowering.py:3528`, `:3520`,
  `index_propagation.py:62` — are the evidence; the paths are not. This keeps the
  artifact honest.
- **Or replace it** with the traceback from `repro/repro.py`, which shows the same
  frames from a public reproduction, and reword §0's HTTP-500 paragraph as context
  rather than as something the reader can verify here.

Either way, §9's *verified here / original context / not claimed* split must still be
accurate afterwards. That table is what makes the artifact credible; check it last.

## 4. Extraction

```bash
# from a directory OUTSIDE the internal repo
mkdir ~/torch-dynamic-shape-inductor-bug && cd ~/torch-dynamic-shape-inductor-bug
git init

rsync -a --exclude 'origin/' --exclude 'notebook/' \
         --exclude 'evidence/dev_harnesses/' --exclude '__pycache__/' \
         --exclude 'PUBLISH.md' --exclude 'CLAUDE.md' \
         <this-folder>/ .

# verify the exclusions took
for d in origin notebook; do test -d $d && echo "STOP: $d came along"; done

# re-scan before the first commit
grep -rniE 'zenai|genai-sd|nexfort|onediff|comfyui|dreamshaper|/prj/|/usr2/|tan-[0-9a-z]*chip|hu-tanngo' . \
  && echo "STOP: leak found" || echo "ok: clean"

git add -A && git commit -m "TorchInductor upsample dynamic-shape fixes: repro, patches, evidence"
```

`PUBLISH.md` is excluded above because it contains the internal names by construction
(the grep pattern itself), and would otherwise match its own scan.

## 5. The claim rule — the highest-risk sentence in the artifact

**Never write "fixed a bug in PyTorch" or "found an open PyTorch bug."**

> Diagnosed a compiler bug on a pinned torch version, shipped a verified local fix,
> confirmed upstream had independently closed the reachable path at two layers while
> the caller-side defect stayed latent on `main`, and prepared three upstream fixes
> plus four issue reports — each with a regression test that fails when only its own
> fix is reverted.

And for the three PRs: **not merged, not filed.** Until an issue is labelled
`actionable` and a PR is opened, the honest status is *"written, GPU-verified,
staged."*

Also disclose every time: **PRs 1 and 3 are not ATen-reachable today** — the
decompositions fire before the lowering. A reader who opens the PR will see that
disclosure there, so the repo must not contradict it.

## 6. Repo name

- `torch-dynamic-shape-inductor-bug` (or `inductor-upsample-dynamic-shapes`)
- *"Three TorchInductor fixes for symbolic shapes in nearest-neighbour upsampling
  lowerings, with the measurement harness that validates them."*

Do **not** name it after the internal project, and avoid the old internal folder name
`fix_nextfort_compile_dynamic` — "nextfort" is an internal tool name.
