# `tools/`

## `state.py` — the only sanctioned way to toggle the patches

```bash
python tools/state.py status                       # ALWAYS before drawing a conclusion
python tools/state.py set pr1=on pr2=off pr3=on
```

Exact-string replacement in **both** directions, per-PR, against the *installed*
torch (located via `torch.__file__`). It **refuses** to run against a file matching
neither the stock nor the patched form, so a half-edited file cannot be silently
mangled — which is how a confusing commented-out state arose historically.

⚠️ **Never toggle state while a long suite is running.** A multi-hour run was
rendered worthless that way. For long runs, `cp -a` the venv and point the run at
the copy (verify distinct inodes).

## `guard_matrix.sh` — proves each test guards its own fix

```bash
PY=/path/to/python bash tools/guard_matrix.sh      # ~15 min on one GPU
```

Runs the six tests with each PR reverted in turn and reports which fail. **A test
that passes with its fix reverted is not a regression test**, so this is the
central claim of the whole series, not a footnote.

Expected (reproduced on both A100 and H100):

| state | forward | concrete-sympy | one-graph | const-reject | const-accept | backward |
|---|---|---|---|---|---|---|
| all ON  | OK | OK | OK | OK | OK | OK |
| pr1 off | **FAIL** | OK | **FAIL** | – | – | OK |
| pr2 off | OK | – | – | **FAIL** | OK | – |
| pr3 off | OK | – | – | – | – | **FAIL** |

It restores all-ON on exit and prints `state.py status`. `PY` defaults to `python`.
