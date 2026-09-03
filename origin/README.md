# `origin/` — where this started

The primary sources. Everything else in this repository is downstream of these
four files, so they are kept verbatim rather than summarized.

⚠️ **Internal paths and names appear here** (the venv path, the internal project
name, product names). That is fine for develop-phase work — these files are the
evidence that the bug was real and observed in production, not constructed. They are
one of the things `PUBLISH.md` says to handle when the public cut is made.

| file | what it is |
|---|---|
| `production_traceback.txt` | ⭐ **the original failure, 711 lines**, captured from the real inference worker. The `TypeError: Cannot convert expression to float` appears 8 times across the retry loop. This is the artifact `README.md` §0 and §2 are describing — the frames quoted there were read out of this file. |
| `goal_original_brief.md` | the five goals this project was built to satisfy, in the user's own words, with the traceback that prompted them. Useful for deciding whether a change is in scope. |
| `fix_md_original_writeup.md` | the first write-up of the bug (`fix.md`), before any of it was reproduced standalone. ⚠️ **Two claims in it are now known wrong:** its §4 says `fallback()` is the defensible fix (it raises `NotImplementedError`; `index_expr` is required), and its §5 lists four open items that are all resolved. Kept because the correction is part of the record. |
| `index_propagation_torch231_stock.py` | the torch-2.3.1 `index_propagation.py` as the internal repo shipped it. **Functionally identical to pristine 2.3.1** — the fix inside is commented out, so copying it over `site-packages` *reproduces* the bug. Superseded for all real use by `../patch/patch.py`, which is idempotent and refuses to touch an unrecognized file. |

## Why the traceback is worth keeping in full

It is the difference between "I read about this bug" and "this happened to a service
I was running." Three things are visible in it that a minimal repro cannot show:

1. **The retry loop** — the same compile failure recurring, which is how it surfaced
   as a sustained outage rather than one bad request.
2. **`suppress_errors`** was not masking it here, which is why it escaped as a 500
   instead of a silent fallback to eager. That distinction is what made it findable.
3. **The real frame numbers** (`lowering.py:3528`, `:3520`, `index_propagation.py:62`)
   in the actual installed torch, which is what `repro/repro.py` asserts against
   rather than matching the error string.
