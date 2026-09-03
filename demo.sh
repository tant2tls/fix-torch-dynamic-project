#!/usr/bin/env bash
# The whole story in one command: fail -> explain -> fix -> verify.
#
#   ./demo.sh              # auto-detect device
#   ./demo.sh cpu          # force CPU (works on a laptop, no GPU needed)
#
# Restores whatever patch state it started in, so it is safe to re-run.

set -u -o pipefail

DEVICE="${1:-auto}"
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

rule() { printf '\n%s\n' "══════════════════════════════════════════════════════════════════════════════"; }
step() { rule; printf 'STEP %s: %s\n' "$1" "$2"; rule; }

# Remember the starting state so we can restore it on exit.
START_STATE="$("$PY" patch/patch.py status | awk '/^state/ {print $3}')"
restore() {
  rule
  if [ "$START_STATE" = "PRISTINE" ]; then
    printf 'Restoring the original (unpatched) torch, as found at start.\n'
    "$PY" patch/patch.py revert
  else
    printf 'Leaving the patch applied (it was already applied at start).\n'
  fi
}
trap restore EXIT

step 0 "Environment and current patch state"
"$PY" patch/patch.py status

step 1 "Reproduce the bug (expect FAILURE)"
"$PY" patch/patch.py revert
"$PY" repro/repro.py --device "$DEVICE"
RC=$?
if [ "$RC" -ne 2 ]; then
  printf '\nUNEXPECTED: repro.py exited %s (wanted 2 = bug reproduced).\n' "$RC"
  printf 'The bug did not reproduce. Check torch version (needs 2.3.1).\n'
  exit 1
fi

step 2 "Why a naive repro does NOT fail (ingredient ablation)"
"$PY" repro/ablation.py --device "$DEVICE"

step 3 "The mechanism: which grad context lets the op reach Inductor"
"$PY" repro/why_it_survives.py --device "$DEVICE"

step 4 "The same failure in a realistic UNet-shaped decoder"
"$PY" repro/unet_like.py --device "$DEVICE"

step 5 "Apply the one-hunk fix"
"$PY" patch/patch.py apply
printf '\nThe diff:\n\n'
sed -n '/^---/,$p' patch/index_propagation.patch

step 6 "Re-compile the same operation (expect SUCCESS)"
"$PY" repro/repro.py --device "$DEVICE"
RC=$?
if [ "$RC" -ne 0 ]; then
  printf '\nUNEXPECTED: repro.py exited %s (wanted 0 = compiled).\n' "$RC"
  exit 1
fi

step 7 "Verify numerics: compiled output vs eager, bit-exact"
"$PY" repro/verify_fix.py --device "$DEVICE"
RC=$?
if [ "$RC" -ne 0 ]; then
  printf '\nUNEXPECTED: verify_fix.py exited %s (wanted 0).\n' "$RC"
  exit 1
fi

step 8 "The UNet decoder, now compiling"
"$PY" repro/unet_like.py --device "$DEVICE"

step 9 "Test suite, with the fix installed"
"$PY" tests/run_tests.py

rule
printf 'DEMO COMPLETE\n'
printf '  bug reproduced -> root cause shown -> fix applied -> numerics verified\n'
rule
