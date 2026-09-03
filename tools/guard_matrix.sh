#!/usr/bin/env bash
# Prove each test guards its own fix: run the new tests with each PR OFF and
# confirm the right ones fail. A test that passes with its fix reverted is not a
# regression test.
set -u
# Run from this folder. Point PY at a python whose torch is the one to patch.
cd "$(dirname "$0")"
PY="${PY:-python}"
T=TestCustomLowering
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/tmp/tritoncache_gm
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductorcache_gm

FWD="$T.test_upsample_nearestnd_symbolic_output_size"
SYMP="$T.test_upsample_nearestnd_concrete_sympy_output_size"
ONEG="$T.test_upsample_nearestnd_symbolic_output_size_one_graph"
REJ="$T.test_ops_constant_rejects_symbolic_value"
ACC="$T.test_ops_constant_accepts_concrete_values"
BWD="$T.test_upsample_nearest2d_backward_symbolic_input_size"

run() { # run <label> <tests...>
  local label="$1"; shift
  local out rc
  out=$(timeout 1800 $PY ../evidence/test_custom_lowering.py "$@" 2>&1)
  rc=$?
  local summary
  summary=$(echo "$out" | grep -E "^(OK|FAILED|ERROR)" | tail -1)
  printf '  %-46s %s\n' "$label" "${summary:-rc=$rc}"
}

echo "############ state matrix: does each test guard its fix? ############"
echo
echo "--- baseline: all three PRs ON (expect every test OK) ---"
$PY ./state.py set pr1=on pr2=on pr3=on >/dev/null
run "forward test"        "$FWD"
run "concrete-sympy test" "$SYMP"
run "one-graph test"      "$ONEG"
run "constant-reject"     "$REJ"
run "constant-accept"     "$ACC"
run "backward test"       "$BWD"

echo
echo "--- PR1 OFF (forward fix reverted) -> forward/sympy/one-graph must FAIL ---"
$PY ./state.py set pr1=off pr2=on pr3=on >/dev/null
run "forward test        (expect FAILED)" "$FWD"
run "concrete-sympy test (expect OK)"     "$SYMP"
run "one-graph test      (expect FAILED)" "$ONEG"
run "backward test       (expect OK)"     "$BWD"

echo
echo "--- PR2 OFF (ops.constant check reverted) -> reject must FAIL ---"
$PY ./state.py set pr1=on pr2=off pr3=on >/dev/null
run "constant-reject     (expect FAILED)" "$REJ"
run "constant-accept     (expect OK)"     "$ACC"
run "forward test        (expect OK)"     "$FWD"

echo
echo "--- PR3 OFF (backward guard reverted) -> backward must FAIL ---"
$PY ./state.py set pr1=on pr2=on pr3=off >/dev/null
run "backward test       (expect FAILED)" "$BWD"
run "forward test        (expect OK)"     "$FWD"

echo
echo "--- restore: all ON ---"
$PY ./state.py set pr1=on pr2=on pr3=on >/dev/null
$PY ./state.py status
