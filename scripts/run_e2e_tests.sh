#!/bin/bash
# run_e2e_tests.sh - Run DeepVol End-to-End Calibration, Pricing & Governance Suites

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

echo "=== DeepVol Production End-to-End Test Suite ==="
echo "Project Root: $PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

E2E_TARGETS=(
    "$PROJECT_ROOT/tests/test_e2e_phase7.py"
    "$PROJECT_ROOT/tests/test_e2e_projection.py"
    "$PROJECT_ROOT/tests/test_api_autocall.py"
    "$PROJECT_ROOT/tests/test_autocall_guardian.py"
)

echo "Executing pytest on E2E targets..."
pytest -v "${E2E_TARGETS[@]}"
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "=== SUCCESS: All End-to-End tests passed successfully! ==="
else
    echo "=== FAILURE: E2E tests failed with exit code $TEST_EXIT_CODE ==="
fi

exit $TEST_EXIT_CODE
