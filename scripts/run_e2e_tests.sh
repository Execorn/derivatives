#!/bin/bash
# run_e2e_tests.sh - Run Phase 7 E2E Calibration & Pricing Framework tests

# Get current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

echo "=== Starting E2E Test Suite for Phase 7 Multi-Asset Calibration ==="
echo "Project Root: $PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run tests
echo "Executing pytest on tests/test_e2e_phase7.py..."
pytest -v "$PROJECT_ROOT/tests/test_e2e_phase7.py"
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "=== SUCCESS: All Phase 7 E2E tests passed successfully! ==="
else
    echo "=== FAILURE: E2E tests failed with exit code $TEST_EXIT_CODE ==="
fi

exit $TEST_EXIT_CODE
