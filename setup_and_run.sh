#!/usr/bin/env bash
# =============================================================================
#  setup_and_run.sh  —  Linux/macOS one-shot setup and pipeline launcher
#
#  1. Creates a Python venv at .venv/ (if absent)
#  2. Installs all dependencies from src/requirements.txt
#  3. Runs the full training pipeline
#  4. Launches the Streamlit UI in the background (logs -> logs/streamlit.log)
#
#  Usage:
#      bash setup_and_run.sh              # full pipeline + UI
#      bash setup_and_run.sh --skip-train # skip training, go straight to UI
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"
STREAMLIT="$VENV/bin/streamlit"

SKIP_TRAIN=false
for arg in "$@"; do
  [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
done

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
  echo "[setup] Creating virtual environment at .venv/ ..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── 2. Dependencies ───────────────────────────────────────────────────────────
echo "[setup] Installing requirements ..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r "$ROOT/src/requirements.txt" --quiet
echo "[setup] Dependencies OK"

# ── 3. Pipeline ───────────────────────────────────────────────────────────────
cd "$ROOT"

if [ "$SKIP_TRAIN" = false ]; then
  echo ""
  echo "[pipeline] Step 1/3 — Data loader ..."
  "$PYTHON" src/data_loader.py

  echo "[pipeline] Step 2/3 — Training surrogate model (200 epochs) ..."
  "$PYTHON" src/train.py --epochs 200

  echo "[pipeline] Step 3/3 — Calibration smoke-test ..."
  "$PYTHON" src/calibrator.py
else
  echo "[pipeline] Training skipped (--skip-train)"
fi

# ── 4. Streamlit UI ───────────────────────────────────────────────────────────
mkdir -p "$ROOT/logs"
echo ""
echo "[ui] Starting Streamlit app at http://localhost:8501"
echo "[ui] Logs -> logs/streamlit.log"
nohup "$STREAMLIT" run src/app.py --server.headless true --browser.gatherUsageStats false \
  > "$ROOT/logs/streamlit.log" 2>&1 < /dev/null &
echo "[ui] PID $! — stop with: kill $!"
