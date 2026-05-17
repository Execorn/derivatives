#!/usr/bin/env bash
# =============================================================================
#  setup_and_run.sh  —  Linux/macOS one-shot setup and pipeline launcher
#
#  1. Creates a Python venv at .venv/ (if absent)
#  2. Installs all dependencies from src/requirements.txt
#  3. Runs the MLP surrogate pipeline (data loader → train → calibration test)
#  4. Generates the LSTM sequence dataset (OU trajectories → W surfaces)
#  5. Trains the LSTM temporal dynamics model
#  6. Launches the Streamlit UI in the background (logs → logs/streamlit.log)
#
#  Usage:
#      bash setup_and_run.sh              # full pipeline + UI
#      bash setup_and_run.sh --skip-train # skip all training, go straight to UI
#      bash setup_and_run.sh --skip-lstm  # skip LSTM steps only (4–5)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"
STREAMLIT="$VENV/bin/streamlit"

SKIP_TRAIN=false
SKIP_LSTM=false
for arg in "$@"; do
  [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
  [[ "$arg" == "--skip-lstm"  ]] && SKIP_LSTM=true
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

# ── 3. MLP surrogate pipeline ─────────────────────────────────────────────────
cd "$ROOT"

if [ "$SKIP_TRAIN" = false ]; then
  echo ""
  echo "[pipeline] Step 1/5 — Data loader (Total Variance transformation) ..."
  "$PYTHON" src/data_loader.py

  echo "[pipeline] Step 2/5 — Training MLP surrogate (200 epochs) ..."
  "$PYTHON" src/train.py --epochs 200

  echo "[pipeline] Step 3/5 — Calibration smoke-test (L-BFGS-B) ..."
  "$PYTHON" src/calibrator.py
else
  echo "[pipeline] Training skipped (--skip-train)"
fi

# ── 4–5. LSTM temporal dynamics pipeline ──────────────────────────────────────
if [ "$SKIP_TRAIN" = false ] && [ "$SKIP_LSTM" = false ]; then
  echo ""
  echo "[lstm] Step 4/5 — Generating LSTM sequence dataset (102,000 windows) ..."
  "$PYTHON" scripts/generate_seq_data.py

  echo "[lstm] Step 5/5 — Training LSTM dynamics model (up to 200 epochs) ..."
  "$PYTHON" src/train_seq.py --epochs 200
elif [ "$SKIP_LSTM" = true ]; then
  echo "[lstm] LSTM training skipped (--skip-lstm)"
fi

# ── 6. Streamlit UI ───────────────────────────────────────────────────────────
mkdir -p "$ROOT/logs"
echo ""
echo "[ui] Starting Streamlit app at http://localhost:8501"
echo "[ui] Logs -> logs/streamlit.log"
nohup "$STREAMLIT" run src/app.py --server.headless true --browser.gatherUsageStats false \
  > "$ROOT/logs/streamlit.log" 2>&1 < /dev/null &
echo "[ui] PID $! — stop with: kill $!"
