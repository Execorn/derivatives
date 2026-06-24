#!/usr/bin/env bash
# =============================================================================
#  setup_and_run.sh  —  Arch Linux one-shot setup and pipeline launcher
#
#  1. Creates a Python venv using uv (if absent)
#  2. Installs requirements and compiles the CUDA Lifted Heston kernel
#  3. Runs the FNO surrogate pipeline (dataset -> train -> UI)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

# Always use the project .venv — do NOT use system or uv-managed interpreters
# to keep ML libraries (torch, cuda, etc.) isolated from the system.
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

SKIP_TRAIN=false
for arg in "$@"; do
  [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
done

echo "============================================================"
echo "    Master's Thesis: Lifted Heston FNO Pipeline Setup       "
echo "============================================================"

# ── 0. GCC-13 for CUDA 12 compatibility (Arch Linux) ─────────────────────────
# CUDA 12.x supports GCC up to version 13. gcc-13 is confirmed installed.
# Hardcode CC/CXX so nvcc never accidentally picks up system gcc-14/15.
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
echo ""
echo "[setup] Step 0: GCC/nvcc compatibility"
echo "  CC=$CC  CXX=$CXX"
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1)
    echo "  nvcc CUDA $CUDA_VER — OK"
else
    echo "  WARNING: nvcc not found — ensure /opt/cuda/bin is on PATH"
fi

# ── 1. Dependencies FIRST (torch headers must exist before compiling CUDA ext) ─
echo ""
echo "[setup] Step 1: Installing requirements (must precede CUDA compilation)..."
$PIP install -e .[app,api,test]

# ── 2. Compile CUDA Extension ──────────────────────────────────────────────────
echo ""
echo "[setup] Step 2: Compiling CUDA Lifted Heston kernel..."
export CUDA_HOME=/opt/cuda
export PATH=$PATH:/opt/cuda/bin
if [ -f "setup.py" ]; then
    # Must use python directly (not pip install -e .) because setup.py
    # imports torch at module level. pip's isolated build env lacks torch,
    # causing ModuleNotFoundError. The venv python has torch installed.
    $PYTHON setup.py build_ext --inplace 2>&1
    echo "[setup] CUDA extension compiled OK."
else
    echo "Warning: setup.py for CUDA extension not found!"
fi

# ── 3. FNO pipeline ───────────────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
  echo ""
  echo "[pipeline] Step 3: Deep Rough LHS Dataset Generation..."
  $PYTHON scripts/generate_dataset_heston.py || echo "Dataset generation skipped/failed"

  echo "[pipeline] Step 4: Training MFNO surrogate..."
  $PYTHON scripts/train_fno_heston.py
else
  echo "[pipeline] Training skipped (--skip-train)"
fi

# ── 4. Streamlit UI ───────────────────────────────────────────────────────────
mkdir -p "$ROOT/logs"
echo ""
echo "[ui] Starting Streamlit app at http://localhost:8501"
echo "[ui] Logs -> logs/streamlit.log"
nohup "$VENV/bin/streamlit" run src/deepvol/app/dashboard.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$ROOT/logs/streamlit.log" 2>&1 </dev/null &
echo "[ui] PID $! — stop with: kill $!"
echo "Done."
