#!/usr/bin/env bash
# =============================================================================
#  make_thesis_assets.sh  —  Generates and collects all benchmark PNGs
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"

if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

if command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
else
    PYTHON="$VENV/bin/python"
fi

echo "============================================================"
echo "    Generating Master's Thesis Assets                       "
echo "============================================================"

mkdir -p "$ROOT/thesis_images"

echo "1. Generating 3D Benchmarks (FNO vs Market)..."
$PYTHON src/benchmark_plots_fno.py

echo "2. Generating Discretization Invariance / Super-Resolution Test..."
$PYTHON src/test_resolution.py

echo "3. Generating Autograd Greeks Heatmaps..."
$PYTHON src/fno_greeks.py

echo "4. Collecting images..."
cp "$ROOT/images/ai_generated/"*.png "$ROOT/thesis_images/" 2>/dev/null || true

echo "============================================================"
echo "    Done. Assets collected in thesis_images/                "
echo "============================================================"
