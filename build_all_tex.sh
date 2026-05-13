#!/usr/bin/env bash
# =============================================================================
#  build_all_tex.sh  —  Compile all LaTeX documents in the project
#
#  Runs the build script for each document in sequence.
#  PDFs are placed next to their source files; junk stays in tex/.latex_cache/
#
#  Usage:  bash build_all_tex.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================="
echo "  Building all LaTeX documents"
echo "================================================================="

echo ""
echo "--- Literature Review ---"
bash "$ROOT/tex/literature_review/build.sh"

echo ""
echo "--- Defense Presentation ---"
bash "$ROOT/tex/presentation/build.sh"

echo ""
echo "================================================================="
echo "  All documents compiled successfully."
echo "  literature_review -> tex/literature_review/main.pdf"
echo "  presentation      -> tex/presentation/presentation.pdf"
echo "================================================================="
