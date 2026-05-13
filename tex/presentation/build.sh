#!/usr/bin/env bash
# =============================================================================
#  tex/presentation/build.sh
#  Compile the defense presentation (presentation.tex) to PDF.
#
#  Strategy: run pdflatex with -output-directory from the script directory.
#  This ensures all auxiliary files land in cache, and any relative paths
#  (like images) resolve correctly. The final PDF is then MOVED here.
#
#  Requirements: pdflatex, beamer, pgfplots, booktabs, fontawesome5
#  Install on Arch:  sudo pacman -S texlive-most
#
#  Usage:  bash tex/presentation/build.sh   (from any directory)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$TEX_ROOT/.latex_cache"
DOC="presentation"

mkdir -p "$CACHE_DIR"
cd "$SCRIPT_DIR"

echo "[build] Compiling $DOC.tex (defense slides) ..."
# Two passes for TikZ cross-references and ToC
pdflatex -interaction=nonstopmode -output-directory="$CACHE_DIR" "$DOC.tex"
pdflatex -interaction=nonstopmode -output-directory="$CACHE_DIR" "$DOC.tex"

# MOVE the PDF back to the document directory so it isn't left in cache
mv "$CACHE_DIR/$DOC.pdf" "$SCRIPT_DIR/$DOC.pdf"
echo "[build] Done  ->  tex/presentation/$DOC.pdf"
