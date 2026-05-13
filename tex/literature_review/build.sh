#!/usr/bin/env bash
# =============================================================================
#  tex/literature_review/build.sh
#  Compile the literature review (main.tex) with pdflatex + biber.
#
#  Strategy: run from script directory with -output-directory.
#  This ensures biber finds references.bib properly, and all junk stays
#  in cache. The final PDF is MOVED here so it's not left behind.
#
#  Usage:  bash tex/literature_review/build.sh   (from any directory)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$TEX_ROOT/.latex_cache"
DOC="main"

mkdir -p "$CACHE_DIR"
cd "$SCRIPT_DIR"

echo "[build] Compiling $DOC.tex (literature review) ..."
pdflatex -interaction=nonstopmode -output-directory="$CACHE_DIR" "$DOC.tex"
biber --output-directory="$CACHE_DIR" "$CACHE_DIR/$DOC"
pdflatex -interaction=nonstopmode -output-directory="$CACHE_DIR" "$DOC.tex"
pdflatex -interaction=nonstopmode -output-directory="$CACHE_DIR" "$DOC.tex"

mv "$CACHE_DIR/$DOC.pdf" "$SCRIPT_DIR/$DOC.pdf"
echo "[build] Done  ->  tex/literature_review/$DOC.pdf"
