#!/usr/bin/env bash
# Compile the Beamer presentation to PDF.
# Requires: texlive-most (Arch: sudo pacman -S texlive-most)
#
# Usage:  bash src/compile_presentation.sh
set -e
cd "$(dirname "$0")"   # run from src/
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex  # second pass for TikZ refs
echo "Done → src/presentation.pdf"
