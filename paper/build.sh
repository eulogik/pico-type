#!/usr/bin/env bash
# Compile the paper for arXiv submission.
set -euo pipefail

# Option 1: tectonic (recommended, no TeX distribution needed)
if command -v tectonic &>/dev/null; then
    echo "==> Compiling with tectonic..."
    tectonic main.tex
    echo "==> paper/main.pdf generated."
    exit 0
fi

# Option 2: pdflatex (two passes for cross-references)
if command -v pdflatex &>/dev/null; then
    echo "==> Compiling with pdflatex (2 passes)..."
    pdflatex main.tex
    pdflatex main.tex
    echo "==> paper/main.pdf generated."
    exit 0
fi

echo "ERROR: Install tectonic (brew install tectonic) or pdflatex (MacTeX/BasicTeX)."
exit 1
