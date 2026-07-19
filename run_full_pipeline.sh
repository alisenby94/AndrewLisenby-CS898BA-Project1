#!/usr/bin/env bash
# Set up the environment (if needed) and run HW3 parts 2-5 in order.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
PY="$VENV/bin/python"
export PYTHONUNBUFFERED=1

# Create venv and install dependencies.
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -r requirements.txt

# torch/torchvision need the CUDA 12.8 nightly wheels for the RTX 5090 (sm_120).
if ! "$PY" -c "import torch, torchvision" 2>/dev/null; then
    "$VENV/bin/pip" install --pre --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/nightly/cu128
    "$VENV/bin/pip" install --pre --no-cache-dir --no-deps torchvision \
        --index-url https://download.pytorch.org/whl/nightly/cu128
fi

# Extract the fish dataset if it hasn't been unpacked yet.
if ! find assets/Fish -name '*.jpg' 2>/dev/null | grep -q .; then
    "$PY" -c "import py7zr; py7zr.SevenZipFile('assets/Fish.7z','r').extractall('assets/')"
fi

# Run the pipeline parts in order.
"$PY" part02/data.py
"$PY" part03/train.py
"$PY" part04/tune.py
"$PY" part05/evaluate.py
