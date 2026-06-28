#!/usr/bin/env bash

set -uo pipefail

# Setup paths and variables for runner.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
LOG="$ROOT/logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
SKIP=0

# You don't have to hit enter after each step if you --skip.
[[ "${1:-}" == "-s" || "${1:-}" == "--skip" ]] && SKIP=1

mkdir -p "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1   # So tqdm progress doesn't get mangled by buffer.

log() { echo -e "$*" | tee -a "$LOG"; }

pause() {
    [[ "$SKIP" -eq 1 || ! -t 0 ]] && return
    read -rsn1 -p ">>> Press any key to continue ('q' to quit)... " key; echo
    [[ "$key" == "q" ]] && { log "Aborted."; exit 0; }
}

run() {
    pause
    log "\n=== $1 ==="
    "$PY" "$2" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]} # Capture exit code and log failure.
    [[ $rc -ne 0 ]] && { log "!! FAILED (exit $rc)"; exit "$rc"; }
}

# Welcome message.
log "CS898BA Homework 2 - Segmentation Pipeline Runner"
log "Log: $LOG"

# Create, activate, and install dependencies in python virtual environment.
if [[ ! -d "$VENV" ]]; then
    log "Creating .venv..."
    python3 -m venv "$VENV" 2>&1 | tee -a "$LOG"
fi
source "$VENV/bin/activate"

log "Installing requirements..."
"$VENV/bin/pip" install -r requirements.txt 2>&1 | tee -a "$LOG"

# Run the segmentation parts in order. Each one depends on Part 2's output.
run "Part 2 - Normalization"       part02/normalization.py
run "Part 3 - Thresholding"        part03/thresholding.py
run "Part 4 - K-Means clustering"  part04/kmeans.py
run "Part 5 - Evaluation"          part05/evaluation.py

# Done.
log "\nPipeline complete ($(date '+%F %T'))."
