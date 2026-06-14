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
log "CS898BA Project 1 - Full Pipeline Runner"
log "Log: $LOG"

# Create, activate, and install dependencies in python virtual environment.
if [[ ! -d "$VENV" ]]; then
    log "Creating .venv..."
    python3 -m venv "$VENV" 2>&1 | tee -a "$LOG"
fi
source "$VENV/bin/activate"

log "Installing requirements..."
"$VENV/bin/pip" install -r requirements.txt 2>&1 | tee -a "$LOG"

# Finally run the pipeline parts in order. Skipping hello world.
run "Part 2 - Basic statistics"   part02/basic_statistics.py
run "Part 2 - Color spaces"       part02/color_spaces.py
run "Part 2 - Affine transforms"  part02/affine_transforms.py
run "Part 2 - Gaussian blur"      part02/gaussian_blur.py
run "Part 3 - Edge detection"     part03/edge_detection.py

# Done.
log "\nPipeline complete ($(date '+%F %T'))."
