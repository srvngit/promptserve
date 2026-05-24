#!/usr/bin/env bash
# Snap laptop webcam + Orbbec wrist cam; run Grounding DINO on ketchup + pink tray.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HACKSTORM_ORBBEC_COLOR_ONLY=1
export HACKSTORM_DETECT_BACKEND=grounding_dino
export HACKSTORM_DINO_DEVICE=cuda
exec uv run python demos/snap_debug.py "$@"
