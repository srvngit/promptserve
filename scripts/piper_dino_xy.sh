#!/usr/bin/env bash
# Taught trajectory + Grounding DINO: shift X/Y only. Z stays from the recording.
#
# Requires: demos/recordings/ketchup_demo.yaml (record with ./scripts/piper_record_demo.sh)
#
#   ./scripts/piper_dino_xy.sh
#   ./scripts/piper_dino_xy.sh ketchup --pick "ketchup bottle" --place "pink tray"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HACKSTORM_ORBBEC_COLOR_ONLY="${HACKSTORM_ORBBEC_COLOR_ONLY:-1}"
export HACKSTORM_DETECT_BACKEND="${HACKSTORM_DETECT_BACKEND:-grounding_dino}"
export HACKSTORM_DINO_DEVICE="${HACKSTORM_DINO_DEVICE:-cuda}"
export HACKSTORM_TABLE_Z_M="${HACKSTORM_TABLE_Z_M:-0.10}"
NAME="${1:-ketchup}"
shift || true
exec uv run python demos/piper_teach_demo.py play-adapt-xy "${NAME}" \
  --pick "ketchup bottle" --place "pink tray" "$@"
