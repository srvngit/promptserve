#!/usr/bin/env bash
# Piper arm on Linux — motion over CAN. VLM only when you pass object names.
#
# Prove the arm moves (no camera):
#   ./scripts/piper_arm.sh smoke
#   ./scripts/piper_arm.sh home
#   ./scripts/piper_arm.sh move --x 0.25 --y 0.0 --z 0.30
#
# Manual grasp/place (no VLM):
#   ./scripts/piper_arm.sh grasp --x 0.28 --y -0.05 --z 0.12
#
# Vision pick-and-place (Orbbec + Grounding DINO):
#   ./scripts/piper_demo_ketchup.sh
#   ./scripts/piper_demo_water.sh
#   ./scripts/piper_demo_smore.sh
#   ./scripts/piper_demo_ketchup.sh --adapt   # DINO X/Y adapt (Z locked)
#   ./scripts/piper_arm.sh demo --pick "ketchup bottle" --place "pink tray"
#
# Test vision only (no arm):
#   ./scripts/piper_arm.sh --backend ollama locate "strawberry"
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# 16 GiB machines cannot load Gemma in-process; Ollama uses GPU quantization instead.
export HACKSTORM_VLM_BACKEND="${HACKSTORM_VLM_BACKEND:-ollama}"
export HACKSTORM_DETECT_BACKEND="${HACKSTORM_DETECT_BACKEND:-grounding_dino}"
export HACKSTORM_DINO_DEVICE="${HACKSTORM_DINO_DEVICE:-cuda}"
export HACKSTORM_DINO_BOX_THRESHOLD="${HACKSTORM_DINO_BOX_THRESHOLD:-0.22}"
export HACKSTORM_DINO_TEXT_THRESHOLD="${HACKSTORM_DINO_TEXT_THRESHOLD:-0.18}"
export HACKSTORM_ORBBEC_COLOR_ONLY="${HACKSTORM_ORBBEC_COLOR_ONLY:-1}"
export HACKSTORM_TABLE_Z_M="${HACKSTORM_TABLE_Z_M:-0.10}"
# Force Orbbec RGB-D — V4L2/webcam has no depth and the IK path requires real
# Z. Override only if you know what you're doing.
export HACKSTORM_CAMERA="${HACKSTORM_CAMERA:-orbbec}"

# Like demos/piper_move_joint.py: default motion when no subcommand is given.
has_cmd=false
for arg in "$@"; do
  case "${arg}" in
    -h|--help) exec uv run hackstorm-piper-arm "$@" ;;
    home|pose|joints|smoke|gripper-open|gripper-close|locate|grasp|place|move|demo|calibrate-wrist)
      has_cmd=true
      break
      ;;
  esac
done
if [[ $# -eq 0 || "${has_cmd}" == false ]]; then
  echo "piper_arm: no command — running smoke (home → lift → gripper → home)" >&2
  set -- "$@" smoke
fi

exec uv run hackstorm-piper-arm "$@"
