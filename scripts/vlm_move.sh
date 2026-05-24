#!/usr/bin/env bash
# VLA-style loop: wrist image → Qwen plans actions → execute → fresh image → repeat.
#
# Setup:
#   export QWEN_API_KEY='sk-...'
#
# Examples:
#   ./scripts/vlm_move.sh pick up the ketchup and put it in the pink tray
#   ./scripts/vlm_move.sh --dry-run pick up the ketchup
#   ./scripts/vlm_move.sh --once move J2 to 45 degrees
#   VLM_MAX_STEPS=40 ./scripts/vlm_move.sh pick up the ketchup
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HACKSTORM_VLM_BACKEND="${HACKSTORM_VLM_BACKEND:-qwen}"
export HACKSTORM_CAMERA="${HACKSTORM_CAMERA:-orbbec}"
export HACKSTORM_ORBBEC_COLOR_ONLY="${HACKSTORM_ORBBEC_COLOR_ONLY:-1}"
export HACKSTORM_TABLE_Z_M="${HACKSTORM_TABLE_Z_M:-0.10}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-qwen-vl-max}"
export OLLAMA_GEMMA_MODEL="${OLLAMA_GEMMA_MODEL:-gemma4:e2b}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if [[ -z "${QWEN_API_KEY:-}" && -z "${DASHSCOPE_API_KEY:-}" && "${HACKSTORM_VLM_BACKEND}" == "qwen" ]]; then
  echo "vlm_move: set QWEN_API_KEY for Qwen cloud (export QWEN_API_KEY='sk-...')" >&2
  echo "  (alias: DASHSCOPE_API_KEY — DashScope hosts the Qwen API)" >&2
  echo "  or use local Ollama: HACKSTORM_VLM_BACKEND=ollama ./scripts/vlm_move.sh ..." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- "pick up the ketchup and put it in the pink tray"
fi

exec uv run python demos/vlm_move.py "$@"
