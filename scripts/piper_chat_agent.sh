#!/usr/bin/env bash
# Strands agent: Qwen tool calling + Kokoro speak + prerecorded arm replays.
#
#   export QWEN_API_KEY='sk-...'
#   ./scripts/piper_chat_agent.sh
#
# Tool-calling uses qwen-plus (NOT qwen-vl-max). Vision at startup uses qwen-vl-max.
#
#   ./scripts/piper_chat_agent.sh --verify-tools          # test speak tool wiring
#   ./scripts/piper_chat_agent.sh --one-shot "say hello"  # single LLM turn
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HACKSTORM_ORBBEC_COLOR_ONLY="${HACKSTORM_ORBBEC_COLOR_ONLY:-1}"
export HACKSTORM_CAMERA="${HACKSTORM_CAMERA:-orbbec}"
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}"
export QWEN_AGENT_MODEL="${QWEN_AGENT_MODEL:-qwen-plus}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-qwen-vl-max}"
export HACKSTORM_ARM_PERSIST="${HACKSTORM_ARM_PERSIST:-1}"

if [[ -z "${QWEN_API_KEY:-}" && -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "piper_chat_agent: set QWEN_API_KEY (export QWEN_API_KEY='sk-...')" >&2
  exit 1
fi

exec uv run python demos/piper_chat_agent.py "$@"
