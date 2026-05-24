#!/usr/bin/env bash
# Strands agent with ONLY the speak tool (Kokoro + Qwen tool calling).
#
# Step 1 — direct TTS, no API key:
#   ./scripts/speak.sh "Testing speakers"
#
# Step 2 — Strands speak tool wiring, no API key:
#   ./scripts/speak_agent.sh --verify-tools
#
# Step 3 — full speak agent (needs Qwen):
#   export QWEN_API_KEY='sk-...'
#   ./scripts/speak_agent.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}"
export QWEN_AGENT_MODEL="${QWEN_AGENT_MODEL:-qwen-plus}"
exec uv run python demos/speak_agent.py "$@"
