#!/usr/bin/env bash
# Direct Kokoro TTS (no LLM, no arm) — fastest way to test speakers.
#
#   ./scripts/speak.sh "Hello from the robot."
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
exec uv run hackstorm-speak "$@"
