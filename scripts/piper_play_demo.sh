#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export HACKSTORM_ARM_PERSIST="${HACKSTORM_ARM_PERSIST:-0}"
NAME="${1:-ketchup}"
shift || true
exec uv run python demos/piper_teach_demo.py play "${NAME}" "$@"
