#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
NAME="${1:-ketchup}"
shift || true
exec uv run python demos/piper_teach_demo.py init-sequence "${NAME}" "$@"
