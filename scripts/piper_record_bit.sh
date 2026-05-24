#!/usr/bin/env bash
# Record one short drag-teach bit. Example:
#   ./scripts/piper_record_bit.sh 02_grasp --role pick
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
exec uv run python demos/piper_teach_demo.py record-bit "$@"
