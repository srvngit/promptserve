#!/usr/bin/env bash
# S'mores demo: replay taught path (no vision). Use DINO XY adapt with --adapt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
REC="${ROOT}/demos/recordings/smore_demo.yaml"
if [[ ! -f "${REC}" ]]; then
  echo "Nothing to run." >&2
  echo "  ./scripts/piper_record_demo.sh smore" >&2
  exit 1
fi
for arg in "$@"; do
  if [[ "${arg}" == "--adapt" ]]; then
    echo "[demo] taught path + DINO XY (Z locked)" >&2
    exec ./scripts/piper_dino_xy.sh smore --pick "s'mores" --place "pink tray"
  fi
done
echo "[demo] prerecorded smore path (joint replay)" >&2
exec ./scripts/piper_play_demo.sh smore "$@"
