#!/usr/bin/env bash
# Ketchup demo: replay taught path (no vision). Use DINO XY adapt with --adapt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
REC="${ROOT}/demos/recordings/ketchup_demo.yaml"
if [[ ! -f "${REC}" ]]; then
  echo "Nothing to run." >&2
  echo "  ./scripts/piper_record_demo.sh ketchup" >&2
  exit 1
fi
for arg in "$@"; do
  if [[ "${arg}" == "--adapt" ]]; then
    echo "[demo] taught path + DINO XY (Z locked)" >&2
    exec ./scripts/piper_dino_xy.sh ketchup --pick "ketchup bottle" --place "pink tray"
  fi
done
echo "[demo] prerecorded ketchup path (joint replay)" >&2
exec ./scripts/piper_play_demo.sh ketchup "$@"
