#!/usr/bin/env bash
# Water demo: replay taught path (no vision). Use DINO XY adapt with --adapt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
REC="${ROOT}/demos/recordings/water_demo.yaml"
if [[ ! -f "${REC}" ]]; then
  echo "Nothing to run." >&2
  echo "  ./scripts/piper_record_demo.sh water" >&2
  exit 1
fi
for arg in "$@"; do
  if [[ "${arg}" == "--adapt" ]]; then
    echo "[demo] taught path + DINO XY (Z locked)" >&2
    exec ./scripts/piper_dino_xy.sh water --pick "water bottle" --place "pink tray"
  fi
done
echo "[demo] prerecorded water path (joint replay)" >&2
exec ./scripts/piper_play_demo.sh water "$@"
