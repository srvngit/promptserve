#!/usr/bin/env bash
# Install pyorbbecsdk2 on the Linux robot host so DaBai DC1 depth works.
#
# pyorbbecsdk2 (PyPI: pyorbbecsdk2, import: pyorbbecsdk) is the maintained
# successor to pyorbbecsdk v1.x — Python 3.8–3.13, Linux x64 + arm64, macOS 13.2+.
#
# Run from the repo root on the robot (NOT this Mac):
#   ./scripts/install_pyorbbecsdk.sh
#
# After it finishes:
#   uv run python demos/orbbec_depth_check.py
#   HACKSTORM_CAMERA=orbbec ./scripts/piper_arm.sh --no-can-auto-init locate "strawberry"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is Linux-only. Run on the robot host (where the DaBai is plugged in)." >&2
  exit 1
fi

echo "==> Installing system USB deps"
sudo apt-get update
sudo apt-get install -y libusb-1.0-0-dev udev

echo "==> Installing pyorbbecsdk2 into the project venv"
uv pip install --upgrade pyorbbecsdk2

echo "==> Locating the udev-rule setup script bundled with the package"
PKG_DIR="$(uv run python -c 'import pyorbbecsdk, os; print(os.path.dirname(pyorbbecsdk.__file__))')"
SETUP="${PKG_DIR}/scripts/env_setup/setup_env.py"
if [[ -f "${SETUP}" ]]; then
  echo "==> Running ${SETUP} (writes /etc/udev/rules.d/*-orbbec.rules)"
  sudo "$(uv run python -c 'import sys;print(sys.executable)')" "${SETUP}"
else
  echo "WARN: ${SETUP} not found. If depth never arrives, fetch udev rules from"
  echo "      https://github.com/orbbec/pyorbbecsdk/tree/main/scripts/env_setup"
fi

echo "==> Reloading udev"
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "==> Sanity check: opening pipeline and waiting for one frame"
uv run python - <<'PY'
from pyorbbecsdk import Pipeline, Config, OBSensorType
pipe = Pipeline()
cfg = Config()
color_profile = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
depth_profile = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
cfg.enable_stream(color_profile)
cfg.enable_stream(depth_profile)
pipe.start(cfg)
try:
    fs = pipe.wait_for_frames(2000)
    assert fs is not None, "no frames within 2s — check USB and unplug/replug"
    depth = fs.get_depth_frame()
    color = fs.get_color_frame()
    assert depth is not None, "depth frame missing (firmware/permissions?)"
    assert color is not None, "color frame missing"
    print(f"OK  depth={depth.get_width()}x{depth.get_height()}  color={color.get_width()}x{color.get_height()}")
finally:
    pipe.stop()
PY

echo "✅ pyorbbecsdk2 installed. Run: uv run python demos/orbbec_depth_check.py"
