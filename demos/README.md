# Demos

## System audio (mic + speakers)

All voice I/O goes through the host's default audio devices via `tts-utils` (`sounddevice`).

### TTS → speakers

```bash
uv run python demos/tts_speak.py "Hello from hackstorm"
# or
uv run python main.py "Hello from hackstorm"
```

On macOS, `auto` uses the built-in `say` command. Optional Piper:

```bash
export TTS_BACKEND=piper
export PIPER_EXECUTABLE=/path/to/piper
export PIPER_MODEL=/path/to/en_US-lessac-medium.onnx
```

Dry-run (synthesize only, no playback):

```bash
uv run python demos/tts_speak.py --dry-run "test"
```

### Mic → speakers (record/playback test)

```bash
uv run python demos/record_playback.py
uv run python demos/record_playback.py --seconds 5
```

## AgileX Piper arm (piper_sdk)

Requires a CAN interface to the arm (typically `can0` at 1 Mbps) and the optional Python extra:

```bash
uv sync --extra piper

# macOS only: USB dongle via IORegistry
uv run python demos/piper_check_usb.py

# Linux: check adapter + SocketCAN, optionally bring up can0
uv run python demos/piper_move_joint.py --check
sudo uv run python demos/piper_move_joint.py --up --check

uv run python demos/piper_move_joint.py --dry-run
uv run python demos/piper_move_joint.py
PIPER_CAN=can0 uv run python demos/piper_move_joint.py --joints-deg 0 30 0 0 0 0
```

### Piper arm only (no voice / agent / web UI)

**Prove the arm moves** (no camera, no VLM):

```bash
./scripts/piper_arm.sh --no-can-auto-init smoke
./scripts/piper_arm.sh --no-can-auto-init home
./scripts/piper_arm.sh --no-can-auto-init move --x 0.25 --y 0.0 --z 0.30
```

Manual grasp (XYZ in meters, skips VLM):

```bash
./scripts/piper_arm.sh --no-can-auto-init grasp --x 0.28 --y -0.05 --z 0.12
```

VLM pick-and-place (robot host — **webcam default**, fixed table Z):

```bash
export HACKSTORM_CAMERA=/dev/video4   # default if omitted
uv run hackstorm-webcam-capture /tmp/test.png   # verify camera
./scripts/piper_arm.sh --no-can-auto-init --camera /dev/video4 demo --pick "strawberry" --place "pink box"
```

Orbbec RGB-D (optional): `--camera orbbec` or `HACKSTORM_CAMERA=orbbec`

Or: `uv run hackstorm-piper-arm --help`
