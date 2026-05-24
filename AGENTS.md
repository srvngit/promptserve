## Learned User Preferences

- Target Linux SocketCAN for AgileX Piper demos (`demos/piper_move_joint.py`); macOS IORegistry checks stay in `demos/piper_check_usb.py` only.
- Run Piper arm demos on the robot Linux host (e.g. `/home/arjun/hackstorm/hackstorm/`), not only the macOS dev workspace path.
- Use UV for Python dependencies (`uv sync`, optional extras like `--extra piper`); do not add or rely on `requirements.txt`.
- When `can0` is already UP via `ip link`, run Piper with `--no-can-auto-init` and skip `--up` to avoid RTNETLINK "Device or resource busy" errors.
- For PCIe-to-CAN or serial CAN adapters on Linux, pass `judge_flag=False` to `CreateCanBus` so piper_sdk does not mis-detect the SocketCAN module.

## Learned Workspace Facts

- Voice I/O uses system mic/speakers via `tts-utils` (`sounddevice` + macOS `say` / Linux `aplay` fallback). No external audio hardware.
- hackstorm integrates AgileX Piper control (piper_sdk over SocketCAN) and `tts-utils` for TTS, playback, and mic capture.
- piper_sdk is vendored at `piper_sdk/`; Piper CAN expects 1 Mbps (`bitrate 1000000`).
- Linux sysfs `/sys/class/net/can0/type` is ARPHRD_CAN numeric `280`, not the string `"can"`.
- `can0` UP via `ip link` only means the adapter is ready; it does not prove the arm is on the bus. Enable timeouts often indicate wiring, power, e-stop, or silent `candump`.
- When `can_auto_init=False`, piper_sdk requires `CreateCanBus()` before `ConnectPort(can_init=False)`.
- RTNETLINK "Device or resource busy" usually means `can0` is already UP or another process holds the CAN socket; stop prior runs and use `--no-can-auto-init` instead of re-running `ip link`.
