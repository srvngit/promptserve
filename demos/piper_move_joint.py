#!/usr/bin/env python3
"""
Move AgileX Piper arm joints over CAN using piper_sdk (Linux SocketCAN).

Prerequisites:
  - Piper powered, e-stop clear, USB CAN adapter bound to gs_usb (candleLight).
  - SocketCAN interface up at 1 Mbps, e.g.::

      sudo ip link set can0 down
      sudo ip link set can0 type can bitrate 1000000
      sudo ip link set can0 up

  - ``uv sync --extra piper``

Examples:
  uv run python demos/piper_move_joint.py --check
  uv run python demos/piper_move_joint.py --up
  uv run python demos/piper_move_joint.py
  PIPER_CAN=can0 uv run python demos/piper_move_joint.py --joints-deg 0 30 0 0 0 0
  uv run python demos/piper_move_joint.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# millidegrees per radian (1000 * 180 / pi), matches piper_sdk demo/V1/piper_joint_ctrl.py
_RAD_TO_SDK = 1000.0 * 180.0 / math.pi

# linux/if_arp.h ARPHRD_CAN — sysfs .../type is numeric, not the string "can"
_ARPHRD_CAN = 280

# AgileX / candleLight / GS-USB adapters commonly bundled with Piper
_KNOWN_ADAPTERS: tuple[tuple[int, int, str], ...] = (
    (0x1D50, 0x606F, "GS-USB / candleLight (AgileX Piper kit)"),
    (0x1209, 0x2323, "candleLight (alternate PID)"),
    (0x1CD2, 0x606F, "CES CANext FD"),
    (0x16D0, 0x10B8, "ABE CANdebugger FD"),
)


@dataclass(frozen=True)
class UsbAdapter:
    bus_device: str
    id_vendor: int
    id_product: int
    description: str

    @property
    def vid_pid(self) -> str:
        return f"{self.id_vendor:04x}:{self.id_product:04x}"

    @property
    def known_label(self) -> str | None:
        for vid, pid, label in _KNOWN_ADAPTERS:
            if self.id_vendor == vid and self.id_product == pid:
                return label
        return None


@dataclass(frozen=True)
class CanInterfaceStatus:
    name: str
    exists: bool
    operstate: str | None
    type: str | None
    bitrate: int | None
    tx_error: int | None
    rx_error: int | None
    rx_packets: int | None
    tx_packets: int | None

    @property
    def is_up(self) -> bool:
        return self.operstate == "up"

    @property
    def is_can(self) -> bool:
        if self.type is None:
            return False
        if self.type.lower() == "can":
            return True
        try:
            return int(self.type) == _ARPHRD_CAN
        except ValueError:
            return False

    @property
    def type_label(self) -> str:
        if self.type is None:
            return "?"
        if self.is_can and self.type != "can":
            return f"can ({self.type})"
        return self.type


def _import_piper():
    try:
        from piper_sdk import C_PiperInterface
    except ImportError as exc:
        raise SystemExit(
            "piper_sdk is not installed. Run: uv sync --extra piper"
        ) from exc
    return C_PiperInterface


def degrees_to_sdk(degrees: float) -> int:
    return round(degrees * 1000.0)


def radians_to_sdk(radians: float) -> int:
    return round(radians * _RAD_TO_SDK)


def parse_joint_targets(
    values: list[float],
    *,
    unit: str,
) -> tuple[int, ...]:
    if len(values) != 6:
        raise ValueError(f"expected 6 joint values, got {len(values)}")
    convert = radians_to_sdk if unit == "rad" else degrees_to_sdk
    return tuple(convert(v) for v in values)


def list_can_interfaces() -> list[str]:
    net = Path("/sys/class/net")
    if not net.is_dir():
        return []
    return sorted(p.name for p in net.iterdir() if p.name.startswith("can"))


def _read_sysfs_int(iface: str, leaf: str) -> int | None:
    path = Path("/sys/class/net") / iface / leaf
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_sysfs_text(iface: str, leaf: str) -> str | None:
    path = Path("/sys/class/net") / iface / leaf
    if not path.is_file():
        return None
    try:
        return path.read_text().strip()
    except OSError:
        return None


def can_interface_status(name: str) -> CanInterfaceStatus:
    base = Path("/sys/class/net") / name
    exists = base.is_dir()
    if not exists:
        return CanInterfaceStatus(
            name=name,
            exists=False,
            operstate=None,
            type=None,
            bitrate=None,
            tx_error=None,
            rx_error=None,
            rx_packets=None,
            tx_packets=None,
        )
    return CanInterfaceStatus(
        name=name,
        exists=True,
        operstate=_read_sysfs_text(name, "operstate"),
        type=_read_sysfs_text(name, "type"),
        bitrate=_read_sysfs_int(name, "can_bittiming/bitrate"),
        tx_error=_read_sysfs_int(name, "statistics/tx_errors"),
        rx_error=_read_sysfs_int(name, "statistics/rx_errors"),
        rx_packets=_read_sysfs_int(name, "statistics/rx_packets"),
        tx_packets=_read_sysfs_int(name, "statistics/tx_packets"),
    )


def scan_usb_lsusb() -> list[UsbAdapter]:
    try:
        proc = subprocess.run(
            ["lsusb"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    adapters: list[UsbAdapter] = []
    for line in proc.stdout.splitlines():
        m = re.match(
            r"Bus (\d+) Device (\d+): ID ([0-9a-f]{4}):([0-9a-f]{4}) (.+)",
            line,
            re.IGNORECASE,
        )
        if not m:
            continue
        bus, dev, vid_s, pid_s, desc = m.groups()
        vid, pid = int(vid_s, 16), int(pid_s, 16)
        adapters.append(
            UsbAdapter(
                bus_device=f"{bus}:{dev}",
                id_vendor=vid,
                id_product=pid,
                description=desc.strip(),
            )
        )
    return adapters


def find_known_adapters(adapters: list[UsbAdapter]) -> list[UsbAdapter]:
    return [a for a in adapters if a.known_label is not None]


def _rtnetlink_busy_help(iface: str) -> str:
    return (
        f"\nRTNETLINK 'Device or resource busy' on {iface!r} usually means:\n"
        f"  • {iface} is already UP (your ip link output is fine — do not run --up again)\n"
        "  • A program still has the CAN socket open (prior python demo, candump, piper_sdk)\n"
        "\nFix:\n"
        "  1. Stop other users of CAN:  pkill -f piper_move_joint; pkill candump\n"
        f"  2. If you must reconfigure:  sudo ip link set {iface} down\n"
        f"     then set type/bitrate/up once\n"
        "  3. Re-run the demo after candump shows arm traffic on the bus\n"
    )


def can_bus_has_traffic(can_name: str, *, wait_s: float = 1.5) -> bool:
    """True if candump hears frames (arm powered and on the bus)."""
    try:
        proc = subprocess.run(
            ["timeout", str(wait_s), "candump", can_name],
            capture_output=True,
            text=True,
            timeout=wait_s + 1.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return bool(proc.stdout.strip())


def ensure_can_bus_alive(can_name: str, *, bitrate: int = 1_000_000) -> None:
    """Restart SocketCAN if the arm is not broadcasting (common after idle / USB glitch)."""
    if can_bus_has_traffic(can_name):
        return
    print(f"[can] no traffic on {can_name} — restarting interface", flush=True)
    try:
        bring_up_can(can_name, bitrate=bitrate, force=True)
    except RuntimeError as exc:
        raise SystemExit(
            f"error: {can_name} has no arm traffic and restart failed: {exc}\n"
            "  Check arm power, e-stop, and CAN cable; then:\n"
            f"    sudo ip link set {can_name} down && "
            f"sudo ip link set {can_name} type can bitrate {bitrate} && "
            f"sudo ip link set {can_name} up"
        ) from exc
    time.sleep(0.5)
    if not can_bus_has_traffic(can_name):
        raise SystemExit(
            f"error: {can_name} is UP but still silent — arm not responding on CAN.\n"
            "  Power the arm on, release e-stop, check the CAN cable."
        )
    print(f"[can] {can_name} traffic ok", flush=True)


def bring_up_can(iface: str, *, bitrate: int, force: bool = False) -> None:
    """Configure and bring up a SocketCAN interface (requires CAP_NET_ADMIN / root)."""
    status = can_interface_status(iface)
    if (
        not force
        and status.exists
        and status.is_up
        and status.is_can
        and status.bitrate == bitrate
    ):
        print(f"{iface} already UP @ {bitrate} bps — skipping ip link (use --force-up to reconfigure)")
        return

    steps: tuple[tuple[str, ...], ...] = (
        ("ip", "link", "set", iface, "down"),
        ("ip", "link", "set", iface, "type", "can", "bitrate", str(bitrate)),
        ("ip", "link", "set", iface, "up"),
    )
    for cmd in steps:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            hint = _rtnetlink_busy_help(iface) if "busy" in err.lower() else ""
            raise RuntimeError(f"{' '.join(cmd)} failed: {err or 'unknown error'}{hint}")


def run_linux_check(
    can_name: str,
    *,
    json_out: bool = False,
) -> int:
    adapters = scan_usb_lsusb()
    known = find_known_adapters(adapters)
    can_ifaces = list_can_interfaces()
    status = can_interface_status(can_name)

    payload = {
        "platform": "linux",
        "usb_adapter_plugged": bool(known),
        "known_adapters": [
            asdict(a) | {"known_label": a.known_label, "vid_pid": a.vid_pid}
            for a in known
        ],
        "can_interfaces": can_ifaces,
        "selected": asdict(status),
        "ready": bool(known) and status.exists and status.is_up and status.is_can,
    }

    if json_out:
        print(json.dumps(payload, indent=2))
        return 0 if payload["ready"] else 1

    if known:
        print("USB CAN adapter: detected (lsusb)")
        for a in known:
            print(f"  {a.description!r} — {a.known_label} ({a.vid_pid} @ bus {a.bus_device})")
    else:
        print("USB CAN adapter: not found in lsusb")
        print("Expected GS-USB / candleLight (1d50:606f). Is the dongle plugged in?")

    print()
    if can_ifaces:
        print("SocketCAN interfaces:", ", ".join(can_ifaces))
    else:
        print("SocketCAN interfaces: none (no can* in /sys/class/net)")

    print(f"\nSelected interface {can_name!r}:")
    if not status.exists:
        print("  missing — create/bind interface or pass --can NAME")
        print("  Try: sudo modprobe gs_usb  (then replug adapter)")
        return 1

    print(f"  type: {status.type_label}  operstate: {status.operstate or '?'}")
    if status.bitrate is not None:
        print(f"  bitrate: {status.bitrate} bps")
    if status.is_up:
        print("  (UP + link/can is OK — do not run --up again; use --no-can-auto-init with the demo)")
    if status.rx_packets is not None or status.tx_packets is not None:
        print(f"  frames: rx={status.rx_packets} tx={status.tx_packets}")
    if status.tx_error is not None or status.rx_error is not None:
        print(f"  errors: tx={status.tx_error} rx={status.rx_error}")

    if not status.is_can:
        print("  error: interface is not type 'can'")
        return 1
    if not status.is_up:
        print("  not UP — bring it up, e.g.:")
        print(f"    sudo ip link set {can_name} down")
        print(f"    sudo ip link set {can_name} type can bitrate 1000000")
        print(f"    sudo ip link set {can_name} up")
        print("  Or re-run with: --up  (same commands via ip)")
        return 1

    print("  ready for piper_sdk")
    return 0


def ensure_can_ready(
    can_name: str,
    *,
    bitrate: int,
    bring_up: bool,
    force: bool = False,
) -> None:
    if bring_up:
        print(f"Bringing up {can_name} @ {bitrate} bps…")
        try:
            bring_up_can(can_name, bitrate=bitrate, force=force)
        except RuntimeError as exc:
            raise SystemExit(f"error: {exc}\n(hint: run with sudo, or omit --up)") from exc

    status = can_interface_status(can_name)
    if not status.exists:
        raise SystemExit(
            f"error: CAN interface {can_name!r} not found.\n"
            f"  Available: {', '.join(list_can_interfaces()) or '(none)'}\n"
            "  Plug in the adapter, load gs_usb if needed, or pass --can."
        )
    if not status.is_can:
        raise SystemExit(
            f"error: {can_name!r} is type {status.type!r}, expected CAN (ARPHRD_CAN={_ARPHRD_CAN})"
        )
    if not status.is_up:
        raise SystemExit(
            f"error: {can_name!r} is not UP (operstate={status.operstate!r}).\n"
            f"  Run: sudo ip link set {can_name} type can bitrate {bitrate} && "
            f"sudo ip link set {can_name} up\n"
            "  Or pass --up (requires root/CAP_NET_ADMIN)."
        )


def _motor_low_speed_rows(piper) -> list[tuple[int, int, bool, bool, bool]]:
    """Per-motor: can_id, voltage_0.1V, enabled, error, voltage_low."""
    low = piper.GetArmLowSpdInfoMsgs()
    rows: list[tuple[int, int, bool, bool, bool]] = []
    for i in range(1, 7):
        m = getattr(low, f"motor_{i}")
        foc = m.foc_status
        rows.append(
            (
                int(getattr(m, "can_id", 0) or 0),
                int(getattr(m, "vol", 0) or 0),
                bool(foc.driver_enable_status),
                bool(foc.driver_error_status),
                bool(foc.voltage_too_low),
            )
        )
    return rows


def arm_feedback_seen(piper) -> bool:
    """True if any motor reports a non-zero CAN id (SDK received arm traffic)."""
    return any(can_id != 0 for can_id, *_ in _motor_low_speed_rows(piper))


def print_enable_diagnostics(
    piper,
    *,
    can_name: str,
    rx_before: int | None,
    rx_after: int | None,
) -> None:
    rows = _motor_low_speed_rows(piper)
    print("\nMotor feedback (from GetArmLowSpdInfoMsgs):", file=sys.stderr)
    print(
        "  motor  can_id  enable  err  v_low  V(0.1V)",
        file=sys.stderr,
    )
    for i, (can_id, vol, enabled, err, v_low) in enumerate(rows, start=1):
        print(
            f"  J{i:<4}  {can_id:6}  {str(enabled):5}  {str(err):4}  {str(v_low):5}  {vol}",
            file=sys.stderr,
        )

    if rx_before is not None and rx_after is not None:
        delta = rx_after - rx_before
        print(
            f"\nCAN {can_name} rx_packets: {rx_before} -> {rx_after} (delta {delta})",
            file=sys.stderr,
        )
        if delta <= 0:
            print(
                "  No RX frames while enabling — host is not hearing the arm on the bus.",
                file=sys.stderr,
            )
    elif not arm_feedback_seen(piper):
        print(
            "\nAll motor can_id are 0 — piper_sdk is not receiving arm feedback yet.",
            file=sys.stderr,
        )

    print(
        "\nIf can0 is UP (ip link shows link/can) but enable fails, check hardware:",
        file=sys.stderr,
    )
    print(
        "  • Arm main power ON, e-stop released (not latched red)",
        file=sys.stderr,
    )
    print(
        "  • CAN H/L/GND between adapter and arm base (not only USB)",
        file=sys.stderr,
    )
    print(
        "  • 120Ω termination at one end of the bus (often on the arm)",
        file=sys.stderr,
    )
    print(
        "  • Bitrate 1000000: sudo ip -details link show can0",
        file=sys.stderr,
    )
    print(
        "  • Live traffic: candump can0   (should see frames when arm is powered)",
        file=sys.stderr,
    )
    print(
        "  • Power-cycle the arm after wiring changes (AgileX README)",
        file=sys.stderr,
    )
    print(
        "  • Try: --no-can-auto-init if you already brought can0 up with ip link",
        file=sys.stderr,
    )


def wait_until_enabled(
    piper,
    *,
    timeout_s: float = 15.0,
    poll_s: float = 1.0,
    verbose: bool = False,
) -> bool:
    """Poll driver enable status (official demo uses ~1s between EnableArm retries)."""
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        rows = _motor_low_speed_rows(piper)
        enabled = all(r[2] for r in rows)
        if verbose:
            flags = "".join("1" if r[2] else "0" for r in rows)
            print(f"  enable attempt {attempt}: J1–J6 enabled={flags}", flush=True)
        if enabled:
            return True
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(poll_s)
    return False


def print_joint_feedback(piper, *, label: str) -> None:
    joints = piper.GetArmJointMsgs().joint_state
    deg = [getattr(joints, f"joint_{i}") / 1000.0 for i in range(1, 7)]
    print(f"{label}: " + ", ".join(f"J{i}={deg[i - 1]:.2f}°" for i in range(1, 7)))


def connect_piper(
    piper,
    *,
    can_name: str,
    bitrate: int,
    can_auto_init: bool,
) -> None:
    """Open CAN per piper_sdk: CreateCanBus is required when can_auto_init is False."""
    if can_auto_init:
        piper.ConnectPort()
        return

    create = getattr(piper, "CreateCanBus", None)
    if create is None:
        raise SystemExit(
            "error: this piper_sdk build has no CreateCanBus(); upgrade piper-sdk or drop --no-can-auto-init"
        )

    create(
        can_name,
        bustype="socketcan",
        expected_bitrate=bitrate,
        judge_flag=False,
    )
    piper.ConnectPort(can_init=False)


def move_joints(
    piper,
    targets_sdk: tuple[int, ...],
    *,
    speed_percent: int,
    hold_s: float,
    stream_hz: float,
) -> None:
    """Send MOVE J commands until hold_s elapses (SDK expects a steady command stream)."""
    interval = 1.0 / stream_hz
    deadline = time.monotonic() + hold_s
    while time.monotonic() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
        piper.JointCtrl(*targets_sdk)
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enable AgileX Piper and move joints via piper_sdk (Linux SocketCAN)."
    )
    parser.add_argument(
        "--can",
        default=os.environ.get("PIPER_CAN", "can0"),
        help="SocketCAN interface name (default: can0 or PIPER_CAN)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1_000_000,
        help="CAN bitrate when using --up (default: 1000000)",
    )
    parser.add_argument(
        "--up",
        action="store_true",
        help="Run ip link to configure bitrate and bring interface up (needs root; skip if already UP)",
    )
    parser.add_argument(
        "--force-up",
        action="store_true",
        help="With --up, re-run ip link even when interface is already UP at the target bitrate",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check lsusb adapter and SocketCAN state; then exit",
    )
    parser.add_argument(
        "--check-json",
        action="store_true",
        help="Like --check but emit JSON",
    )
    parser.add_argument(
        "--joints-deg",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Target joint angles in degrees (SDK units: 0.001°)",
    )
    parser.add_argument(
        "--joints-rad",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Target joint angles in radians",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=30,
        metavar="PCT",
        help="Move speed 0–100%% (default: 30)",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=3.0,
        help="Seconds to stream joint commands (default: 3)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Command stream rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--enable-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for motor enable (default: 15)",
    )
    parser.add_argument(
        "--no-can-auto-init",
        action="store_true",
        help="Skip SDK CAN activation (recommended when can0 is already UP via ip link)",
    )
    parser.add_argument(
        "--can-auto-init",
        action="store_true",
        help="Let piper_sdk run CAN setup (do not combine with manual ip link / --up)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Connect and print CAN/motor diagnostics only; do not move",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-attempt enable status",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print targets only; do not open CAN or move the arm",
    )
    args = parser.parse_args(argv)

    if sys.platform != "linux" and not args.dry_run and not args.check and not args.check_json:
        print(
            "warning: this demo targets Linux SocketCAN; CAN control may not work on "
            f"{sys.platform}",
            file=sys.stderr,
        )

    if args.check or args.check_json:
        return run_linux_check(args.can, json_out=args.check_json)

    if args.joints_deg and args.joints_rad:
        print("error: use only one of --joints-deg or --joints-rad", file=sys.stderr)
        return 2

    if args.joints_deg:
        unit = "deg"
        raw = list(args.joints_deg)
    elif args.joints_rad:
        unit = "rad"
        raw = list(args.joints_rad)
    else:
        # Small offset from home (official piper_joint_ctrl.py pose B)
        unit = "rad"
        raw = [0.2, 0.2, -0.2, 0.3, -0.2, 0.5]

    targets_sdk = parse_joint_targets(raw, unit=unit)
    if not 0 <= args.speed <= 100:
        print("error: --speed must be between 0 and 100", file=sys.stderr)
        return 2

    print(f"CAN: {args.can}")
    print(f"Target ({unit}): {raw}")
    print(f"SDK units (0.001°): {targets_sdk}")
    print(f"Speed: {args.speed}%  hold: {args.hold}s  rate: {args.rate} Hz")

    if args.dry_run:
        return 0

    if args.probe:
        args.verbose = True

    pre = can_interface_status(args.can)
    rx_before = pre.rx_packets

    if args.can_auto_init and args.no_can_auto_init:
        print("error: use only one of --can-auto-init or --no-can-auto-init", file=sys.stderr)
        return 2

    # Avoid double ip link: SDK init + manual --up both trigger RTNETLINK busy.
    can_auto_init = args.can_auto_init
    if not args.can_auto_init and not args.no_can_auto_init:
        can_auto_init = not (pre.is_up and pre.is_can)
        if pre.is_up and pre.is_can:
            print(
                f"{args.can} already UP — using --no-can-auto-init "
                "(pass --can-auto-init to let piper_sdk configure CAN)"
            )
    elif args.no_can_auto_init:
        can_auto_init = False

    ensure_can_ready(
        args.can,
        bitrate=args.bitrate,
        bring_up=args.up,
        force=args.force_up,
    )

    C_PiperInterface = _import_piper()
    piper = C_PiperInterface(
        can_name=args.can,
        can_auto_init=can_auto_init,
    )

    print("Connecting CAN…")
    if can_auto_init:
        print("  (piper_sdk can_auto_init — SDK owns CAN setup)")
    else:
        print("  (CreateCanBus + ConnectPort; can0 must already be UP via ip link)")
    connect_piper(
        piper,
        can_name=args.can,
        bitrate=args.bitrate,
        can_auto_init=can_auto_init,
    )
    time.sleep(1.0)  # reader thread + initial arm feedback (matches piper_sdk demos)
    piper.MotionCtrl_1(0x01, 0x00, 0x00)  # estop
    time.sleep(0.3)
    piper.MotionCtrl_1(0x02, 0x00, 0x00)  # recover
    time.sleep(0.3)
    piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    deadline = time.monotonic() + args.enable_timeout
    while time.monotonic() < deadline:
        if piper.EnablePiper():
            break
        time.sleep(0.01)
    else:
        post = can_interface_status(args.can)
        print("error: EnablePiper timed out", file=sys.stderr)
        print_enable_diagnostics(
            piper,
            can_name=args.can,
            rx_before=rx_before,
            rx_after=post.rx_packets,
        )
        return 1

    piper.GripperCtrl(0, 1000, 0x01, 0)

    print("Motors enabled.")
    print_joint_feedback(piper, label="Current")

    if args.probe:
        post = can_interface_status(args.can)
        print_enable_diagnostics(
            piper,
            can_name=args.can,
            rx_before=rx_before,
            rx_after=post.rx_packets,
        )
        return 0

    print("Streaming MOVE J commands…")
    move_joints(
        piper,
        targets_sdk,
        speed_percent=args.speed,
        hold_s=args.hold,
        stream_hz=args.rate,
    )

    print_joint_feedback(piper, label="Final")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
