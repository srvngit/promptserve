#!/usr/bin/env python3
"""
Detect AgileX Piper USB-to-CAN adapters on macOS via IORegistry (ioreg).

The stock Piper bundle uses a GS-USB / candleLight adapter (VID 0x1D50, PID 0x606F).
It appears in IORegistry but not as /dev/cu.* — piper_sdk on macOS uses its serial
CAN backend instead of Linux SocketCAN.

  uv run python demos/piper_check_usb.py
  uv run python demos/piper_check_usb.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

# AgileX / candleLight / GS-USB adapters commonly bundled with Piper
_KNOWN_ADAPTERS: tuple[tuple[int, int, str], ...] = (
    (0x1D50, 0x606F, "GS-USB / candleLight (AgileX Piper kit)"),
    (0x1209, 0x2323, "candleLight (alternate PID)"),
    (0x1CD2, 0x606F, "CES CANext FD"),
    (0x16D0, 0x10B8, "ABE CANdebugger FD"),
)

_NAME_HINTS = (
    "candlelight",
    "can adapter",
    "gs_usb",
    "gs-usb",
    "canable",
    "piper",
)


@dataclass(frozen=True)
class UsbDevice:
    name: str
    vendor: str
    id_vendor: int | None
    id_product: int | None
    location_id: str | None
    serial: str | None

    @property
    def vid_pid(self) -> str:
        if self.id_vendor is None or self.id_product is None:
            return "unknown"
        return f"{self.id_vendor:04x}:{self.id_product:04x}"

    @property
    def known_label(self) -> str | None:
        if self.id_vendor is None or self.id_product is None:
            return None
        for vid, pid, label in _KNOWN_ADAPTERS:
            if self.id_vendor == vid and self.id_product == pid:
                return label
        return None


def _parse_ioreg_devices(text: str) -> list[UsbDevice]:
    """Parse flat ioreg -p IOUSB -l output into USB device records."""
    blocks = re.split(r"\n(?=\s*\+-o )", text)
    devices: list[UsbDevice] = []

    for block in blocks:
        name_match = re.search(r'\+-o ([^@]+)@', block)
        if not name_match:
            continue
        name = name_match.group(1).strip()

        def _int_field(key: str) -> int | None:
            m = re.search(rf'"{key}" = (\d+)', block)
            return int(m.group(1)) if m else None

        def _str_field(key: str) -> str | None:
            m = re.search(rf'"{key}" = "([^"]*)"', block)
            return m.group(1) if m else None

        id_vendor = _int_field("idVendor")
        id_product = _int_field("idProduct")
        vendor = _str_field("USB Vendor Name") or _str_field("kUSBVendorString") or ""
        product = _str_field("USB Product Name") or _str_field("kUSBProductString") or name
        location = _str_field("locationID")
        serial = _str_field("USB Serial Number") or _str_field("kUSBSerialNumberString")

        # Skip anonymous hubs/controllers without product identity
        if id_vendor is None and not vendor and "controller" in name.lower():
            continue

        devices.append(
            UsbDevice(
                name=product,
                vendor=vendor,
                id_vendor=id_vendor,
                id_product=id_product,
                location_id=location,
                serial=serial,
            )
        )

    return devices


def _is_piper_candidate(dev: UsbDevice) -> bool:
    if dev.known_label:
        return True
    blob = f"{dev.name} {dev.vendor}".lower()
    return any(h in blob for h in _NAME_HINTS)


def scan_usb_ioreg() -> list[UsbDevice]:
    proc = subprocess.run(
        ["ioreg", "-p", "IOUSB", "-l", "-w", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_ioreg_devices(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect Piper CAN USB adapter via IORegistry")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--all-usb", action="store_true", help="List every USB device node from ioreg")
    args = parser.parse_args(argv)

    try:
        devices = scan_usb_ioreg()
    except FileNotFoundError:
        print("error: ioreg not found (macOS only)", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"error: ioreg failed: {exc}", file=sys.stderr)
        return 2

    matches = [d for d in devices if _is_piper_candidate(d)]

    if args.json:
        payload = {
            "plugged_in": bool(matches),
            "adapters": [asdict(d) | {"known_label": d.known_label, "vid_pid": d.vid_pid} for d in matches],
        }
        if args.all_usb:
            payload["all_usb"] = [asdict(d) | {"vid_pid": d.vid_pid} for d in devices]
        print(json.dumps(payload, indent=2))
        return 0 if matches else 1

    if args.all_usb:
        print("All IOUSB devices:")
        for d in devices:
            print(f"  - {d.name!r} ({d.vendor}) {d.vid_pid} @ {d.location_id or '?'}")
        print()

    if not matches:
        print("Piper CAN adapter: not detected in IORegistry")
        print("Expected a candleLight / GS-USB dongle (VID 0x1D50, PID 0x606F).")
        return 1

    print("Piper CAN adapter: detected")
    for d in matches:
        label = d.known_label or "possible CAN adapter (name match)"
        print(f"  {d.name!r} — {label}")
        print(f"    vendor: {d.vendor or '(unknown)'}  vid:pid: {d.vid_pid}  location: {d.location_id or '?'}")
        if d.serial:
            print(f"    serial: {d.serial}")
    print()
    print("Note: On macOS this USB device does not create /dev/cu.*; use piper_sdk serial CAN mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
