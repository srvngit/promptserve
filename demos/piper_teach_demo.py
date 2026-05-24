#!/usr/bin/env python3
"""
Record and replay Piper drag-teach demos — with optional X/Z vision adaptation.

Record (motors OFF, drag by hand):
  ./scripts/piper_record_demo.sh ketchup

Replay recorded joints:
  ./scripts/piper_play_demo.sh ketchup

Adaptive replay (vision shifts trajectory in X+Z):
  ./scripts/piper_play_adapt.sh ketchup

Prerecorded bits (recommended — record short moves, run adapted sequence):
  ./scripts/piper_init_sequence.sh ketchup
  ./scripts/piper_record_bit.sh 01_home_to_pick --role transit
  ./scripts/piper_record_bit.sh 02_grasp --role pick
  ./scripts/piper_play_bits_adapt.sh ketchup
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

from demos.piper_move_joint import (
    can_interface_status,
    connect_piper,
    ensure_can_ready,
)
from hackstorm.arm.teach_bits import (
    BitRole,
    adapt_bit_for_role,
    adapt_bit_for_role_xy,
    bit_path,
    compute_sequence_offsets,
    compute_sequence_offsets_xy,
    enrich_bit_payload,
    load_bit_frames,
    load_sequence,
    sequence_path,
    write_sequence_template,
)
from hackstorm.arm.teach_trajectory import (
    adapt_frames_xy,
    adapt_frames_xz,
    clamp_gripper_um,
    compress_frames,
    enrich_recording_metadata,
    find_markers,
    joints_deg_to_rad,
    joints_deg_to_sdk,
    vision_adapt_targets,
    vision_adapt_targets_xy,
)

_RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
_STREAM_HZ = 200.0
_GRIPPER_OPEN_UM = 0


def _import_piper():
    from piper_sdk import C_PiperInterface

    return C_PiperInterface


def _default_recording(name: str) -> Path:
    slug = name.replace(" ", "_").lower()
    if not slug.endswith("_demo"):
        slug = f"{slug}_demo"
    return _RECORDINGS_DIR / f"{slug}.yaml"


def _read_joints_deg(piper) -> list[float]:
    joints = piper.GetArmJointMsgs().joint_state
    return [getattr(joints, f"joint_{i}") / 1000.0 for i in range(1, 7)]


def _read_gripper_um(piper) -> int:
    g = piper.GetArmGripperMsgs().gripper_state
    return clamp_gripper_um(int(getattr(g, "grippers_angle", 0) or 0))


def _open_piper(can: str, *, no_can_auto_init: bool, can_auto_init: bool):
    pre = can_interface_status(can)
    can_auto_init_val = _resolve_can_auto_init_from_flags(
        no_can_auto_init=no_can_auto_init,
        can_auto_init=can_auto_init,
        pre=pre,
    )
    ensure_can_ready(can, bitrate=1_000_000, bring_up=False)
    C_PiperInterface = _import_piper()
    piper = C_PiperInterface(can_name=can, can_auto_init=can_auto_init_val)
    connect_piper(
        piper,
        can_name=can,
        bitrate=1_000_000,
        can_auto_init=can_auto_init_val,
    )
    time.sleep(1.0)
    return piper


def _arm_status(piper) -> str:
    try:
        st = piper.GetArmStatus().arm_status
        return (
            f"ctrl=0x{st.ctrl_mode:02x} mode=0x{st.mode_feed:02x} "
            f"status=0x{st.arm_status:02x}"
        )
    except Exception:
        return "status unavailable"


def _effective_speed(speed_percent: int) -> int:
    return max(int(speed_percent), 50)


def _enable_for_motion(piper, *, speed_percent: int) -> None:
    spd = _effective_speed(speed_percent)
    piper.MotionCtrl_1(0x01, 0x00, 0x00)  # estop — clears any teach/drag state
    time.sleep(0.3)
    piper.MotionCtrl_1(0x02, 0x00, 0x00)  # recover
    time.sleep(0.3)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, spd, 0x00)
        piper.EnablePiper()
        st = piper.GetArmStatus().arm_status
        if str(st.ctrl_mode) == "CAN_CTRL(0x1)":
            break
        time.sleep(0.02)
    else:
        raise RuntimeError(f"arm never reached CAN_CTRL — {_arm_status(piper)}")
    print(f"[teach] motors enabled @ {spd}% — {_arm_status(piper)}", flush=True)


def _disable_for_drag(piper) -> None:
    piper.MotionCtrl_1(0x02, 0x00, 0x00)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not piper.DisablePiper():
            break
        time.sleep(0.02)
    print("Motors OFF — drag the arm by hand (gravity-assist).", flush=True)


def _joints_changed(a: list[float], b: list[float], *, threshold_deg: float) -> bool:
    return any(abs(x - y) > threshold_deg for x, y in zip(a, b))


def _load_recording(path: Path) -> dict:
    if not path.is_file():
        print(f"Recording not found: {path}", file=sys.stderr)
        print("Record first: ./scripts/piper_record_demo.sh ketchup", file=sys.stderr)
        raise SystemExit(1)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stream_to_frame(
    piper,
    frame: dict,
    *,
    speed_percent: int,
    hold_s: float,
) -> None:
    """Stream joints only — same as smoke test (no GripperCtrl in the loop)."""
    targets = joints_deg_to_sdk(frame["joints_deg"])
    spd = _effective_speed(speed_percent)
    interval = 1.0 / _STREAM_HZ
    deadline = time.monotonic() + max(hold_s, 0.15)
    while time.monotonic() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, spd, 0x00)
        piper.JointCtrl(*targets)
        time.sleep(interval)


def _playback_with_controller(
    frames: list[dict],
    *,
    can: str,
    speed_percent: int,
    time_scale: float,
) -> None:
    """Replay via shared Piper session — re-enables motors before each run."""
    from hackstorm.arm.replay_session import playback_frames

    spd = _effective_speed(speed_percent)
    print(
        f"Playing {len(frames)} waypoints @ {spd}% ({frames[-1]['t']:.1f}s)",
        flush=True,
    )
    playback_frames(
        frames,
        can=can,
        speed_percent=spd,
        time_scale=time_scale,
    )


def _playback_frames(
    piper,
    frames: list[dict],
    *,
    speed_percent: int,
    time_scale: float,
    already_enabled: bool = False,
    label: str = "",
) -> None:
    if not already_enabled:
        _enable_for_motion(piper, speed_percent=speed_percent)
    piper.GripperCtrl(_GRIPPER_OPEN_UM, 1000, 0x01, 0)
    if not frames:
        return
    live = _read_joints_deg(piper)
    tag = f"{label} " if label else ""
    print(
        f"{tag}streaming {len(frames)} waypoints live J2={live[1]:+.1f}° "
        f"→ {frames[0]['joints_deg'][1]:+.1f}°",
        flush=True,
    )
    for i, frame in enumerate(frames):
        if i + 1 < len(frames):
            hold = (frames[i + 1]["t"] - frame["t"]) * time_scale
        else:
            hold = 0.5 * time_scale
        _stream_to_frame(piper, frame, speed_percent=speed_percent, hold_s=hold)


def _depth_valid_for_adapt() -> bool:
    """Orbbec USB2 / color-only mode has no real depth — adapt X only."""
    return os.environ.get("HACKSTORM_ORBBEC_COLOR_ONLY", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _vision_scan(
    *,
    pick: str,
    place: str,
    camera: str,
    backend: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from hackstorm.perception.locate import locate_in_frame
    from hackstorm.perception.open_camera import open_camera
    from hackstorm.vision.detector import resolve_detection_backend

    table_z = float(os.environ.get("HACKSTORM_TABLE_Z_M", "0.10"))
    kwargs: dict = {
        "backend": resolve_detection_backend(backend),
        "default_z_m": table_z,
    }
    cam = open_camera(camera)
    cam.open()
    try:
        print("[adapt] settling camera...", flush=True)
        for _ in range(12):
            cam.capture()
            time.sleep(0.08)
        frame = cam.capture()
        try:
            from PIL import Image

            out = Path("captures/adapt_scan.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.color_rgb).save(out)
            print(f"[adapt] saved {out}", flush=True)
        except Exception:
            pass
        pick_pt = locate_in_frame(pick, frame, **kwargs)
        place_pt = locate_in_frame(place, frame, **kwargs)
    finally:
        cam.close()

    if pick_pt is None or place_pt is None:
        missing = [n for n, p in (("pick", pick_pt), ("place", place_pt)) if p is None]
        print(f"Vision failed — could not find: {', '.join(missing)}", file=sys.stderr)
        print("See captures/debug/ — or pass --pick-x/y --place-x/y", file=sys.stderr)
        raise SystemExit(1)
    return pick_pt, place_pt


def _vision_scan_xy(
    *,
    pick: str,
    place: str,
    camera: str,
    backend: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """DINO bbox → arm X/Y only (homography if calibrated, else table-depth fallback)."""
    from hackstorm.perception.locate import (
        has_tabletop_calibration,
        load_calibration,
        locate_in_frame,
    )
    from hackstorm.perception.open_camera import open_camera
    from hackstorm.vision.detector import detect_with_raw, pick_best_in_frame, resolve_detection_backend

    table_z = float(os.environ.get("HACKSTORM_TABLE_Z_M", "0.10"))
    det_backend = resolve_detection_backend(backend)
    kwargs: dict = {"backend": det_backend, "default_z_m": table_z}
    homography = load_calibration() if has_tabletop_calibration() else None

    cam = open_camera(camera)
    cam.open()
    try:
        print("[adapt-xy] settling camera...", flush=True)
        for _ in range(12):
            cam.capture()
            time.sleep(0.08)
        frame = cam.capture()
        try:
            from PIL import Image

            out = Path("captures/adapt_xy_scan.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.color_rgb).save(out)
            print(f"[adapt-xy] saved {out}", flush=True)
        except Exception:
            pass

        h, w = frame.color_rgb.shape[:2]

        def _xy_for(label: str) -> tuple[float, float, float] | None:
            if homography is not None:
                boxes, _raw = detect_with_raw(frame.color_rgb, label, backend=det_backend)
                target = pick_best_in_frame(boxes, label, width=w, height=h)
                if target is None:
                    return None
                ax, ay = homography.pixel_to_arm_xy(target.center_xy[0], target.center_xy[1])
                print(f"[adapt-xy] {label!r} homography → ({ax:.3f}, {ay:.3f}) m", flush=True)
                return (ax, ay, table_z)
            return locate_in_frame(label, frame, **kwargs)

        pick_pt = _xy_for(pick)
        place_pt = _xy_for(place)
    finally:
        cam.close()

    if pick_pt is None or place_pt is None:
        missing = [n for n, p in (("pick", pick_pt), ("place", place_pt)) if p is None]
        print(f"Vision failed — could not find: {', '.join(missing)}", file=sys.stderr)
        print("See captures/debug/ — or pass --pick-x/y --place-x/y", file=sys.stderr)
        raise SystemExit(1)
    return pick_pt, place_pt


def _record_drag_loop(
    piper,
    *,
    rate: float,
    threshold_deg: float,
) -> list[dict]:
    frames: list[dict] = []
    t0 = time.monotonic()
    last_joints: list[float] | None = None
    last_gripper: int | None = None
    min_interval = 1.0 / max(rate, 1.0)
    try:
        while True:
            joints = _read_joints_deg(piper)
            gripper = _read_gripper_um(piper)
            now = time.monotonic()
            changed = last_joints is None or _joints_changed(
                last_joints, joints, threshold_deg=threshold_deg
            )
            gripper_changed = last_gripper is None or gripper != last_gripper
            due = not frames or (now - t0 - frames[-1]["t"]) >= min_interval
            if changed or gripper_changed or due:
                frames.append(
                    {
                        "t": round(now - t0, 3),
                        "joints_deg": [round(j, 3) for j in joints],
                        "gripper_um": gripper,
                    }
                )
                last_joints = joints
                last_gripper = gripper
                if len(frames) == 1 or len(frames) % 10 == 0:
                    print(
                        f"  frame {len(frames):4d}  t={frames[-1]['t']:6.2f}s  "
                        f"J2={joints[1]:+.1f}°  gripper={gripper}",
                        flush=True,
                    )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped recording.", flush=True)
    return frames


def cmd_record(args: argparse.Namespace) -> int:
    can = args.can or os.environ.get("PIPER_CAN", "can0")
    out = Path(args.output) if args.output else _default_recording(args.name)

    print(f"CAN: {can}")
    print(f"Output: {out}")
    piper = _open_piper(
        can,
        no_can_auto_init=args.no_can_auto_init,
        can_auto_init=args.can_auto_init,
    )
    _enable_for_motion(piper, speed_percent=30)
    _disable_for_drag(piper)

    print(
        "\n=== RECORDING ===\n"
        "Drag: home → ketchup (close gripper) → pink tray (open) → home\n"
        "Press Ctrl+C when done.\n",
        flush=True,
    )

    frames = _record_drag_loop(
        piper, rate=args.rate, threshold_deg=args.threshold_deg
    )

    if len(frames) < 2:
        print(
            f"Only {len(frames)} frame(s) captured — drag the arm longer, then Ctrl+C.",
            file=sys.stderr,
        )
        return 1

    meta = enrich_recording_metadata(frames)
    payload = {
        "name": args.name,
        "can": can,
        "speed_percent": args.speed,
        "frames": frames,
        **meta,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    m = meta["markers"]
    print(f"\nok: saved {len(frames)} frames ({frames[-1]['t']:.1f}s) → {out}")
    print(
        f"  pick ref  idx={m['pick_idx']}  xyz={m['pick_xyz_m']}\n"
        f"  place ref idx={m['place_idx']} xyz={m['place_xyz_m']}"
    )
    print(f"Adapt: ./scripts/piper_play_adapt.sh {args.name}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    path = Path(args.file) if args.file else _default_recording(args.name)
    raw = _load_recording(path)
    frames = compress_frames(raw["frames"])
    speed = int(args.speed or raw.get("speed_percent", 30))
    can = args.can or raw.get("can") or os.environ.get("PIPER_CAN", "can0")
    time_scale = max(args.time_scale, 0.1)

    print(
        f"Playing {path.name}: {len(frames)} waypoints "
        f"(from {len(raw['frames'])} recorded), {frames[-1]['t']:.1f}s @ {speed}%",
        flush=True,
    )
    _playback_with_controller(frames, can=can, speed_percent=speed, time_scale=time_scale)
    print("ok: playback complete")
    return 0


def cmd_play_adapt(args: argparse.Namespace) -> int:
    path = Path(args.file) if args.file else _default_recording(args.name)
    raw = _load_recording(path)
    frames = compress_frames(raw["frames"])
    speed = int(args.speed or raw.get("speed_percent", 30))
    can = args.can or raw.get("can") or os.environ.get("PIPER_CAN", "can0")
    time_scale = max(args.time_scale, 0.1)
    markers = find_markers(frames)
    depth_valid = _depth_valid_for_adapt()

    print(
        f"[adapt] markers pick@{markers.pick_idx} place@{markers.place_idx} "
        f"ref pick_xyz={[round(v,3) for v in markers.pick_xyz_m]}",
        flush=True,
    )

    if args.pick_x is not None and args.place_x is not None:
        pick_pt = (args.pick_x, args.pick_y or markers.pick_xyz_m[1], args.pick_z or markers.pick_xyz_m[2])
        place_pt = (
            args.place_x,
            args.place_y or markers.place_xyz_m[1],
            args.place_z or markers.place_xyz_m[2],
        )
        print(f"[adapt] manual pick {pick_pt}  place {place_pt}", flush=True)
    else:
        pick_pt, place_pt = _vision_scan(
            pick=args.pick,
            place=args.place,
            camera=args.camera,
            backend=args.backend,
        )
        print(f"[adapt] vision pick {pick_pt}  place {place_pt}", flush=True)

    pick_pt, place_pt = vision_adapt_targets(
        pick_pt, place_pt, markers, depth_valid=depth_valid
    )
    if not depth_valid:
        print("[adapt] color-only — shifting X from vision, keeping taught Y/Z", flush=True)

    adapted, pick_off, place_off = adapt_frames_xz(
        frames, markers,
        pick_target_xyz_m=pick_pt,
        place_target_xyz_m=place_pt,
    )
    print(
        f"[adapt] X/Z shift at pick:  dx={pick_off.dx_m:+.3f} m  dz={pick_off.dz_m:+.3f} m\n"
        f"[adapt] X/Z shift at place: dx={place_off.dx_m:+.3f} m  dz={place_off.dz_m:+.3f} m",
        flush=True,
    )

    # Open arm and replay AFTER vision (same path as smoke test).
    _playback_with_controller(adapted, can=can, speed_percent=speed, time_scale=time_scale)
    print("ok: adaptive playback complete")
    return 0


def cmd_play_adapt_xy(args: argparse.Namespace) -> int:
    """Replay taught path with DINO X/Y shift only — Z stays from recording."""
    path = Path(args.file) if args.file else _default_recording(args.name)
    raw = _load_recording(path)
    frames = compress_frames(raw["frames"])
    speed = int(args.speed or raw.get("speed_percent", 30))
    can = args.can or raw.get("can") or os.environ.get("PIPER_CAN", "can0")
    time_scale = max(args.time_scale, 0.1)
    markers = find_markers(frames)

    print(
        f"[adapt-xy] markers pick@{markers.pick_idx} place@{markers.place_idx} "
        f"ref pick_xyz={[round(v,3) for v in markers.pick_xyz_m]} "
        f"place_xyz={[round(v,3) for v in markers.place_xyz_m]}",
        flush=True,
    )

    if args.pick_x is not None and args.place_x is not None:
        pick_pt = (args.pick_x, args.pick_y or markers.pick_xyz_m[1], markers.pick_xyz_m[2])
        place_pt = (
            args.place_x,
            args.place_y or markers.place_xyz_m[1],
            markers.place_xyz_m[2],
        )
        print(f"[adapt-xy] manual pick xy=({pick_pt[0]:.3f},{pick_pt[1]:.3f})  "
              f"place xy=({place_pt[0]:.3f},{place_pt[1]:.3f})", flush=True)
    else:
        pick_pt, place_pt = _vision_scan_xy(
            pick=args.pick,
            place=args.place,
            camera=args.camera,
            backend=args.backend,
        )
        print(f"[adapt-xy] vision pick {pick_pt}  place {place_pt}", flush=True)

    pick_pt, place_pt = vision_adapt_targets_xy(pick_pt, place_pt, markers)
    print(
        "[adapt-xy] locked Z from teach — using vision X/Y only:\n"
        f"  pick  → ({pick_pt[0]:.3f}, {pick_pt[1]:.3f}, {pick_pt[2]:.3f}) m\n"
        f"  place → ({place_pt[0]:.3f}, {place_pt[1]:.3f}, {place_pt[2]:.3f}) m",
        flush=True,
    )

    adapted, pick_off, place_off = adapt_frames_xy(
        frames,
        markers,
        pick_target_xyz_m=pick_pt,
        place_target_xyz_m=place_pt,
    )
    print(
        f"[adapt-xy] shift at pick:  dx={pick_off.dx_m:+.3f} dy={pick_off.dy_m:+.3f} m\n"
        f"[adapt-xy] shift at place: dx={place_off.dx_m:+.3f} dy={place_off.dy_m:+.3f} m",
        flush=True,
    )

    _playback_with_controller(adapted, can=can, speed_percent=speed, time_scale=time_scale)
    print("ok: adapt-xy playback complete")
    return 0


def cmd_init_sequence(args: argparse.Namespace) -> int:
    out = write_sequence_template(args.name)
    print(f"ok: sequence template → {out}")
    print("Record each bit (motors OFF, drag, Ctrl+C):")
    print("  ./scripts/piper_record_bit.sh 01_home_to_pick --role transit")
    print("  ./scripts/piper_record_bit.sh 02_grasp --role pick")
    print("  ./scripts/piper_record_bit.sh 03_to_tray --role transit")
    print("  ./scripts/piper_record_bit.sh 04_release --role place")
    print("  ./scripts/piper_record_bit.sh 05_to_home --role none")
    print("Then set reference pick/place xyz in sequence.yaml from bit end_xyz_m")
    print(f"Run: ./scripts/piper_play_bits_adapt.sh {args.name}")
    return 0


def cmd_record_bit(args: argparse.Namespace) -> int:
    role: BitRole = args.role  # type: ignore[assignment]
    can = args.can or os.environ.get("PIPER_CAN", "can0")
    out = Path(args.output) if args.output else bit_path(args.name)
    print(f"BIT RECORD  role={role}  → {out}")
    piper = _open_piper(
        can,
        no_can_auto_init=args.no_can_auto_init,
        can_auto_init=args.can_auto_init,
    )
    _enable_for_motion(piper, speed_percent=30)
    _disable_for_drag(piper)
    print("Drag ONE short motion. Ctrl+C when done.\n", flush=True)
    frames = _record_drag_loop(
        piper, rate=args.rate, threshold_deg=args.threshold_deg
    )
    if len(frames) < 2:
        print(f"Only {len(frames)} frame(s) — record a longer motion.", file=sys.stderr)
        return 1
    payload = enrich_bit_payload(
        args.name,
        frames,
        role=role,
        can=can,
        speed_percent=args.speed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"ok: {len(frames)} frames → {out}")
    print(f"  end_xyz_m={payload['end_xyz_m']}")
    return 0


def _play_sequence(
    seq_path: Path,
    *,
    adapt: bool,
    adapt_xy: bool,
    args: argparse.Namespace,
) -> int:
    seq = load_sequence(seq_path)
    speed = int(args.speed or seq.speed_percent)
    can = args.can or seq.can or os.environ.get("PIPER_CAN", "can0")
    time_scale = max(args.time_scale, 0.1)
    pick_off = place_off = None

    piper = _open_piper(
        can,
        no_can_auto_init=args.no_can_auto_init,
        can_auto_init=args.can_auto_init,
    )
    _enable_for_motion(piper, speed_percent=speed)

    if adapt or adapt_xy:
        if args.pick_x is not None and args.place_x is not None:
            pick_pt = (
                args.pick_x,
                args.pick_y or seq.reference_pick_xyz_m[1],
                seq.reference_pick_xyz_m[2],
            )
            place_pt = (
                args.place_x,
                args.place_y or seq.reference_place_xyz_m[1],
                seq.reference_place_xyz_m[2],
            )
        else:
            scan = _vision_scan_xy if adapt_xy else _vision_scan
            pick_pt, place_pt = scan(
                pick=args.pick or seq.pick_label,
                place=args.place or seq.place_label,
                camera=args.camera,
                backend=args.backend,
            )
        if adapt_xy:
            pick_off, place_off = compute_sequence_offsets_xy(
                seq,
                pick_target_xyz_m=pick_pt,
                place_target_xyz_m=place_pt,
            )
            print(
                f"[bits-xy] pick shift dx={pick_off.dx_m:+.3f} dy={pick_off.dy_m:+.3f}  "
                f"place dx={place_off.dx_m:+.3f} dy={place_off.dy_m:+.3f}  (Z locked)",
                flush=True,
            )
        else:
            depth_valid = _depth_valid_for_adapt()
            pick_off, place_off = compute_sequence_offsets(
                seq,
                pick_target_xyz_m=pick_pt,
                place_target_xyz_m=place_pt,
                depth_valid=depth_valid,
            )
            print(
                f"[bits] pick shift dx={pick_off.dx_m:+.3f} dz={pick_off.dz_m:+.3f}  "
                f"place dx={place_off.dx_m:+.3f} dz={place_off.dz_m:+.3f}",
                flush=True,
            )
            if not depth_valid:
                print("[bits] color-only camera — adapting X only", flush=True)

    for i, spec in enumerate(seq.bits, start=1):
        if not spec.path.is_file():
            print(f"Missing bit: {spec.path}", file=sys.stderr)
            return 1
        frames = load_bit_frames(spec.path)
        label = f"[bit {i}/{len(seq.bits)} {spec.path.stem} role={spec.role}]"
        if adapt_xy and pick_off is not None and place_off is not None:
            frames = adapt_bit_for_role_xy(
                frames, spec.role, pick_off=pick_off, place_off=place_off
            )
        elif adapt and pick_off is not None and place_off is not None:
            frames = adapt_bit_for_role(
                frames, spec.role, pick_off=pick_off, place_off=place_off
            )
        print(label, flush=True)
        _playback_frames(
            piper,
            frames,
            speed_percent=speed,
            time_scale=time_scale,
            already_enabled=True,
            label=label,
        )

    print("ok: sequence complete")
    return 0


def cmd_play_bits(args: argparse.Namespace) -> int:
    path = Path(args.file) if args.file else sequence_path(args.name)
    if not path.is_file():
        print(f"No sequence at {path}", file=sys.stderr)
        print(f"Run: ./scripts/piper_init_sequence.sh {args.name}", file=sys.stderr)
        return 1
    return _play_sequence(path, adapt=False, adapt_xy=False, args=args)


def cmd_play_bits_adapt(args: argparse.Namespace) -> int:
    path = Path(args.file) if args.file else sequence_path(args.name)
    if not path.is_file():
        print(f"No sequence at {path}", file=sys.stderr)
        print(f"Run: ./scripts/piper_init_sequence.sh {args.name}", file=sys.stderr)
        return 1
    return _play_sequence(path, adapt=True, adapt_xy=False, args=args)


def cmd_play_bits_adapt_xy(args: argparse.Namespace) -> int:
    path = Path(args.file) if args.file else sequence_path(args.name)
    if not path.is_file():
        print(f"No sequence at {path}", file=sys.stderr)
        print(f"Run: ./scripts/piper_init_sequence.sh {args.name}", file=sys.stderr)
        return 1
    return _play_sequence(path, adapt=False, adapt_xy=True, args=args)


def cmd_list(_args: argparse.Namespace) -> int:
    if not _RECORDINGS_DIR.is_dir():
        print("(no recordings yet)")
        return 0
    files = sorted(_RECORDINGS_DIR.glob("*.yaml"))
    if not files:
        print("(no recordings yet)")
        return 0
    for p in files:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        n = len(raw.get("frames", []))
        dur = raw["frames"][-1]["t"] if n else 0
        mk = raw.get("markers", {})
        print(f"  {p.stem}: {n} frames, {dur:.1f}s  pick_idx={mk.get('pick_idx', '?')}")
    return 0


def _resolve_can_auto_init_from_flags(
    *,
    no_can_auto_init: bool,
    can_auto_init: bool,
    pre,
) -> bool:
    if no_can_auto_init and can_auto_init:
        raise SystemExit("use only one of --no-can-auto-init / --can-auto-init")
    if no_can_auto_init:
        return False
    if can_auto_init:
        return True
    if pre.is_up and pre.is_can:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record / replay / adapt Piper teach demos")
    parser.add_argument("--can", default=None)
    parser.add_argument("--no-can-auto-init", action="store_true")
    parser.add_argument("--can-auto-init", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Disable motors, drag arm, save joints")
    rec.add_argument("name", nargs="?", default="ketchup_demo")
    rec.add_argument("-o", "--output", default=None)
    rec.add_argument("--rate", type=float, default=10.0)
    rec.add_argument("--threshold-deg", type=float, default=0.5)
    rec.add_argument("--speed", type=int, default=30)

    play = sub.add_parser("play", help="Replay recording (joint-space)")
    play.add_argument("name", nargs="?", default="ketchup_demo")
    play.add_argument("-f", "--file", default=None)
    play.add_argument("--speed", type=int, default=None)
    play.add_argument("--time-scale", type=float, default=1.0)

    adapt = sub.add_parser(
        "play-adapt",
        help="Vision X/Z adapts recorded path, then replays",
    )
    adapt.add_argument("name", nargs="?", default="ketchup_demo")
    adapt.add_argument("-f", "--file", default=None)
    adapt.add_argument("--pick", default="ketchup bottle")
    adapt.add_argument("--place", default="pink tray")
    adapt.add_argument("--camera", default=os.environ.get("HACKSTORM_CAMERA", "orbbec"))
    adapt.add_argument("--backend", default=os.environ.get("HACKSTORM_DETECT_BACKEND", "grounding_dino"))
    adapt.add_argument("--pick-x", type=float, default=None)
    adapt.add_argument("--pick-y", type=float, default=None)
    adapt.add_argument("--pick-z", type=float, default=None)
    adapt.add_argument("--place-x", type=float, default=None)
    adapt.add_argument("--place-y", type=float, default=None)
    adapt.add_argument("--place-z", type=float, default=None)
    adapt.add_argument("--speed", type=int, default=None)
    adapt.add_argument("--time-scale", type=float, default=1.0)

    adapt_xy = sub.add_parser(
        "play-adapt-xy",
        help="DINO shifts X+Y on taught path; Z locked from recording",
    )
    adapt_xy.add_argument("name", nargs="?", default="ketchup_demo")
    adapt_xy.add_argument("-f", "--file", default=None)
    adapt_xy.add_argument("--pick", default="ketchup bottle")
    adapt_xy.add_argument("--place", default="pink tray")
    adapt_xy.add_argument("--camera", default=os.environ.get("HACKSTORM_CAMERA", "orbbec"))
    adapt_xy.add_argument("--backend", default=os.environ.get("HACKSTORM_DETECT_BACKEND", "grounding_dino"))
    adapt_xy.add_argument("--pick-x", type=float, default=None)
    adapt_xy.add_argument("--pick-y", type=float, default=None)
    adapt_xy.add_argument("--place-x", type=float, default=None)
    adapt_xy.add_argument("--place-y", type=float, default=None)
    adapt_xy.add_argument("--speed", type=int, default=None)
    adapt_xy.add_argument("--time-scale", type=float, default=1.0)

    sub.add_parser("list", help="List recordings")

    init_seq = sub.add_parser("init-sequence", help="Create bit sequence template")
    init_seq.add_argument("name", nargs="?", default="ketchup")

    rec_bit = sub.add_parser("record-bit", help="Record one short motion bit")
    rec_bit.add_argument("name")
    rec_bit.add_argument(
        "--role",
        choices=("none", "pick", "place", "transit"),
        default="transit",
    )
    rec_bit.add_argument("-o", "--output", default=None)
    rec_bit.add_argument("--rate", type=float, default=10.0)
    rec_bit.add_argument("--threshold-deg", type=float, default=0.5)
    rec_bit.add_argument("--speed", type=int, default=30)

    bits = sub.add_parser("play-bits", help="Replay bit sequence (no vision)")
    bits.add_argument("name", nargs="?", default="ketchup")
    bits.add_argument("-f", "--file", default=None)
    bits.add_argument("--speed", type=int, default=None)
    bits.add_argument("--time-scale", type=float, default=1.0)

    bits_ad = sub.add_parser("play-bits-adapt", help="Vision-adapted bit sequence")
    bits_ad.add_argument("name", nargs="?", default="ketchup")
    bits_ad.add_argument("-f", "--file", default=None)
    bits_ad.add_argument("--pick", default=None)
    bits_ad.add_argument("--place", default=None)
    bits_ad.add_argument("--camera", default=os.environ.get("HACKSTORM_CAMERA", "orbbec"))
    bits_ad.add_argument("--backend", default=os.environ.get("HACKSTORM_DETECT_BACKEND", "grounding_dino"))
    bits_ad.add_argument("--pick-x", type=float, default=None)
    bits_ad.add_argument("--pick-y", type=float, default=None)
    bits_ad.add_argument("--pick-z", type=float, default=None)
    bits_ad.add_argument("--place-x", type=float, default=None)
    bits_ad.add_argument("--place-y", type=float, default=None)
    bits_ad.add_argument("--place-z", type=float, default=None)
    bits_ad.add_argument("--speed", type=int, default=None)
    bits_ad.add_argument("--time-scale", type=float, default=1.0)

    bits_xy = sub.add_parser("play-bits-adapt-xy", help="Bit sequence + DINO X/Y (Z locked)")
    bits_xy.add_argument("name", nargs="?", default="ketchup")
    bits_xy.add_argument("-f", "--file", default=None)
    bits_xy.add_argument("--pick", default=None)
    bits_xy.add_argument("--place", default=None)
    bits_xy.add_argument("--camera", default=os.environ.get("HACKSTORM_CAMERA", "orbbec"))
    bits_xy.add_argument("--backend", default=os.environ.get("HACKSTORM_DETECT_BACKEND", "grounding_dino"))
    bits_xy.add_argument("--pick-x", type=float, default=None)
    bits_xy.add_argument("--pick-y", type=float, default=None)
    bits_xy.add_argument("--place-x", type=float, default=None)
    bits_xy.add_argument("--place-y", type=float, default=None)
    bits_xy.add_argument("--speed", type=int, default=None)
    bits_xy.add_argument("--time-scale", type=float, default=1.0)

    args = parser.parse_args(argv)
    if args.command == "record":
        return cmd_record(args)
    if args.command == "play":
        return cmd_play(args)
    if args.command == "play-adapt":
        return cmd_play_adapt(args)
    if args.command == "play-adapt-xy":
        return cmd_play_adapt_xy(args)
    if args.command == "init-sequence":
        return cmd_init_sequence(args)
    if args.command == "record-bit":
        return cmd_record_bit(args)
    if args.command == "play-bits":
        return cmd_play_bits(args)
    if args.command == "play-bits-adapt":
        return cmd_play_bits_adapt(args)
    if args.command == "play-bits-adapt-xy":
        return cmd_play_bits_adapt_xy(args)
    if args.command == "list":
        return cmd_list(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
