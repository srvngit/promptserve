#!/usr/bin/env python3
"""
Direct Piper arm control — no voice, agent, or web UI.

Motion commands (home, move, smoke, gripper) run immediately over CAN.
grasp / place / demo use the detector (Grounding DINO by default) + Orbbec
only when you pass an object description; use --x --y --z to skip vision entirely.

Examples (Linux robot host, can0 @ 1 Mbps):
  ./scripts/piper_arm.sh --no-can-auto-init home
  ./scripts/piper_arm.sh --no-can-auto-init smoke          # prove arm moves
  ./scripts/piper_arm.sh --no-can-auto-init move --x 0.25 --y 0.0 --z 0.30
  ./scripts/piper_arm.sh --no-can-auto-init grasp --x 0.28 --y -0.05 --z 0.12
  ./scripts/piper_arm.sh --no-can-auto-init grasp "strawberry"
  ./scripts/piper_arm.sh --no-can-auto-init demo --pick "strawberry" --place "pink box"
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

from hackstorm.arm import PiperController, Pose

Vec3 = tuple[float, float, float]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Piper arm — motion first; detector runs only when you name an object.",
    )
    parser.add_argument("--can", default=None, help="SocketCAN (default PIPER_CAN / can0)")
    parser.add_argument(
        "--no-can-auto-init",
        action="store_true",
        help="can0 already UP → CreateCanBus(judge_flag=False)",
    )
    parser.add_argument("--can-auto-init", action="store_true")
    parser.add_argument("--workspace", type=str, default=None, help="workspace.yaml path")
    parser.add_argument(
        "--model",
        default=None,
        help="VLM model id (default: google/gemma-4-E2B-it)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("HACKSTORM_VLM_DEVICE"),
        help="VLM device: cpu (auto on ≤4 GiB GPU), cuda, mps; or HACKSTORM_VLM_DEVICE",
    )
    parser.add_argument(
        "--backend",
        choices=("grounding_dino", "yolo", "transformers", "ollama"),
        default=os.environ.get("HACKSTORM_DETECT_BACKEND", "grounding_dino"),
        help="detection backend (default: grounding_dino)",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_GEMMA_MODEL", "gemma4:2b"),
        help="Ollama model when --backend ollama",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--camera",
        default="orbbec",
        help=(
            "Camera source — Orbbec RGB-D only ('orbbec'/'orbbecsdk'/'rgbd'). "
            "V4L2 webcams are rejected: the IK path needs real depth."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("home", help="Move to home pose (no camera)")

    sub.add_parser("pose", help="Print live J6 end-pose + joints from arm feedback")

    sub.add_parser("joints", help="Print current joint angles (degrees, no motion)")

    sub.add_parser(
        "smoke",
        help="Sanity check: home → lift → gripper close/open → home (no camera)",
    )

    sub.add_parser("gripper-open", help="Open gripper")
    sub.add_parser("gripper-close", help="Close gripper")

    locate = sub.add_parser("locate", help="detector + depth → print arm-frame XYZ (m)")
    locate.add_argument("object", help='e.g. "strawberry"')

    grasp = sub.add_parser("grasp", help="Grasp at object (detector) or manual XYZ")
    grasp.add_argument("object", nargs="?", default=None, help="detector target description")
    grasp.add_argument("--x", type=float, default=None, help="manual meters (skips detector)")
    grasp.add_argument("--y", type=float, default=None)
    grasp.add_argument("--z", type=float, default=None)

    place = sub.add_parser("place", help="Place at object (detector) or manual XYZ")
    place.add_argument("object", nargs="?", default=None)
    place.add_argument("--x", type=float, default=None)
    place.add_argument("--y", type=float, default=None)
    place.add_argument("--z", type=float, default=None)

    move = sub.add_parser("move", help="Cartesian move (no detector)")
    move.add_argument("--x", type=float, required=True)
    move.add_argument("--y", type=float, required=True)
    move.add_argument("--z", type=float, required=True)

    demo = sub.add_parser("demo", help="Detector pick-and-place (single-scan)")
    demo.add_argument("--pick", required=True, help='e.g. "strawberry"')
    demo.add_argument("--place", required=True, help='e.g. "pink box"')
    demo.add_argument("--pick-x", type=float, default=None, help="skip vision for pick (meters)")
    demo.add_argument("--pick-y", type=float, default=None)
    demo.add_argument("--pick-z", type=float, default=None)
    demo.add_argument("--place-x", type=float, default=None, help="skip vision for place (meters)")
    demo.add_argument("--place-y", type=float, default=None)
    demo.add_argument("--place-z", type=float, default=None)

    cal = sub.add_parser(
        "calibrate-wrist",
        help="Wrist-mounted DaBai: fit T_cam_to_arm from N marker placements",
    )
    cal.add_argument(
        "--marker",
        default="small red sticky note",
        help='detector target placed under the gripper (e.g. "red sticker", "coin")',
    )
    cal.add_argument("--table-z", type=float, required=True, help="table surface Z in arm frame (m)")
    cal.add_argument(
        "--hover-z",
        type=float,
        default=None,
        help="hover Z above the marker (default: table_z + 0.10)",
    )
    cal.add_argument(
        "--points",
        nargs="+",
        metavar="X,Y",
        default=None,
        help=(
            "comma-separated arm XY pairs (meters) for each calibration spot;"
            " default = 4 shrunken corners of workspace bounds"
        ),
    )

    return parser


def _resolve_can_auto_init(args: argparse.Namespace) -> bool | None:
    if args.no_can_auto_init and args.can_auto_init:
        print("Use only one of --no-can-auto-init and --can-auto-init", file=sys.stderr)
        raise SystemExit(2)
    if args.no_can_auto_init:
        return False
    if args.can_auto_init:
        return True
    return None


def _make_controller(args: argparse.Namespace) -> PiperController:
    from pathlib import Path

    kwargs: dict = {"can_auto_init": _resolve_can_auto_init(args)}
    if args.can:
        kwargs["can_name"] = args.can
    if args.workspace:
        kwargs["workspace_path"] = Path(args.workspace)
    return PiperController(**kwargs)


def _manual_xyz(args: argparse.Namespace) -> Vec3 | None:
    if args.x is None and args.y is None and args.z is None:
        return None
    if args.x is None or args.y is None or args.z is None:
        print("Manual override requires --x, --y, and --z together", file=sys.stderr)
        raise SystemExit(2)
    return (args.x, args.y, args.z)


def _needs_vision(args: argparse.Namespace) -> bool:
    if args.command == "locate":
        return True
    if args.command == "demo":
        return True
    if args.command in ("grasp", "place") and _manual_xyz(args) is None:
        return bool(getattr(args, "object", None) or getattr(args, "pick", None))
    return False


_ORBBEC_SOURCES = {"orbbec", "orbbecsdk", "rgbd"}


def _open_vision_camera(args: argparse.Namespace):
    from hackstorm.perception.open_camera import open_camera

    source = (args.camera or os.environ.get("HACKSTORM_CAMERA", "orbbec")).strip()
    if source.lower() not in _ORBBEC_SOURCES:
        print(
            f"error: camera source {source!r} is not Orbbec RGB-D.\n"
            "  The piper-arm IK path requires real depth (no webcam fallback).\n"
            "  Use 'orbbec' (or unset --camera / HACKSTORM_CAMERA).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    camera = open_camera(source)
    camera.open()
    logging.info("Vision camera: %s (Orbbec RGB-D)", source)
    return camera


def _vlm_kwargs(args: argparse.Namespace) -> dict:
    from hackstorm.vision.detector import resolve_detection_backend

    kwargs: dict = {"backend": resolve_detection_backend(args.backend)}
    if args.model:
        kwargs["model_id"] = args.model
    if args.device:
        kwargs["device"] = args.device
    if args.ollama_model:
        kwargs["ollama_model"] = args.ollama_model
    if args.ollama_host:
        kwargs["ollama_host"] = args.ollama_host
    return kwargs


def _resolve_target_vlm(
    args: argparse.Namespace,
    *,
    description: str,
    arm: PiperController | None = None,
) -> Vec3:
    """Capture from the Orbbec at home pose and locate ``description`` in arm frame.

    Real depth required — no default_z_m fallback. If the Orbbec returns 0 mm
    at the detection pixel, the call fails loud instead of guessing a Z.
    """
    from hackstorm.perception.locate import locate_object

    if arm is not None:
        print("[scan] going home before capture (static T_cam_to_arm assumes home pose)", flush=True)
        arm.go_home()

    camera = _open_vision_camera(args)
    try:
        print("[scan] settling exposure (5 frames discarded)...", flush=True)
        for _ in range(5):
            camera.capture()
        kwargs = _vlm_kwargs(args)
        kwargs["default_z_m"] = None  # depth-required; no XY/Z hardcoding
        point = locate_object(description, camera, **kwargs)
    finally:
        camera.close()

    if point is None:
        print(
            f"Could not locate {description!r} with real depth — arm did NOT move.\n"
            "  Try: better lighting, a less reflective object, or check the\n"
            "  captures/debug/ image. The Orbbec returned 0 mm at the bbox.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    logging.info(
        "detector(%s) located %r at arm-frame (%.3f, %.3f, %.3f) m",
        args.backend, description, *point,
    )
    return point


def _manual_vec3(x, y, z, *, label: str) -> Vec3 | None:
    if x is None and y is None and z is None:
        return None
    if x is None or y is None or z is None:
        print(f"Manual {label} requires all of --{label}-x/y/z", file=sys.stderr)
        raise SystemExit(2)
    return (x, y, z)


def _resolve_demo_targets(
    args: argparse.Namespace,
    *,
    pick: str,
    place: str,
) -> tuple[Vec3, Vec3]:
    """Vision scan and/or manual XYZ overrides for demo pick + place."""
    manual_pick = _manual_vec3(
        args.pick_x, args.pick_y, args.pick_z, label="pick"
    )
    manual_place = _manual_vec3(
        args.place_x, args.place_y, args.place_z, label="place"
    )
    if manual_pick is not None and manual_place is not None:
        print(f"[demo] manual pick {manual_pick}  place {manual_place}", flush=True)
        return manual_pick, manual_place
    if manual_pick is not None or manual_place is not None:
        print("Provide both manual pick AND place coords, or neither (vision scan).", file=sys.stderr)
        raise SystemExit(2)
    return _resolve_pair_from_single_scan(args, pick=pick, place=place)


def _resolve_pair_from_single_scan(
    args: argparse.Namespace,
    *,
    pick: str,
    place: str,
) -> tuple[Vec3, Vec3]:
    """One capture at the home pose, two detector runs (pick + place).

    Real depth required — no default_z_m fallback so missing depth fails loud.
    """
    from hackstorm.perception.locate import locate_in_frame

    camera = _open_vision_camera(args)
    try:
        print("[scan] settling exposure (15 frames discarded)...", flush=True)
        for _ in range(15):
            camera.capture()
            time.sleep(0.08)
        frame = camera.capture()
        try:
            from PIL import Image

            out = Path("captures/demo_scan.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.color_rgb).save(out)
            print(f"[scan] saved {out}", flush=True)
        except Exception:
            logging.exception("could not save demo scan frame")
    finally:
        camera.close()

    h, w = frame.color_rgb.shape[:2]
    nonzero = int((frame.depth_mm > 0).sum())
    total = h * w
    print(
        f"[scan] frame {w}x{h}  depth pixels valid: {nonzero}/{total} "
        f"({100.0 * nonzero / total:.1f}%)  intrinsics fx={frame.intrinsics.fx:.0f} "
        f"fy={frame.intrinsics.fy:.0f}",
        flush=True,
    )
    if nonzero == 0:
        table_z = float(os.environ.get("HACKSTORM_TABLE_Z_M", "0.12"))
        print(
            f"[scan] WARN: depth map empty — using table Z={table_z:.3f} m for pick/place "
            "(point camera at table; set HACKSTORM_TABLE_Z_M if grasp height is wrong).",
            flush=True,
        )
        kwargs = _vlm_kwargs(args)
        kwargs["default_z_m"] = table_z
        pick_pt = locate_in_frame(pick, frame, **kwargs)
        place_pt = locate_in_frame(place, frame, **kwargs)
        missing = [name for name, pt in (("pick", pick_pt), ("place", place_pt)) if pt is None]
        if missing:
            print(
                f"Scan missing target(s): {', '.join(missing)} — see captures/debug/.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        assert pick_pt is not None and place_pt is not None
        logging.info("Scan (table-Z fallback): pick %r at %s  place %r at %s", pick, pick_pt, place, place_pt)
        return pick_pt, place_pt

    if nonzero / total < 0.10:
        table_z = float(os.environ.get("HACKSTORM_TABLE_Z_M", "0.12"))
        print(
            f"[scan] WARN: sparse depth ({100.0 * nonzero / total:.1f}%) — "
            f"using table Z={table_z:.3f} m where depth missing.",
            flush=True,
        )
        default_z = table_z
    else:
        default_z = None

    kwargs = _vlm_kwargs(args)
    kwargs["default_z_m"] = default_z

    pick_pt = locate_in_frame(pick, frame, **kwargs)
    place_pt = locate_in_frame(place, frame, **kwargs)

    missing = [name for name, pt in (("pick", pick_pt), ("place", place_pt)) if pt is None]
    if missing:
        print(
            f"Scan missing target(s): {', '.join(missing)} — see captures/debug/.\n"
            "  Reposition the camera or rename the target and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    assert pick_pt is not None and place_pt is not None
    logging.info("Scan: pick %r at %s  place %r at %s", pick, pick_pt, place, place_pt)
    return pick_pt, place_pt


def _resolve_target(
    args: argparse.Namespace,
    *,
    description: str | None,
    arm: PiperController | None = None,
) -> Vec3:
    manual = _manual_xyz(args)
    if manual is not None:
        logging.info("Manual XYZ (no detector): %s", manual)
        return manual
    if not description:
        print("Pass an object description or --x --y --z", file=sys.stderr)
        raise SystemExit(2)
    return _resolve_target_vlm(args, description=description, arm=arm)


def _parse_xy_pair(spec: str) -> tuple[float, float]:
    parts = spec.replace(" ", "").split(",")
    if len(parts) != 2:
        raise SystemExit(f"--points entry must be 'X,Y' in meters; got {spec!r}")
    return float(parts[0]), float(parts[1])


def _calibration_points(args, workspace) -> list[tuple[float, float]]:
    if args.points:
        return [_parse_xy_pair(p) for p in args.points]
    b = workspace.bounds
    inset = 0.05
    return [
        (b.x_min + inset, b.y_min + inset),
        (b.x_max - inset, b.y_min + inset),
        (b.x_max - inset, b.y_max - inset),
        (b.x_min + inset, b.y_max - inset),
    ]


def _run_wrist_calibration(args, arm) -> int:
    """Interactive: place a marker at N known arm-XY positions; fit T_cam_to_arm."""
    import numpy as np
    import yaml

    from hackstorm.perception.calibrate import solve_from_pairs_mm
    from hackstorm.perception.frames import (
        depth_median_bbox,
        depth_median_patch,
        pixel_depth_to_camera_point,
    )
    from hackstorm.vision.detector import detect_with_raw, pick_best

    points = _calibration_points(args, arm.workspace)
    table_z = args.table_z
    hover_z = args.hover_z if args.hover_z is not None else (table_z + 0.10)
    home = arm.workspace.home_pose

    print(f"\n=== wrist-camera calibration ===")
    print(f"marker   : {args.marker!r}")
    print(f"table_z  : {table_z:.3f} m   hover_z: {hover_z:.3f} m")
    print(f"points   : {len(points)}")
    for i, (x, y) in enumerate(points, 1):
        print(f"  {i}: ({x:.3f}, {y:.3f})")

    pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    detector_kwargs = _vlm_kwargs(args)

    arm.go_home()
    camera = _open_vision_camera(args)
    try:
        for i, (x, y) in enumerate(points, 1):
            print(f"\n--- point {i}/{len(points)}  arm XY=({x:.3f}, {y:.3f}) ---")
            hover_pose = Pose(x, y, hover_z, home.rx, home.ry, home.rz)
            try:
                orient = arm._grasp_orient()  # noqa: SLF001 — calibration uses live RPY
                hover_pose = Pose(x, y, hover_z, orient.rx, orient.ry, orient.rz)
            except Exception:
                pass
            arm.move_to(hover_pose)
            input(
                "  Arm is hovering. Place the marker EXACTLY under the gripper tip.\n"
                "  Press ENTER when placed (arm will retreat to scan pose)... "
            )
            arm.go_home()

            print("  Settling exposure...")
            for _ in range(5):
                camera.capture()
            frame = camera.capture()

            boxes, _raw = detect_with_raw(frame.color_rgb, args.marker, **detector_kwargs)
            target = pick_best(boxes, args.marker)
            if target is None:
                print(f"  detector found nothing matching {args.marker!r} — skipping point")
                continue

            u = int(round(target.center_xy[0]))
            v = int(round(target.center_xy[1]))
            depth_mm = depth_median_bbox(frame.depth_mm, target.center_xy, target.size_wh)
            if depth_mm <= 0:
                depth_mm = depth_median_patch(frame.depth_mm, u, v)
            if depth_mm <= 0:
                print("  no valid depth at marker — skipping point (try a less reflective marker)")
                continue

            p_cam_mm = pixel_depth_to_camera_point(u, v, depth_mm, frame.intrinsics)
            p_arm_mm = (x * 1000.0, y * 1000.0, table_z * 1000.0)
            print(
                f"  pixel=({u},{v}) depth={int(depth_mm)} mm  "
                f"cam=({p_cam_mm[0]:.0f},{p_cam_mm[1]:.0f},{p_cam_mm[2]:.0f}) mm  "
                f"arm=({p_arm_mm[0]:.0f},{p_arm_mm[1]:.0f},{p_arm_mm[2]:.0f}) mm"
            )
            pairs.append((p_cam_mm, p_arm_mm))
    finally:
        camera.close()
        try:
            arm.go_home()
        except Exception:
            logging.exception("go_home after calibration")

    if len(pairs) < 3:
        print(f"\nCalibration failed: need ≥3 valid points, got {len(pairs)}.", file=sys.stderr)
        return 1

    transform, residuals = solve_from_pairs_mm(pairs)
    mean_err = float(np.mean(residuals))
    max_err = float(np.max(residuals))
    print(f"\n=== fit ===")
    print(f"points used : {len(pairs)}")
    print(f"residuals   : mean={mean_err:.1f} mm  max={max_err:.1f} mm")
    print("T_cam_to_arm (mm):")
    for row in transform:
        print(f"  [{row[0]: .4f}, {row[1]: .4f}, {row[2]: .4f}, {row[3]: .2f}]")

    config_path = Path("config/camera_arm.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["T_cam_to_arm"] = [[float(v) for v in row] for row in transform.tolist()]
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(f"\nwrote {config_path}")
    if max_err > 20:
        print(f"WARN: max residual {max_err:.0f} mm > 20 mm — recalibrate with more points or steadier marker.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command is None:
        print(
            "No command — running smoke (home → lift → gripper → home). "
            "Try: smoke | home | move | demo | --help",
            flush=True,
        )
        args.command = "smoke"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # locate = vision only, no arm (Orbbec wrist mount: T_cam_to_arm assumes
    # the wrist is at the configured home pose during capture).
    if args.command == "locate":
        pt = _resolve_target_vlm(args, description=args.object, arm=None)
        print(f"ok: {args.object!r} at x={pt[0]:.3f} y={pt[1]:.3f} z={pt[2]:.3f} m")
        return 0

    if args.command == "demo":
        from hackstorm.vision.detector import resolve_detection_backend

        if resolve_detection_backend(args.backend) == "grounding_dino":
            from hackstorm.vision.grounding_dino_detector import prepare_grounding_dino

            print("[demo] loading Grounding DINO (first run downloads weights)...", flush=True)
            prepare_grounding_dino(device=args.device)

    if _needs_vision(args) and args.command != "demo":
        logging.info("Vision path — loading camera/detector for %s", args.command)

    arm = _make_controller(args)
    can_name = args.can or os.environ.get("PIPER_CAN", "can0")
    print(f"Connecting Piper on {can_name}...", flush=True)
    arm.connect()
    print("Piper connected.", flush=True)

    home = arm.workspace.home_pose

    try:
        match args.command:
            case "home":
                arm.go_home()
                print("ok: home")
            case "pose":
                live = arm.read_current_pose()
                orient = arm._grasp_orient()  # noqa: SLF001
                joints = arm._read_current_joints_rad()  # noqa: SLF001
                print(
                    f"live end-pose: x={live.x:.3f} y={live.y:.3f} z={live.z:.3f} m  "
                    f"rpy=({live.rx:.1f}, {live.ry:.1f}, {live.rz:.1f}) deg"
                )
                print(
                    f"IK orient: rx={orient.rx:.1f} ry={orient.ry:.1f} rz={orient.rz:.1f} deg"
                )
                print(
                    f"joints deg: {[round(j * 57.2958, 2) for j in joints]}"
                )
            case "joints":
                joints = arm._read_current_joints_rad()  # noqa: SLF001
                print(f"joints deg: {[round(j * 57.2958, 2) for j in joints]}")
            case "smoke":
                cur = arm._read_current_joints_rad()  # noqa: SLF001
                nudge = list(cur)
                nudge[1] += math.radians(15)
                spd = arm.workspace.speed_percent
                print(f"ok: joints before {[round(j * 57.2958, 1) for j in cur]}", flush=True)
                arm._stream_joints(tuple(nudge), speed_percent=spd, label="smoke:nudge")  # noqa: SLF001
                arm.gripper_close()
                print("ok: gripper close")
                arm.gripper_open()
                print("ok: gripper open")
                arm._stream_joints(cur, speed_percent=spd, label="smoke:return")  # noqa: SLF001
                after = arm._read_current_joints_rad()  # noqa: SLF001
                print(f"ok: joints after {[round(j * 57.2958, 1) for j in after]}", flush=True)
                print("ok: smoke complete — arm moved")
            case "gripper-open":
                arm.gripper_open()
                print("ok: gripper open")
            case "gripper-close":
                arm.gripper_close()
                print("ok: gripper closed")
            case "move":
                arm.move_to(Pose(args.x, args.y, args.z, 0, 0, 0))
                print(f"ok: move ({args.x}, {args.y}, {args.z})")
            case "grasp":
                pt = _resolve_target(args, description=args.object, arm=arm)
                print(arm.grasp_at(pt))
            case "place":
                pt = _resolve_target(args, description=args.object, arm=arm)
                print(arm.place_at(pt))
            case "calibrate-wrist":
                return _run_wrist_calibration(args, arm)
            case "demo":
                # Scan ONCE at home pose, then execute. No re-detection during motion.
                arm.go_home()
                arm.gripper_open()
                # Lift slightly so wrist cam sees table (not a blurry close-up).
                scan = arm.workspace.home_pose
                scan_pose = Pose(scan.x, scan.y, scan.z + 0.08, scan.rx, scan.ry, scan.rz)
                print("[demo] lifting to scan height (+8 cm) for clearer wrist view...", flush=True)
                arm.move_to(scan_pose)
                print(
                    "ok: at scan pose — ketchup + pink tray should be in wrist camera view",
                    flush=True,
                )
                time.sleep(1.5)
                print(f"ok: capturing for {args.pick!r} + {args.place!r}")
                demo_pick, demo_place = _resolve_pair_from_single_scan(
                    args,
                    pick=args.pick,
                    place=args.place,
                )
                print(f"pick  → {demo_pick}")
                print(f"place → {demo_place}")
                print(arm.grasp_at(demo_pick))
                print(arm.place_at(demo_place))
                arm.go_home()
                print("ok: demo complete")
            case _:
                print(f"unknown command: {args.command}", file=sys.stderr)
                return 2
    finally:
        try:
            arm.gripper_open()
        except Exception:
            logging.exception("gripper open on exit")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        logging.exception("piper_arm failed")
        raise SystemExit(1)
