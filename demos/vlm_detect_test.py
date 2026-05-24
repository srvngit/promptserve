"""Quick detection test — no arm. Use to debug camera + detector."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from hackstorm.vision.detector import DEFAULT_MODEL, detect_with_raw, pick_best
from hackstorm.vision.grounding_dino_detector import DEFAULT_DINO_MODEL
from hackstorm.vision.yolo_detector import DEFAULT_YOLO_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test object detect on one webcam frame (no arm).")
    parser.add_argument("object", help='e.g. "strawberry"')
    parser.add_argument("--camera", default=os.environ.get("HACKSTORM_CAMERA"))
    parser.add_argument("--model", default=None, help=f"YOLO or HF model (default: {DEFAULT_YOLO_MODEL})")
    parser.add_argument(
        "--backend",
        choices=("grounding_dino", "yolo", "transformers", "ollama"),
        default=os.environ.get("HACKSTORM_DETECT_BACKEND", "grounding_dino"),
    )
    parser.add_argument("--device", default=os.environ.get("HACKSTORM_VLM_DEVICE"))
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_GEMMA_MODEL", "gemma4:2b"))
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=float(os.environ.get("HACKSTORM_YOLO_CONF", "0.12")),
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=float(os.environ.get("HACKSTORM_DINO_BOX_THRESHOLD", "0.28")),
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=float(os.environ.get("HACKSTORM_DINO_TEXT_THRESHOLD", "0.22")),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    model_id = args.model
    if model_id is None:
        if args.backend == "yolo":
            model_id = DEFAULT_YOLO_MODEL
        elif args.backend == "grounding_dino":
            model_id = DEFAULT_DINO_MODEL
        else:
            model_id = DEFAULT_MODEL

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from hackstorm.perception.open_camera import default_camera_source, open_camera

    camera = args.camera or default_camera_source()
    print(f"Opening camera {camera}...", flush=True)
    cam = open_camera(camera)
    cam.open()
    try:
        frame = cam.capture()
        print(f"Frame {frame.color_rgb.shape[1]}x{frame.color_rgb.shape[0]}", flush=True)
        print(f"backend={args.backend} — loading model...", flush=True)
        boxes, raw = detect_with_raw(
            frame.color_rgb,
            args.object,
            backend=args.backend,
            model_id=model_id,
            device=args.device,
            ollama_model=args.ollama_model,
            yolo_conf=args.yolo_conf,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    finally:
        cam.close()

    target = pick_best(boxes, args.object)
    print(f"detections: {len(boxes)}", flush=True)
    if args.verbose:
        print(raw, file=sys.stderr)
    if target:
        print(
            f"ok: {args.object!r} center=({target.center_xy[0]:.0f}, {target.center_xy[1]:.0f})",
            flush=True,
        )
        return 0
    print(f"no match for {args.object!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("vlm-detect-test failed")
        raise SystemExit(1)
