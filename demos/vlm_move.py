#!/usr/bin/env python3
"""
VLA-style loop: fresh wrist image → Qwen plans action chunk → execute → repeat until done.

  ./scripts/vlm_move.sh pick up the ketchup and put it in the pink tray
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

from hackstorm.arm import PiperController, Pose
from hackstorm.arm.piper_controller import load_workspace
from hackstorm.arm.piper_ik import JOINT_LIMITS_RAD
from hackstorm.perception.open_camera import open_camera
from hackstorm.vision.detector import run_ollama
from hackstorm.vision.qwen_vl import DEFAULT_QWEN_VL_MODEL, run_qwen_vl
from hackstorm.vision.vlm_motor import (
    LOOP_CONTEXT,
    MOTION_RETRY_SUFFIX,
    MOTOR_PROMPT,
    SINGLE_SHOT_CONTEXT,
    MotorAction,
    has_motion_actions,
    parse_motor_plan,
)

_MIN_HOLD_S = 5.0
_SETTLE_S = 1.0
_CAPTURE_DIR = Path("captures/vlm_loop")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VLA-style VLM loop → Piper arm")
    p.add_argument(
        "task",
        nargs=argparse.REMAINDER,
        help='e.g. "pick up the ketchup and put it in the pink tray"',
    )
    p.add_argument("--can", default=os.environ.get("PIPER_CAN", "can0"))
    p.add_argument("--camera", default=os.environ.get("HACKSTORM_CAMERA", "orbbec"))
    p.add_argument(
        "--backend",
        choices=("qwen", "ollama"),
        default=os.environ.get("HACKSTORM_VLM_BACKEND", "qwen"),
    )
    p.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_VL_MODEL", DEFAULT_QWEN_VL_MODEL),
    )
    p.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_GEMMA_MODEL", "gemma4:e2b"),
    )
    p.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("VLM_MAX_STEPS", "30")),
        help="max observe→act cycles (default 30)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="single observe→act cycle instead of looping until task_done",
    )
    p.add_argument("--dry-run", action="store_true", help="one plan from current image, no arm motion")
    p.add_argument("--home", action="store_true", help="return to home pose when finished")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _capture_fresh_frame(camera_name: str, step: int) -> object:
    """Open camera, grab one frame, close — avoids Orbbec USB crashes during arm motion."""
    last_err: Exception | None = None
    for attempt in range(1, 4):
        cam = open_camera(camera_name)
        try:
            cam.open()
            for _ in range(10):
                cam.capture()
                time.sleep(0.05)
            frame = cam.capture()
            _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            out = _CAPTURE_DIR / f"step_{step:03d}.png"
            try:
                from PIL import Image

                Image.fromarray(frame.color_rgb).save(out)
                print(f"[vla] frame step {step} → {out}", flush=True)
            except Exception as exc:
                print(f"[vla] WARN: could not save frame: {exc}", flush=True)
            return frame.color_rgb
        except Exception as exc:
            last_err = exc
            print(f"[vla] camera attempt {attempt}/3 failed: {exc}", flush=True)
            time.sleep(0.8)
        finally:
            try:
                cam.close()
            except Exception:
                pass
    msg = f"camera capture failed after 3 attempts (step {step})"
    raise RuntimeError(msg) from last_err


def _joints_deg_from_arm(arm: PiperController) -> list[float]:
    joints = arm._read_current_joints_rad()  # noqa: SLF001
    return [round(j * 180.0 / math.pi, 1) for j in joints]


def _clamp_joints_deg(joints_deg: tuple[float, ...]) -> tuple[float, ...]:
    out: list[float] = []
    for j, (lo, hi) in zip(joints_deg, JOINT_LIMITS_RAD, strict=True):
        lo_d = lo * 57.2957795
        hi_d = hi * 57.2957795
        out.append(min(max(j, lo_d), hi_d))
    return tuple(out)


def _format_prompt(
    *,
    task: str,
    joints_deg: list[float],
    workspace,
    step: int,
    max_steps: int,
    loop: bool,
    extra: str = "",
) -> str:
    b = workspace.bounds
    loop_context = (
        LOOP_CONTEXT.format(step=step, max_steps=max_steps)
        if loop
        else SINGLE_SHOT_CONTEXT
    )
    return MOTOR_PROMPT.format(
        task=task,
        joints_deg=joints_deg,
        x_min=b.x_min,
        x_max=b.x_max,
        y_min=b.y_min,
        y_max=b.y_max,
        z_min=b.z_min,
        z_max=b.z_max,
        loop_context=loop_context + extra,
    )


def _call_vlm(
    *,
    backend: str,
    image,
    prompt: str,
    step: int,
    qwen_model: str,
    ollama_model: str,
    ollama_host: str,
) -> str:
    from hackstorm.vision.detector import _ensure_pil_rgb

    pil = _ensure_pil_rgb(image)
    if backend == "qwen":
        print(f"[vla] step {step}: Qwen {qwen_model!r}...", flush=True)
        return run_qwen_vl(pil, model=qwen_model, prompt=prompt)
    print(f"[vla] step {step}: Ollama {ollama_model!r}...", flush=True)
    return run_ollama(pil, model=ollama_model, prompt=prompt, host=ollama_host)


def _ask_vlm_for_plan(
    *,
    backend: str,
    task: str,
    image,
    joints_deg: list[float],
    workspace,
    step: int,
    max_steps: int,
    loop: bool,
    qwen_model: str,
    ollama_model: str,
    ollama_host: str,
) -> tuple[str, bool, list[MotorAction], str]:
    prompt = _format_prompt(
        task=task,
        joints_deg=joints_deg,
        workspace=workspace,
        step=step,
        max_steps=max_steps,
        loop=loop,
    )
    raw = _call_vlm(
        backend=backend,
        image=image,
        prompt=prompt,
        step=step,
        qwen_model=qwen_model,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        print(f"[vla] raw response:\n{raw}", flush=True)

    try:
        reasoning, task_done, actions = parse_motor_plan(raw)
    except Exception as exc:
        print(f"[vla] parse failed: {exc} — retrying VLM once", flush=True)
        retry_prompt = prompt + "\n\nReturn valid ```json``` only. Previous response was invalid."
        raw = _call_vlm(
            backend=backend,
            image=image,
            prompt=retry_prompt,
            step=step,
            qwen_model=qwen_model,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
        )
        reasoning, task_done, actions = parse_motor_plan(raw)

    if not task_done and actions and not has_motion_actions(actions):
        print("[vla] plan has no motion — re-asking Qwen", flush=True)
        fix = MOTION_RETRY_SUFFIX.format(joints_deg=joints_deg)
        raw = _call_vlm(
            backend=backend,
            image=image,
            prompt=prompt + "\n\n" + fix,
            step=step,
            qwen_model=qwen_model,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
        )
        reasoning, task_done, actions = parse_motor_plan(raw)

    return reasoning, task_done, actions, raw


def _hold_for_joints(current_deg: list[float], target_deg: tuple[float, ...], base_hold: float) -> float:
    delta = max(abs(c - t) for c, t in zip(current_deg, target_deg, strict=True))
    return max(_MIN_HOLD_S, base_hold, 2.0 + delta / 10.0)


def _execute_one_action(
    arm: PiperController,
    act: MotorAction,
    *,
    workspace,
    step: int,
    idx: int,
    allow_home: bool,
) -> bool:
    """Run one action. Returns False if motion failed (IK etc.) but loop should continue."""
    spd = max(workspace.speed_percent, 50)
    orient = arm._grasp_orient()  # noqa: SLF001
    before = _joints_deg_from_arm(arm)
    print(f"[vla] step {step} act {idx}: {act.type}  joints={before}", flush=True)

    try:
        match act.type:
            case "joints_deg":
                assert act.joints_deg is not None
                target = _clamp_joints_deg(act.joints_deg)
                hold = _hold_for_joints(before, target, act.hold_s)
                rad = tuple(j * math.pi / 180.0 for j in target)
                arm._stream_joints(  # noqa: SLF001
                    rad,
                    speed_percent=spd,
                    hold_s=hold,
                    label=f"vla:s{step}:j{idx}",
                )
            case "xyz_m":
                assert act.x is not None and act.y is not None and act.z is not None
                hold = max(_MIN_HOLD_S, act.hold_s)
                pose = Pose(act.x, act.y, act.z, orient.rx, orient.ry, orient.rz)
                arm._move_cartesian(  # noqa: SLF001
                    pose,
                    speed_percent=spd,
                    hold_s=hold,
                    label=f"vla:s{step}:xyz{idx}",
                )
            case "gripper":
                if act.gripper == "close":
                    arm.gripper_close()
                else:
                    arm.gripper_open()
            case "home":
                if allow_home:
                    arm.go_home()
                else:
                    print("[vla] skipping home (pass --home)", flush=True)
            case "wait":
                time.sleep(max(act.wait_s or 0.0, 0.0))
            case _:
                print(f"[vla] skip unknown {act.type!r}", flush=True)
    except Exception as exc:
        print(f"[vla] action failed ({act.type}): {exc}", flush=True)
        arm.ensure_motion_mode()
        return False

    after = _joints_deg_from_arm(arm)
    if act.type in ("joints_deg", "xyz_m"):
        moved = max(abs(a - b) for a, b in zip(before, after, strict=True))
        print(f"[vla] after act {idx}: joints={after}  Δ={moved:.1f}°", flush=True)
        if moved < 0.5 and act.type == "joints_deg":
            print("[vla] barely moved — retry once", flush=True)
            arm.ensure_motion_mode()
            assert act.joints_deg is not None
            target = _clamp_joints_deg(act.joints_deg)
            hold = max(_hold_for_joints(before, target, act.hold_s), 8.0)
            rad = tuple(j * math.pi / 180.0 for j in target)
            try:
                arm._stream_joints(rad, speed_percent=spd, hold_s=hold, label=f"vla:s{step}:retry{idx}")  # noqa: SLF001
            except Exception as exc:
                print(f"[vla] retry failed: {exc}", flush=True)
                return False
    return True


def _execute_actions(
    arm: PiperController,
    actions: list[MotorAction],
    *,
    workspace,
    step: int,
    allow_home: bool,
) -> None:
    arm.ensure_motion_mode()
    for i, act in enumerate(actions, start=1):
        if act.type in ("joints_deg", "xyz_m", "home"):
            arm.pause_position_hold()
            try:
                _execute_one_action(
                    arm,
                    act,
                    workspace=workspace,
                    step=step,
                    idx=i,
                    allow_home=allow_home,
                )
            finally:
                arm.resume_position_hold()
        else:
            _execute_one_action(
                arm,
                act,
                workspace=workspace,
                step=step,
                idx=i,
                allow_home=allow_home,
            )


def _run_vla_loop(
    *,
    arm: PiperController,
    camera_name: str,
    task: str,
    workspace,
    backend: str,
    qwen_model: str,
    ollama_model: str,
    ollama_host: str,
    max_steps: int,
    loop: bool,
    allow_home: bool,
) -> None:
    errors = 0

    for step in range(1, max_steps + 1):
        try:
            image = _capture_fresh_frame(camera_name, step)
            joints_deg = _joints_deg_from_arm(arm)
            print(f"[vla] step {step} observe — joints={joints_deg}", flush=True)

            reasoning, task_done, actions, raw = _ask_vlm_for_plan(
                backend=backend,
                task=task,
                image=image,
                joints_deg=joints_deg,
                workspace=workspace,
                step=step,
                max_steps=max_steps,
                loop=loop,
                qwen_model=qwen_model,
                ollama_model=ollama_model,
                ollama_host=ollama_host,
            )

            plan_path = _CAPTURE_DIR / f"step_{step:03d}_plan.json"
            plan_path.write_text(raw, encoding="utf-8")
            print(f"[vla] step {step} plan: {reasoning}")
            for i, a in enumerate(actions, 1):
                print(f"  {i}. {a}")

            if task_done:
                print(f"[vla] task_done at step {step}", flush=True)
                break

            if not actions:
                print(f"[vla] step {step}: empty plan — continuing", flush=True)
                continue

            if not has_motion_actions(actions):
                print(f"[vla] step {step}: still no motion in plan — continuing", flush=True)
                continue

            _execute_actions(arm, actions, workspace=workspace, step=step, allow_home=allow_home)
            time.sleep(_SETTLE_S)

            if not loop:
                break
        except Exception as exc:
            errors += 1
            print(f"[vla] step {step} crashed: {exc}", flush=True)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                traceback.print_exc()
            try:
                arm.ensure_motion_mode()
            except Exception:
                pass
            if errors >= 5:
                print("[vla] too many step errors — stopping", flush=True)
                break
            time.sleep(1.0)
            continue
    else:
        print(f"[vla] reached max steps ({max_steps}) without task_done", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    task = " ".join(args.task).strip()
    if not task:
        task = "pick up the ketchup and put it in the pink tray"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    workspace = load_workspace()
    loop = not args.once

    if args.dry_run:
        image = _capture_fresh_frame(args.camera, 1)
        joints_deg = [0.0] * 6
        print("[vla] dry-run — zero joints in prompt", flush=True)
        reasoning, task_done, actions, raw = _ask_vlm_for_plan(
            backend=args.backend,
            task=task,
            image=image,
            joints_deg=joints_deg,
            workspace=workspace,
            step=1,
            max_steps=args.max_steps,
            loop=loop,
            qwen_model=args.qwen_model,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
        )
        print(f"[vla] reasoning: {reasoning}  task_done={task_done}")
        for i, a in enumerate(actions, 1):
            print(f"  {i}. {a}")
        (_CAPTURE_DIR / "step_001_plan.json").write_text(raw, encoding="utf-8")
        print("ok: dry-run saved captures/vlm_loop/")
        return 0

    arm = PiperController(can_name=args.can)
    print(f"[vla] connecting Piper on {args.can}...", flush=True)
    arm.connect()
    arm.start_position_hold()
    print(f"[vla] live joints: {_joints_deg_from_arm(arm)}", flush=True)

    try:
        _run_vla_loop(
            arm=arm,
            camera_name=args.camera,
            task=task,
            workspace=workspace,
            backend=args.backend,
            qwen_model=args.qwen_model,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            max_steps=args.max_steps,
            loop=loop,
            allow_home=args.home,
        )

        if args.home:
            arm.pause_position_hold()
            try:
                arm.go_home()
            except Exception as exc:
                print(f"[vla] go_home failed: {exc}", flush=True)
            finally:
                arm.resume_position_hold()
    finally:
        arm.stop_position_hold()

    print("ok: vla loop complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        logging.exception("vlm_move failed")
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
