"""Check NVIDIA driver + PyTorch CUDA before running VLM on the robot host."""

from __future__ import annotations

import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    ok = True

    _section("NVIDIA driver (nvidia-smi)")
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi: NOT FOUND")
        print("  Install NVIDIA drivers, or you are not on a GPU host.")
        ok = False
    else:
        code, out = _run(["nvidia-smi"])
        print(out or "(no output)")
        if code != 0:
            print(f"  exit {code} — driver may be missing or kernel module not loaded")
            ok = False

    _section("PyTorch")
    try:
        import torch
    except ImportError:
        print("torch not installed — run: uv sync")
        sys.exit(1)

    print(f"torch {torch.__version__}")
    print(f"torch.version.cuda (build): {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  cuda:{i} {name} ({mem_gb:.1f} GiB)")
        _section("CUDA smoke test")
        try:
            x = torch.ones(4, device="cuda")
            y = x + x
            torch.cuda.synchronize()
            print(f"tensor add on GPU: {y.cpu().tolist()} — OK")
        except Exception as exc:
            print(f"FAILED: {exc}")
            print("  Driver may be installed but broken, or CUDA/PyTorch versions mismatch.")
            ok = False
    else:
        ok = False
        print("\nPyTorch cannot see CUDA. Common causes:")
        print("  - NVIDIA driver not loaded (reboot after install, check dmesg)")
        print("  - CPU-only PyTorch wheel (reinstall with CUDA index)")
        print("  - Driver/CUDA version too old for this torch build")

    _section("Recommendation")
    if torch.cuda.is_available() and ok:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 6:
            print(f"GPU has {vram_gb:.1f} GiB — VLM will use CPU automatically (Gemma needs >4 GiB).")
            print("  export HACKSTORM_VLM_DEVICE=cpu   # explicit")
            print("  ./scripts/piper_arm.sh --no-can-auto-init --device cpu demo ...")
        else:
            print("GPU ready. Use default or:")
            print("  export HACKSTORM_VLM_DEVICE=cuda")
        print("  ./scripts/piper_arm.sh --no-can-auto-init demo --pick 'strawberry' --place 'box'")
    else:
        print("GPU not ready. Options:")
        print("  1. Fix driver (see above), then re-run: uv run hackstorm-vlm-check")
        print("  2. Run VLM on CPU (slow): export HACKSTORM_VLM_DEVICE=cpu")
        print("  3. Use Ollama on another machine if you have one")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
