"""Synthesize speech to 16-bit PCM for host playback."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    pcm: bytes
    sample_rate_hz: int
    channels: int


def _read_wav_pcm(path: Path, *, target_rate: int) -> SynthesisResult:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit WAV, got {wf.getsampwidth() * 8}-bit")
        rate = wf.getframerate()
        ch = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    if not pcm:
        raise ValueError("WAV contains no PCM frames")
    if rate != target_rate or ch != 1:
        raise ValueError(
            f"backend produced {rate} Hz {ch}ch; re-run with target {target_rate} Hz mono"
        )
    return SynthesisResult(pcm=pcm, sample_rate_hz=rate, channels=ch)


def _synthesize_macos_say(text: str, *, sample_rate_hz: int) -> SynthesisResult:
    say = shutil.which("say")
    afconvert = shutil.which("afconvert")
    if not say or not afconvert:
        raise RuntimeError("macOS say/afconvert not found")

    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "speech.aiff"
        wav = Path(tmp) / "speech.wav"
        subprocess.run([say, "-o", str(aiff), text], check=True, capture_output=True)
        subprocess.run(
            [
                afconvert,
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{sample_rate_hz}",
                "-c",
                "1",
                str(aiff),
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        return _read_wav_pcm(wav, target_rate=sample_rate_hz)


def _synthesize_piper(text: str, *, sample_rate_hz: int, piper_bin: str, model: str) -> SynthesisResult:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speech.wav"
        cmd = [piper_bin, "--model", model, "--output_file", str(wav)]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True)
        result = _read_wav_pcm(wav, target_rate=sample_rate_hz)
        if result.sample_rate_hz != sample_rate_hz:
            raise ValueError(
                f"piper model rate {result.sample_rate_hz} != {sample_rate_hz}; "
                "use a matching model or resample externally"
            )
        return result


def _synthesize_stub(text: str, *, sample_rate_hz: int) -> SynthesisResult:
    """Short tone burst so the demo runs without a TTS engine."""
    import array
    import math

    duration_s = min(2.0, 0.15 + len(text) * 0.03)
    n = int(sample_rate_hz * duration_s)
    freq = 440.0 + (len(text) % 12) * 20.0
    buf = array.array("h")
    for i in range(n):
        t = i / sample_rate_hz
        env = 1.0
        if t < 0.02:
            env = t / 0.02
        elif t > duration_s - 0.05:
            env = max(0.0, (duration_s - t) / 0.05)
        sample = int(12000 * env * math.sin(2 * math.pi * freq * t))
        buf.append(max(-32768, min(32767, sample)))
    return SynthesisResult(pcm=buf.tobytes(), sample_rate_hz=sample_rate_hz, channels=1)


def synthesize_pcm(
    text: str,
    *,
    sample_rate_hz: int = 16000,
    backend: str | None = None,
) -> SynthesisResult:
    """
    Return mono 16-bit little-endian PCM.

    Backend order (unless ``backend`` is set):
    1. ``piper`` if ``PIPER_EXECUTABLE`` + ``PIPER_MODEL`` are set
    2. ``say`` on macOS
    3. ``stub`` (beep) with a logged warning
    """
    text = text.strip()
    if not text:
        raise ValueError("text must be non-empty")

    chosen = (backend or os.environ.get("TTS_BACKEND", "")).lower().strip()

    if chosen in ("", "auto"):
        piper_bin = os.environ.get("PIPER_EXECUTABLE") or shutil.which("piper")
        model = os.environ.get("PIPER_MODEL", "")
        if piper_bin and model:
            try:
                return _synthesize_piper(
                    text, sample_rate_hz=sample_rate_hz, piper_bin=piper_bin, model=model
                )
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
        if sys.platform == "darwin":
            try:
                result = _synthesize_macos_say(text, sample_rate_hz=sample_rate_hz)
                if result.pcm:
                    return result
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
        import warnings

        warnings.warn("no TTS engine; using stub tone (set TTS_BACKEND=say on macOS)", stacklevel=2)
        return _synthesize_stub(text, sample_rate_hz=sample_rate_hz)

    if chosen == "say":
        return _synthesize_macos_say(text, sample_rate_hz=sample_rate_hz)
    if chosen == "piper":
        piper_bin = os.environ.get("PIPER_EXECUTABLE") or shutil.which("piper")
        model = os.environ.get("PIPER_MODEL", "")
        if not piper_bin or not model:
            raise ValueError("piper backend requires PIPER_EXECUTABLE and PIPER_MODEL")
        return _synthesize_piper(text, sample_rate_hz=sample_rate_hz, piper_bin=piper_bin, model=model)
    if chosen == "stub":
        return _synthesize_stub(text, sample_rate_hz=sample_rate_hz)

    raise ValueError(f"unknown TTS_BACKEND: {backend!r}")
