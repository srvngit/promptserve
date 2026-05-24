"""Host microphone and speaker I/O (system default devices)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave

from tts_utils.synthesize import SynthesisResult, synthesize_pcm

DEFAULT_RATE_HZ = 16_000
DEFAULT_CHANNELS = 1


def pcm_duration_s(
    pcm: bytes,
    *,
    sample_rate_hz: int = DEFAULT_RATE_HZ,
    channels: int = DEFAULT_CHANNELS,
) -> float:
    return len(pcm) / (2 * channels * sample_rate_hz)


def _write_wav(path: str, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm)


def _play_pcm_subprocess(pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        _write_wav(path, pcm, sample_rate_hz=sample_rate_hz, channels=channels)
        if sys.platform == "darwin":
            subprocess.run(["afplay", path], check=True)
            return
        for cmd in (("aplay", "-q", path), ("paplay", path)):
            if shutil.which(cmd[0]):
                subprocess.run(list(cmd), check=True)
                return
        raise RuntimeError("no audio player found (install sounddevice, or afplay/aplay/paplay)")
    finally:
        os.unlink(path)


def _play_pcm_sounddevice(pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
    import numpy as np
    import sounddevice as sd

    samples = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    sd.play(samples, samplerate=sample_rate_hz, blocking=True)


def play_pcm(
    audio: SynthesisResult | bytes,
    *,
    sample_rate_hz: int = DEFAULT_RATE_HZ,
    channels: int = DEFAULT_CHANNELS,
) -> None:
    """Play mono 16-bit PCM on the default system output device."""
    if isinstance(audio, SynthesisResult):
        pcm = audio.pcm
        sample_rate_hz = audio.sample_rate_hz
        channels = audio.channels
    else:
        pcm = audio

    if not pcm:
        raise ValueError("PCM is empty")

    try:
        _play_pcm_sounddevice(pcm, sample_rate_hz=sample_rate_hz, channels=channels)
    except (ImportError, OSError, RuntimeError):
        _play_pcm_subprocess(pcm, sample_rate_hz=sample_rate_hz, channels=channels)


def record_pcm(
    duration_s: float,
    *,
    sample_rate_hz: int = DEFAULT_RATE_HZ,
    channels: int = DEFAULT_CHANNELS,
    device: int | str | None = None,
) -> SynthesisResult:
    """Record from the default system input device."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    import sounddevice as sd

    frames = int(duration_s * sample_rate_hz)
    recording = sd.rec(
        frames,
        samplerate=sample_rate_hz,
        channels=channels,
        dtype="int16",
        device=device,
    )
    sd.wait()
    return SynthesisResult(
        pcm=recording.tobytes(),
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


def speak(
    text: str,
    *,
    sample_rate_hz: int = DEFAULT_RATE_HZ,
    backend: str | None = None,
) -> SynthesisResult:
    """Synthesize text and play it on the default system output."""
    result = synthesize_pcm(text, sample_rate_hz=sample_rate_hz, backend=backend)
    play_pcm(result)
    return result
