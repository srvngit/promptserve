#!/usr/bin/env python3
"""
Demo: synthesize speech and play on the default system output.

  uv run python demos/tts_speak.py "Hello from hackstorm"
  uv run python main.py "Hello from hackstorm"
"""

from __future__ import annotations

import argparse
import sys

from tts_utils import pcm_duration_s, play_pcm, speak, synthesize_pcm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTS → system speakers")
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello from hackstorm. TTS is working.",
        help="Text to speak",
    )
    parser.add_argument("--rate", type=int, default=16000, help="PCM sample rate Hz")
    parser.add_argument(
        "--backend",
        choices=("auto", "say", "piper", "stub"),
        default="auto",
        help="TTS engine (auto: piper env, then macOS say, else stub tone)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Synthesize only, do not play")
    args = parser.parse_args(argv)

    print(f"Synthesizing: {args.text!r}")
    if args.dry_run:
        result = synthesize_pcm(args.text, sample_rate_hz=args.rate, backend=args.backend)
    else:
        result = speak(args.text, sample_rate_hz=args.rate, backend=args.backend)
        print("Played on system speakers.")
        return 0

    if not result.pcm:
        print("error: synthesis produced empty PCM", file=sys.stderr)
        return 1
    duration_s = pcm_duration_s(result.pcm, sample_rate_hz=result.sample_rate_hz, channels=result.channels)
    print(
        f"PCM: {len(result.pcm)} bytes, {result.sample_rate_hz} Hz, "
        f"{result.channels} ch (~{duration_s:.2f}s)"
    )
    if not args.dry_run:
        play_pcm(result)
        print("Played on system speakers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
