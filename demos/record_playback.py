#!/usr/bin/env python3
"""
Demo: record from the default system mic and play it back on the speakers.

  uv run python demos/record_playback.py
  uv run python demos/record_playback.py --seconds 5
"""

from __future__ import annotations

import argparse
import sys

from tts_utils import pcm_duration_s, play_pcm, record_pcm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record from system mic → play on speakers")
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Recording length in seconds (default: 3)",
    )
    parser.add_argument("--rate", type=int, default=16000, help="PCM sample rate Hz")
    args = parser.parse_args(argv)

    print(f"Recording {args.seconds:.1f}s from default mic...", flush=True)
    captured = record_pcm(args.seconds, sample_rate_hz=args.rate)
    duration_s = pcm_duration_s(
        captured.pcm,
        sample_rate_hz=captured.sample_rate_hz,
        channels=captured.channels,
    )
    print(f"Captured {len(captured.pcm)} bytes (~{duration_s:.2f}s). Playing back...", flush=True)
    play_pcm(captured)
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
