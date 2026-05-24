#!/usr/bin/env python3
"""Kokoro speak — same backend as the Strands agent ``speak`` tool."""

from __future__ import annotations

import argparse
import sys

from hackstorm.agent.kokoro_tts import speak_kokoro


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Speak text on system speakers (Kokoro TTS — agent speak tool backend)",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help='Words to speak (default: "Hello from hackstorm.")',
    )
    args = parser.parse_args(argv)
    text = " ".join(args.text).strip() or "Hello from hackstorm."
    print(f"[speak] {text!r}", flush=True)
    try:
        speak_kokoro(text)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
