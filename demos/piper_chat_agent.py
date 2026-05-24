#!/usr/bin/env python3
"""
Interactive Strands agent for Piper pick-and-place demos.

Architecture:
  - Qwen-VL (qwen-vl-max): describes wrist-camera image at startup
  - Strands Agent + Qwen (qwen-plus): chat loop with function calling
  - Tools: speak (Kokoro), water/ketchup/smore_in_tray (YAML replay)

  export QWEN_API_KEY='sk-...'
  ./scripts/piper_chat_agent.sh
"""

from __future__ import annotations

import argparse
import os
import sys

from hackstorm.agent.strands_agent import (
    DEFAULT_QWEN_AGENT_MODEL,
    bootstrap_system_prompt,
    build_qwen_agent,
    extract_text,
    run_turn,
    verify_tools,
)
from hackstorm.agent.tools import SYSTEM_PROMPT
from hackstorm.vision.qwen_vl import (
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_QWEN_VL_MODEL,
    resolve_api_key,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strands chat agent for Piper demos")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Qwen model for Strands tool calling (default: {DEFAULT_QWEN_AGENT_MODEL})",
    )
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("QWEN_VL_MODEL", DEFAULT_QWEN_VL_MODEL),
        help="Qwen-VL model for startup workspace image (default: qwen-vl-max)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
        help="DashScope OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip wrist-camera capture + VLM scene describe",
    )
    parser.add_argument(
        "--verify-tools",
        action="store_true",
        help="Test Strands tool wiring (direct speak call, no LLM)",
    )
    parser.add_argument(
        "--one-shot",
        metavar="PROMPT",
        help="Run a single agent turn then exit (for testing tool calling)",
    )
    args = parser.parse_args(argv)

    if args.verify_tools:
        agent, _trace = build_qwen_agent(
            api_key=os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "verify-only",
            base_url=args.base_url,
            model_id=args.model or DEFAULT_QWEN_AGENT_MODEL,
            system_prompt=SYSTEM_PROMPT,
        )
        return 0 if verify_tools(agent) else 1

    try:
        api_key = resolve_api_key()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    system_prompt = bootstrap_system_prompt(
        api_key=api_key,
        use_camera=not args.no_camera,
        vlm_model=args.vlm_model,
    )

    agent_model = args.model or os.environ.get("QWEN_AGENT_MODEL", DEFAULT_QWEN_AGENT_MODEL)
    print(
        f"[agent] Strands Agent model={agent_model!r} vlm={args.vlm_model!r} "
        f"tools=speak,water_in_tray,ketchup_in_tray,smore_in_tray",
        flush=True,
    )

    agent, trace = build_qwen_agent(
        api_key=api_key,
        base_url=args.base_url,
        model_id=agent_model,
        system_prompt=system_prompt,
    )

    if args.one_shot:
        print(f"[agent] one-shot: {args.one_shot!r}", flush=True)
        result = run_turn(agent, args.one_shot)
        print(f"\nAgent: {extract_text(result)}\n", flush=True)
        print(f"[agent] stop={result.stop_reason} tool_calls={trace.tool_calls}", flush=True)
        return 0

    print("[agent] starting session...", flush=True)
    bootstrap = (
        "Session started. Greet the user briefly using what you know about the workspace. "
        "Say you can pick water, ketchup, or s'mores into the pink tray when asked. "
        "All three prerecorded demos (water, ketchup, smore) are ready."
    )
    result = run_turn(agent, bootstrap)
    print(f"\nAgent: {extract_text(result)}\n", flush=True)
    print(f"[agent] stop={result.stop_reason} tool_calls={trace.tool_calls}", flush=True)

    print("Type a message (empty line or 'quit' to exit).\n", flush=True)
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[agent] bye", flush=True)
            break
        if not line or line.lower() in {"quit", "exit", "q"}:
            print("[agent] bye", flush=True)
            break
        trace.tool_calls = 0
        result = run_turn(agent, line)
        print(f"\nAgent: {extract_text(result)}\n", flush=True)
        print(f"[agent] stop={result.stop_reason} tool_calls={trace.tool_calls}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
