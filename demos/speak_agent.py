#!/usr/bin/env python3
"""
Strands agent with ONLY the speak tool — no camera, no arm.

Test Kokoro + Qwen tool calling before running full pick-and-place demos:

  export QWEN_API_KEY='sk-...'
  ./scripts/speak_agent.sh
  ./scripts/speak_agent.sh --one-shot "Use speak to say hello"
"""

from __future__ import annotations

import argparse
import os
import sys

from strands import Agent
from strands.handlers.callback_handler import PrintingCallbackHandler

from hackstorm.agent.strands_agent import (
    DEFAULT_QWEN_AGENT_MODEL,
    build_qwen_agent,
    extract_text,
    run_turn,
    verify_tools,
)
from hackstorm.agent.tools import speak
from hackstorm.vision.qwen_vl import DEFAULT_DASHSCOPE_BASE_URL, resolve_api_key

SPEAK_ONLY_PROMPT = """\
You are a voice assistant for a robot demo bench. You have one tool:

- speak(text): say text aloud with Kokoro TTS on the system speakers.

ALWAYS use speak() when the user wants you to say something aloud.
Keep spoken lines short (1-2 sentences). You may also reply in text after speaking.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strands agent — speak tool only")
    parser.add_argument("--model", default=os.environ.get("QWEN_AGENT_MODEL", DEFAULT_QWEN_AGENT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL))
    parser.add_argument("--verify-tools", action="store_true", help="Direct speak() call, no LLM")
    parser.add_argument("--one-shot", metavar="PROMPT", help="Single agent turn then exit")
    args = parser.parse_args(argv)

    if args.verify_tools:
        agent, _ = build_qwen_agent(
            api_key="verify-only",
            base_url=args.base_url,
            model_id=args.model,
            system_prompt=SPEAK_ONLY_PROMPT,
            callback_handler=PrintingCallbackHandler(verbose_tool_use=True),
        )
        # build_qwen_agent registers all tools — override with speak-only agent
        agent = Agent(
            model=agent.model,
            tools=[speak],
            system_prompt=SPEAK_ONLY_PROMPT,
            callback_handler=PrintingCallbackHandler(verbose_tool_use=True),
            name="Hackstorm Speak Agent",
        )
        return 0 if verify_tools(agent) else 1

    try:
        api_key = resolve_api_key()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"[speak-agent] model={args.model!r} tool=speak only", flush=True)
    base_agent, trace = build_qwen_agent(
        api_key=api_key,
        base_url=args.base_url,
        model_id=args.model,
        system_prompt=SPEAK_ONLY_PROMPT,
    )
    agent = Agent(
        model=base_agent.model,
        tools=[speak],
        system_prompt=SPEAK_ONLY_PROMPT,
        callback_handler=PrintingCallbackHandler(verbose_tool_use=True),
        name="Hackstorm Speak Agent",
    )

    if args.one_shot:
        result = run_turn(agent, args.one_shot)
        print(f"\nAgent: {extract_text(result)}\n", flush=True)
        print(f"[speak-agent] tool_calls={trace.tool_calls}", flush=True)
        return 0

    print("[speak-agent] ready — ask me to say something.\n", flush=True)
    result = run_turn(agent, "Greet the user with speak() and ask what they want you to say.")
    print(f"\nAgent: {extract_text(result)}\n", flush=True)

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[speak-agent] bye", flush=True)
            break
        if not line or line.lower() in {"quit", "exit", "q"}:
            print("[speak-agent] bye", flush=True)
            break
        trace.tool_calls = 0
        result = run_turn(agent, line)
        print(f"\nAgent: {extract_text(result)}\n", flush=True)
        print(f"[speak-agent] tool_calls={trace.tool_calls}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
