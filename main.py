import argparse
import asyncio

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from agent.agent import app as default_app
from agent.agent import root_agent
from agent.compaction import build_app
from agent.config import DEFAULT_PERSONA_ID, MODEL_NAME
from agent.state import PERSONA_ID_STATE_KEY, list_persona_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDE Duckie — Multi-agent onboarding assistant for Google FDE Nooglers."
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_PERSONA_ID,
        choices=list_persona_ids(),
        help=f"Mock FDE Noogler persona to load (default: {DEFAULT_PERSONA_ID}).",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print tool/specialist invocations and compaction events as they occur.",
    )
    parser.add_argument(
        "--no-compaction",
        action="store_true",
        help="Disable ADK context compaction for this run.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional single prompt to run non-interactively.",
    )
    return parser.parse_args()


async def run_turn(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    text: str,
    show_tools: bool = False,
) -> None:
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=text)],
    )
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if show_tools and event.actions and event.actions.compaction:
            comp = event.actions.compaction
            print(
                f"[compaction] summarized history "
                f"({comp.start_timestamp:.0f}..{comp.end_timestamp:.0f})"
            )
        if event.content and event.content.parts:
            for part in event.content.parts:
                if show_tools and part.function_call:
                    print(f"[tool -> {part.function_call.name}] {part.function_call.args}")
                if part.text:
                    print(f"{event.author}: {part.text}")


async def main() -> None:
    load_dotenv()
    args = parse_args()

    active_app = (
        build_app(root_agent=root_agent, name=default_app.name, compaction_enabled=False)
        if args.no_compaction
        else default_app
    )
    compaction_on = bool(active_app.events_compaction_config)

    async with InMemoryRunner(app=active_app) as runner:
        session = await runner.session_service.create_session(
            app_name=active_app.name,
            user_id=args.user,
            state={PERSONA_ID_STATE_KEY: args.user},
        )

        if args.prompt:
            await run_turn(
                runner,
                args.user,
                session.id,
                " ".join(args.prompt),
                show_tools=args.show_tools,
            )
            return

        print(
            f"FDE Duckie ready (model={MODEL_NAME}, persona={args.user}, "
            f"compaction={'on' if compaction_on else 'off'}). "
            "Type '/exit' or Ctrl-D to quit."
        )
        while True:
            try:
                prompt = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            if prompt.lower() == "/exit":
                break
            await run_turn(
                runner,
                args.user,
                session.id,
                prompt,
                show_tools=args.show_tools,
            )


if __name__ == "__main__":
    asyncio.run(main())
