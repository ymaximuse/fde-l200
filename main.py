import argparse
import asyncio
import uuid

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.genai import types

from agent.agent import create_app
from agent.config import load_config, set_config
from agent.db import configure_storage
from agent.observability import configure_logging, configure_otel
from agent.secrets import hydrate_runtime_secrets
from agent.state import PERSONA_ID_STATE_KEY, list_persona_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDE Duckie — Multi-agent onboarding assistant for Google FDE Nooglers."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a JSON config file (e.g. config.json) or an inline JSON "
            "object configuring model, compaction, logging, otel, database, and default_user."
        ),
    )
    parser.add_argument(
        "--user",
        default=None,
        choices=list_persona_ids(),
        help="Optional override for default_user persona from --config.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID to resume/create (defaults to 'default-<user>').",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a brand-new session ID instead of resuming the default session.",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print tool/specialist invocations and compaction events as they occur.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional single prompt to run non-interactively.",
    )
    return parser.parse_args()


async def run_turn(
    runner: Runner,
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

    cfg = set_config(load_config(args.config))
    user_id = args.user or cfg.default_user

    hydrate_runtime_secrets(cfg)
    configure_logging(cfg)
    tracer_provider = configure_otel(cfg)
    session_service = configure_storage(cfg)
    active_app = create_app(cfg)

    session_id = (
        f"session-{uuid.uuid4().hex[:8]}"
        if args.new_session
        else (args.session or f"default-{user_id}")
    )

    try:
        async with Runner(
            app=active_app,
            session_service=session_service,
        ) as runner:
            session = await session_service.get_session(
                app_name=active_app.name,
                user_id=user_id,
                session_id=session_id,
            )
            resumed = session is not None
            if session is None:
                session = await session_service.create_session(
                    app_name=active_app.name,
                    user_id=user_id,
                    session_id=session_id,
                    state={PERSONA_ID_STATE_KEY: user_id},
                )

            if args.prompt:
                await run_turn(
                    runner,
                    user_id,
                    session.id,
                    " ".join(args.prompt),
                    show_tools=args.show_tools,
                )
                return

            print(
                f"FDE Duckie ready (model={cfg.model.name}, persona={user_id}, "
                f"session={session.id} [{'resumed' if resumed else 'new'}], "
                f"db={cfg.database.backend}, "
                f"compaction={'on' if cfg.compaction.enabled else 'off'}, "
                f"otel={cfg.otel.exporter}). "
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
                    user_id,
                    session.id,
                    prompt,
                    show_tools=args.show_tools,
                )
    finally:
        close_fn = getattr(session_service, "close", None)
        if callable(close_fn):
            await close_fn()
        if tracer_provider is not None:
            tracer_provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
