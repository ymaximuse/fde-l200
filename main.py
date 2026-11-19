import argparse
import asyncio
from collections.abc import Callable
from typing import Any
import uuid

from dotenv import load_dotenv
from google.adk.flows.llm_flows.functions import (
    REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
)
from google.adk.runners import Runner
from google.genai import types

from agent.agent import create_app
from agent.config import load_config, set_config
from agent.infra.db import configure_storage
from agent.infra.secrets import hydrate_runtime_secrets
from agent.plugins.observability import configure_logging, configure_otel
from agent.state import PERSONA_ID_STATE_KEY, list_persona_ids
from agent.sub_agents.complexity_analyzer_agent import COMPLEXITY_ANALYZER_NAME


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


def _prompt_tool_confirmation(
    tool_name: str,
    tool_args: dict[str, Any],
    hint: str | None = None,
) -> bool:
    """Prompt the user on stdin to approve or reject a confirmation-gated tool call."""
    prompt = (
        f"[confirmation required] Allow tool `{tool_name}` "
        f"with args {tool_args}? [y/N]: "
    )
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


async def run_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    text: str,
    show_tools: bool = False,
    confirm_fn: Callable[[str, dict[str, Any], str | None], bool] | None = None,
) -> None:
    confirm = confirm_fn or _prompt_tool_confirmation
    next_message: types.Content | None = types.Content(
        role="user",
        parts=[types.Part.from_text(text=text)],
    )
    resume_invocation_id: str | None = None

    while next_message is not None:
        pending_confirmations: list[types.FunctionCall] = []
        paused_invocation_id: str | None = None

        run_kwargs: dict[str, Any] = {
            "user_id": user_id,
            "session_id": session_id,
            "new_message": next_message,
        }
        if resume_invocation_id is not None:
            run_kwargs["invocation_id"] = resume_invocation_id

        async for event in runner.run_async(**run_kwargs):
            if show_tools and event.actions and event.actions.compaction:
                comp = event.actions.compaction
                print(
                    f"[compaction] summarized history "
                    f"({comp.start_timestamp:.0f}..{comp.end_timestamp:.0f})"
                )
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        if part.function_call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
                            pending_confirmations.append(part.function_call)
                            paused_invocation_id = event.invocation_id
                        elif show_tools:
                            print(
                                f"[tool -> {part.function_call.name}] "
                                f"{part.function_call.args}"
                            )
                    if part.text:
                        if event.author == COMPLEXITY_ANALYZER_NAME:
                            if show_tools:
                                print(f"[complexity] {part.text}")
                        else:
                            print(f"{event.author}: {part.text}")

        if not pending_confirmations:
            break

        response_parts: list[types.Part] = []
        for conf_call in pending_confirmations:
            call_args = conf_call.args or {}
            original_fc = call_args.get("originalFunctionCall") or {}
            orig_name = str(original_fc.get("name") or "unknown_tool")
            orig_args = dict(original_fc.get("args") or {})
            tool_conf = call_args.get("toolConfirmation") or {}
            hint = tool_conf.get("hint")

            approved = confirm(orig_name, orig_args, hint)
            response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=conf_call.id,
                        name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                        response={"confirmed": approved},
                    )
                )
            )

        next_message = types.Content(role="user", parts=response_parts)
        resume_invocation_id = paused_invocation_id


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

            routing_status = (
                f"on->{cfg.model_routing.capable_model}"
                if cfg.model_routing.enabled
                else "off"
            )
            if not cfg.compaction.enabled:
                compaction_status = "off"
            elif cfg.compaction.async_background:
                compaction_status = f"async(first={cfg.compaction.first_n_messages})"
            else:
                compaction_status = "sync"
            print(
                f"FDE Duckie ready (model={cfg.model.name}, routing={routing_status}, "
                f"persona={user_id}, "
                f"session={session.id} [{'resumed' if resumed else 'new'}], "
                f"db={cfg.database.backend}, "
                f"compaction={compaction_status}, "
                f"pii={'on' if cfg.pii.enabled else 'off'}, "
                f"model_armor={'on' if cfg.security.model_armor.enabled else 'off'}, "
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
