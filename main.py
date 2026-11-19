import asyncio
import sys

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from agent.agent import root_agent

APP_NAME = "fde-l200"
USER_ID = "local_user"


async def run_turn(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    text: str,
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
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"{event.author}: {part.text}")


async def main() -> None:
    load_dotenv()

    async with InMemoryRunner(agent=root_agent, app_name=APP_NAME) as runner:
        session = await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
        )

        if len(sys.argv) > 1:
            await run_turn(runner, USER_ID, session.id, " ".join(sys.argv[1:]))
            return

        print("ADK runner ready. Type '/exit' or Ctrl-D to quit.")
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
            await run_turn(runner, USER_ID, session.id, prompt)


if __name__ == "__main__":
    asyncio.run(main())
