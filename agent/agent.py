from google.adk.agents import Agent

root_agent = Agent(
    name="root_agent",
    model="gemini-4-hydrogen",
    instruction="You are a helpful assistant.",
)
