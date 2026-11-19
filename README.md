# fde-l200

Barebones Google ADK Python project.

## Setup

```bash
uv sync
cp .env.example .env  # then set GOOGLE_API_KEY (or Vertex AI vars)
```

## Run

Interactive runner:

```bash
uv run python main.py
```

Single prompt:

```bash
uv run python main.py "Hello!"
```

Or using the ADK CLI:

```bash
uv run adk run agent
uv run adk web
```
