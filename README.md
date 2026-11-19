# FDE Duckie

Multi-agent Google ADK onboarding assistant for new Google Forward Deployed Engineers (**FDE Nooglers**), designed specifically to run reliably on small models (**Gemini Flash-Lite**).

## Architecture (Flash-Lite-First Hub-and-Spoke)

`fde_duckie` (root coordinator) dispatches via ADK `AgentTool` to 3 narrow specialist agents, keeping tool counts tiny (1–3 tools per specialist), preventing transfer loops, and handling multi-domain compound queries in a single turn:

1. **`enablement_agent`** — Checks & updates FDE enablement prerequisites (**L200**, **L300** [6/7 labs], **L400 Bootcamp** [blocked until L300 is complete], **PCA**, plus optional **GAIL**/**CDL**), computes pacing (`on_track` / `at_risk` / `overdue`), and explains `go/fde-enablement-dash` sync delays.
2. **`onboarding_guide_agent`** — Guides Day-1 setup (**Cloudtop**, **Argolis**, `go/fde-github`), 30/60/90-day ramp milestones, pod/buddy/Chat contacts, and deterministic **"What should I do next?"** prioritization.
3. **`duckie_proxy_agent`** — Forwards general, non-FDE-specific Google questions (Workday, gBenefits, corporate card, Piper/CitC, compliance training, `go/wtf`) to a clean `DuckieClient` stub ready to be swapped with real Duckie MCP/RPC.

## Context Compaction

FDE Duckie exports an ADK `App` (`agent.agent.app`) configured with `EventsCompactionConfig` (`agent/compaction.py`) so both `main.py` and `adk run agent` / `adk web` automatically compact long onboarding conversations using a dedicated Flash-Lite `LlmEventSummarizer`:

- **Sliding-window trigger**: Compacts every `5` user turns with `1` turn of overlap (`FDE_DUCKIE_COMPACTION_INTERVAL`, `FDE_DUCKIE_COMPACTION_OVERLAP`).
- **Token-threshold trigger**: Compacts post-invocation whenever prompt tokens reach `8000`, retaining the last `2` raw events (`FDE_DUCKIE_COMPACTION_TOKEN_THRESHOLD`, `FDE_DUCKIE_COMPACTION_RETENTION_SIZE`).
- Disable anytime via `FDE_DUCKIE_COMPACTION_ENABLED=false` or `uv run python main.py --no-compaction`.

## Setup

```bash
uv sync
cp .env.example .env  # set GOOGLE_API_KEY (or Vertex AI vars) and optional FDE_DUCKIE_MODEL
```

## Run

Interactive CLI (defaults to `alex_noogler`, Day 12):

```bash
uv run python main.py
uv run python main.py --user priya_fde --show-tools
```

Single prompt examples:

```bash
uv run python main.py --user priya_fde "Can I register for L400 Bootcamp yet?"
uv run python main.py --user alex_noogler "What should I focus on next?"
uv run python main.py "How do I enroll in health benefits on gBenefits?"
uv run python main.py --user alex_noogler "Check my L200 status and how do I apply for a corporate credit card?"
```

Or using the ADK CLI / Web UI:

```bash
uv run adk run agent
uv run adk web
```

## Mocked Backends (`agent/mocks/`)

All mocked/faked data and stubs are isolated inside `agent/mocks/` and wired through protocol interfaces in `agent/backends.py`:

- `agent/mocks/duckie.py` (`StubDuckieClient`) — stand-in for real Duckie MCP/RPC (`go/duckie`).
- `agent/mocks/profile_store.py` (`MockNooglerProfileStore`) + `agent/mocks/data/noogler_profiles.json` — stand-in for `go/fde-enablement-dash` & Pod Onboarding Tracker.
- `agent/mocks/knowledge_base.py` (`MockFdeKnowledgeBase`) + `agent/mocks/data/fde_kb.json` — stand-in for grounded FDE g3doc / Vertex AI Search.

To wire real integrations later, call `set_duckie_client`, `set_profile_store`, or `set_knowledge_base` in `agent/backends.py`.

### Pre-seeded Mock Personas

- `alex_noogler` (Day 12) — Argolis in progress, `go/fde-github` not started, L200 pending Final Project Assessment, L400 blocked.
- `priya_fde` (Day 42) — Setup & L200 complete, L300 at 4/7 labs (needs 2 more to unlock L400), PCA in progress.
- `jordan_fde` (Day 70) — L200, L300 (6/7), and L400 complete; PCA at risk past Day 60 pacing target.

## Tests

Run offline unit tests (no API key required):

```bash
uv run python -m unittest discover -s tests -v
```
