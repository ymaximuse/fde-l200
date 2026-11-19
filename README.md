# FDE Duckie

Multi-agent Google ADK onboarding assistant for new Google Forward Deployed Engineers (**FDE Nooglers**), designed specifically to run reliably on small models (**Gemini Flash-Lite**).

## Architecture (Sequential Complexity Routing + Hub-and-Spoke)

`fde_duckie` is an ADK `SequentialAgent` with two steps:
1. **`complexity_analyzer_agent`** (always `gemini-2.5-flash-lite`) — Classifies each incoming turn as `simple` or `complex` (`ComplexityAssessment`) and stores the result in `session.state["request_complexity"]`.
2. **`fde_duckie_coordinator`** — Dispatches via ADK `AgentTool` to 3 narrow specialist agents, keeping tool counts tiny (1–3 tools per specialist), preventing transfer loops, and handling multi-domain compound queries in a single turn:

1. **`enablement_agent`** — Checks & updates FDE enablement prerequisites (**L200**, **L300** [6/7 labs], **L400 Bootcamp** [blocked until L300 is complete], **PCA**, plus optional **GAIL**/**CDL**), computes pacing (`on_track` / `at_risk` / `overdue`), and explains `go/fde-enablement-dash` sync delays.
2. **`onboarding_guide_agent`** — Guides Day-1 setup (**Cloudtop**, **Argolis**, `go/fde-github`), 30/60/90-day ramp milestones, pod/buddy/Chat contacts, and deterministic **"What should I do next?"** prioritization.
3. **`duckie_proxy_agent`** — Forwards general, non-FDE-specific Google questions (Workday, gBenefits, corporate card, Piper/CitC, compliance training, `go/wtf`) to a clean `DuckieClient` stub ready to be swapped with real Duckie MCP/RPC.

## Configuration (`--config` JSON)

All runtime settings (model, compaction, structured logging, OpenTelemetry, and default persona) are consolidated into a single validated Pydantic JSON config (`agent/config.py`) passed via `--config`. `.env` is reserved strictly for credentials/secrets.

```bash
uv sync
cp .env.example .env                # set GOOGLE_API_KEY (or Vertex AI credentials)
uv run python scripts/setup_db.py   # initialize & seed local SQLite DB (data/fde_duckie.db)
```

### Local SQLite Persistence (`agent/infra/db.py` & `scripts/setup_db.py`)

By default (`database.backend="sqlite"`, `database.url="sqlite:///data/fde_duckie.db"` in `config.json`), FDE Duckie persists both **ADK session state + conversation history** (via ADK's native `SqliteSessionService`) and **Noogler profile updates** (`noogler_profiles` table) in a local SQLite database:

```bash
# Seed or reset the local dev database
uv run python scripts/setup_db.py
uv run python scripts/setup_db.py --reset

# Sessions automatically resume across runs (or pass --new-session / --session <id>)
uv run python main.py --user priya_fde "Mark my L300 labs as 6 completed"
uv run python main.py --user priya_fde "Am I eligible for L400 now?"

# Use the same SQLite database in ADK Web UI
uv run adk web --session_service_uri sqlite:///data/fde_duckie.db
```

Set `"database": {"backend": "memory"}` in `--config` for ephemeral in-memory runs, or `"backend": "database"` with a SQLAlchemy URI (`google-adk[db]`) for Postgres/MySQL.

### HashiCorp Vault Secrets (`agent/infra/secrets.py` & `scripts/manage_secrets.py`)

By default (`secrets.backend="vault"` in `config.json`), FDE Duckie fetches runtime secrets (`GOOGLE_API_KEY`, `OTEL_EXPORTER_OTLP_HEADERS`) from **HashiCorp Vault KV v2** (`VAULT_ADDR`, `VAULT_TOKEN` / `~/.vault-token`), with seamless fallback to `.env` if Vault is not running:

```bash
# 1. Start local HashiCorp Vault in dev mode
vault server -dev -dev-root-token-id=dev-root-token
export VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=dev-root-token

# 2. Store secrets in Vault (or import from .env)
uv run python scripts/manage_secrets.py set GOOGLE_API_KEY
uv run python scripts/manage_secrets.py import-env
uv run python scripts/manage_secrets.py list
```

### Terraform (`infra/terraform/`)

A single local Terraform config (`hashicorp/vault` + built-in `terraform_data`) reads `config.json`, seeds runtime secrets into HashiCorp Vault KV v2, and initializes/seeds `data/fde_duckie.db` via `scripts/setup_db.py`. See [infra/terraform/README.md](infra/terraform/README.md).

Pass `--config` as either a JSON file path or an inline JSON override:

```bash
# Load from a JSON config file (defaults to ./config.json)
uv run python main.py --config config.json

# Or pass inline JSON config directly
uv run python main.py --config '{"default_user": "priya_fde", "otel": {"exporter": "console"}}' "Can I register for L400 Bootcamp yet?"
```

For `adk run agent` and `adk web`, `config.json` in the project root is loaded automatically, or set `FDE_DUCKIE_CONFIG=path/to/config.json` in `.env`.

### Context Compaction, Model Routing, Security & Observability (`agent/plugins/`)

- **Strategic Model Routing (`agent/plugins/routing.py`)**: Enabled by default (`model_routing.enabled: true` in `config.json`) via `ModelRoutingPlugin`. Reads `session.state["request_complexity"]` (populated by step-1 `complexity_analyzer_agent`) in `before_model_callback` and routes `simple` requests to `gemini-2.5-flash-lite` (`model.name`) and `complex` requests to `gemini-2.5-flash` (`model_routing.capable_model`), while keeping `complexity_analyzer_agent` pinned to Flash-Lite. Emits structured `model.routed` logs and OTel span attributes.
- **Context Compaction (`agent/plugins/compaction.py`)**: Enabled by default via ADK `App` (`agent/agent.py`) + `EventsCompactionConfig` using a Flash-Lite `LlmEventSummarizer` with both sliding-window (`interval: 5`, `overlap_size: 1`) and token-threshold (`token_threshold: 8000`, `event_retention_size: 2`) triggers.
- **PII Redaction & Scrubbing (`agent/plugins/pii.py`)**: Enabled by default (`pii.enabled: true` in `config.json`) via `PiiRedactionPlugin` and `JsonFormatter` log scrubbing. Uses lightweight regexes to redact `EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`, and `API_KEY` across user inputs, tool results, model responses/events, and structured JSON logs.
- **Gemini Model Armor (`agent/plugins/security.py`)**: `ModelArmorPlugin` (opt-in for Vertex AI) attaches Gemini inline `types.ModelArmorConfig` (`prompt_template_name` / `response_template_name`) to every LLM request and returns a safe fallback on `SAFETY`/`PROHIBITED_CONTENT`/`BLOCKLIST`/`SPII` blocks. Enable via `--config`:
  ```bash
  uv run python main.py --config '{"security": {"model_armor": {"enabled": true, "prompt_template_name": "projects/MY_PROJECT/locations/us-central1/templates/MY_TEMPLATE"}}}'
  ```
- **Human-in-the-Loop Tool Confirmation (`agent/agent.py` & `main.py`)**: `switch_noogler_persona` is wrapped in ADK's `FunctionTool(..., require_confirmation=True)` with `ResumabilityConfig(is_resumable=True)` on `App`. Before switching the active Noogler session identity, ADK pauses the invocation and requests explicit human approval (`adk_request_confirmation`), handled interactively in `main.py` (`[y/N]`) as well as natively in `adk web` / `adk run`.
- **Structured Logging & OpenTelemetry (`agent/plugins/observability.py`)**: `ObservabilityPlugin` is attached to `app.plugins`, emitting single-line trace-correlated JSON logs (`trace_id`/`span_id`) for agent/tool events while ADK handles OTel spans natively. Configure `otel.exporter` as `none`, `console`, or `otlp`.

## Run

Interactive CLI:

```bash
uv run python main.py
uv run python main.py --config config.json --show-tools
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

All mocked/faked data and stubs are isolated inside `agent/mocks/` and wired through protocol interfaces in `agent/infra/backends.py`:

- `agent/mocks/duckie.py` (`StubDuckieClient`) — stand-in for real Duckie MCP/RPC (`go/duckie`).
- `agent/mocks/profile_store.py` (`MockNooglerProfileStore`) + `agent/mocks/data/noogler_profiles.json` — stand-in for `go/fde-enablement-dash` & Pod Onboarding Tracker.
- `agent/mocks/knowledge_base.py` (`MockFdeKnowledgeBase`) + `agent/mocks/data/fde_kb.json` — stand-in for grounded FDE g3doc / Vertex AI Search.

To wire real integrations later, call `set_duckie_client`, `set_profile_store`, or `set_knowledge_base` in `agent/infra/backends.py`.

### Pre-seeded Mock Personas

- `alex_noogler` (Day 12) — Argolis in progress, `go/fde-github` not started, L200 pending Final Project Assessment, L400 blocked.
- `priya_fde` (Day 42) — Setup & L200 complete, L300 at 4/7 labs (needs 2 more to unlock L400), PCA in progress.
- `jordan_fde` (Day 70) — L200, L300 (6/7), and L400 complete; PCA at risk past Day 60 pacing target.

## Tests & Offline Golden Evaluation

Run offline unit tests (no API key required):

```bash
uv run python -m unittest discover -s tests -v
```

### Automated Golden Dataset Offline Eval (`evals/` & `scripts/run_eval.py`)

A 12-case golden dataset in native Google ADK `EvalSet` format (`evals/golden_dataset.evalset.json`) + `EvalConfig` (`evals/test_config.json`) evaluates tool routing (`tool_trajectory_avg_score`) and response quality (`response_match_score`) across all 3 personas and specialists:

```bash
# Run via scripts/run_eval.py (loads config.json + Vault/.env secrets)
uv run python scripts/run_eval.py --config config.json

# Or directly via Google ADK's CLI
uv run adk eval agent evals/golden_dataset.evalset.json --config_file_path evals/test_config.json
```
