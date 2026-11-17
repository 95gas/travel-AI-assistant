# WeRoad Agentic Travel Assistant

An end-to-end, safety-guarded agentic travel advisor built with **Python**, **Pydantic v2**, **PydanticAI**, and the **Model Context Protocol (MCP)**. It helps travelers discover **WeRoad.it** group itineraries, inspects live scheduled departure dates (*turni*), verifies availability statuses and age cohorts, and pairs matching tour dates with **Skyscanner** round-trip flight prices, durations, stops, and direct booking links.

---

## Key Features

- **Two-Stage Agentic Orchestration (`src/ai_assistant/core/orchestrator.py`)**:
  - **Stage 1 — Intent Extraction**: Parses natural-language travel queries into a structured `TravelSearchIntent` schema (destination, normalized Italian URL slug keywords, departure airport & 3-letter IATA code, destination IATA code, date bounds, trip duration, budget, travel style, and traveler age).
  - **Departure Airport Gate**: Automatically prompts the user for their departure airport if omitted (`needs_departure_airport = True`) so flight searches are always anchored to a real origin hub.
  - **Stage 2 — Tool-Equipped Search & Synthesis**: Uses MCP tools to search the WeRoad catalog, inspect live departure dates, look up matching round-trip flights on Skyscanner, and synthesize a structured `TravelProposalResponse` in the user's language.
- **Model Context Protocol (MCP) Tool Server (`src/ai_assistant/mcp_server/weroad_mcp.py`)**:
  - Exposes `search_weroad_tours`, `get_tour_departures`, and `search_flights` over `stdio` using `FastMCP`.
- **Live WeRoad Scraper & Sitemap Indexer (`src/ai_assistant/tools/weroad_parser.py`)**:
  - Discovers active `/viaggi/<slug>` itineraries from WeRoad XML sitemaps with a 6-hour local JSON cache (`.weroad_catalog_cache.json`) and `/destinazioni/<keyword>` fallback discovery.
  - Parses embedded Next.js `__NEXT_DATA__`, JSON-LD blocks, and DOM elements to extract departure dates, EUR pricing, active promotions, age cohorts (`18-24`, `25-35`, `35-49`, `Tutti`), and statuses (`Confermato`, `Quasi pieno`, `Disponibile`, `Sold out`).
- **Skyscanner Round-Trip Flight Integration (`src/ai_assistant/tools/flight_search.py`)**:
  - Resolves cities, countries, and airports to 3-letter IATA codes, builds pre-filled Skyscanner round-trip URLs (`skyscanner.it`), queries live Skyscanner itineraries via RapidAPI when `RAPIDAPI_KEY` is configured, and falls back to route-calibrated EUR cost and duration benchmarks.
- **Input & Output Safety Guardrails (`src/ai_assistant/core/guardrail.py`)**:
  - Moderates every incoming user query and outgoing assistant response using **Meta Llama Guard** (`openrouter:meta-llama/llama-guard-4-12b`) with configurable fail-closed/fail-open policies across safety categories `S1`–`S13`.
- **Multiple User Interfaces (`main.py`, `server.py`, `streamlit_app.py`)**:
  - Run via an interactive or single-query **CLI**, a **FastAPI** REST API (`/api/v1/chat`, `/health`, `/metrics`), or a **Streamlit** web chat UI with tour cards, departure tables, flight comparisons, and trace inspection links.
- **Full OpenTelemetry Observability (`src/ai_assistant/observability/observability.py`)**:
  - Exports distributed traces to **Grafana Tempo**, correlated structured JSON logs (`trace_id`, `span_id`) to **Grafana Loki**, and token/latency/guardrail metrics to **Prometheus** and **Grafana**.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                              Client Entrypoints                                   |
|        CLI (main.py)   |   Streamlit UI (streamlit_app.py)   |   REST Clients     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v (HTTP /api/v1/chat or direct call)
+-----------------------------------------------------------------------------------+
|                        Step 0: Input Safety Guardrail                             |
|                LlamaGuardModerator.check_input() [Llama Guard]                    |
+-----------------------------------------+-----------------------------------------+
                                          | (Safe)
                                          v
+-----------------------------------------------------------------------------------+
|                    Stage 1: Natural-Language Intent Parser                        |
|           GenericAgent -> TravelSearchIntent (Destination, Dates, IATA)           |
+-----------------------------------------+-----------------------------------------+
                                          |
                     +--------------------+--------------------+
                     | (Departure Airport Missing)             | (Airport Present)
                     v                                         v
+-----------------------------------------+   +-------------------------------------+
|   Prompt User for Departure Airport     |   | Stage 2: Tool-Equipped Advisor      |
|   (needs_departure_airport = True)      |   | GenericAgent -> TravelProposalResp. |
+-----------------------------------------+   +------------------+------------------+
                                                                 | (MCP stdio)
                                                                 v
                                              +-------------------------------------+
                                              |   WeRoad MCP Server (weroad_mcp.py) |
                                              |  - search_weroad_tours()            |
                                              |  - get_tour_departures()            |
                                              |  - search_flights()                 |
                                              +---------+-----------------+---------+
                                                        |                 |
                                                        v                 v
                                              +-----------------+ +-----------------+
                                              |  WeRoadClient   | |FlightSearchClt. |
                                              |  (weroad.it)    | |(Skyscanner.it)  |
                                              +-----------------+ +-----------------+
                                                                 |
                                                                 v
+-----------------------------------------------------------------------------------+
|                        Step 3: Output Safety Guardrail                            |
|               LlamaGuardModerator.check_output() [Llama Guard]                    |
+-----------------------------------------+-----------------------------------------+
                                          | (Safe)
                                          v
+-----------------------------------------------------------------------------------+
|            TravelProposalResponse (Tours, Departures, Flights, Telemetry)         |
+-----------------------------------------------------------------------------------+
```

---

## Project Structure

```text
travel_assistant/
├── main.py                                      # CLI & interactive multi-turn terminal entrypoint
├── server.py                                    # FastAPI REST backend (/api/v1/chat, /health, /metrics)
├── streamlit_app.py                             # Streamlit interactive web chat UI
├── pyproject.toml                               # Project metadata & dependencies (uv / pip)
├── uv.lock                                      # Locked dependency graph
├── docker-compose.yaml                          # Docker Compose stack for Tempo, Loki, Prometheus & Grafana
├── config/
│   ├── config.yaml                              # Centralized pipeline, Llama Guard, scraper, flight, server & OTel config
│   └── mcp_config.json                          # MCP stdio server command definition ("weroad")
└── src/
    └── ai_assistant/
        ├── core/
        │   ├── agent.py                         # GenericAgent wrapper around PydanticAI Agent, MCP & OTel
        │   ├── config_management.py             # Singleton ConfigManager loading .env and YAML configs
        │   ├── guardrail.py                     # LlamaGuardModerator for input/output safety checks (S1–S13)
        │   └── orchestrator.py                  # Two-stage WeRoadTravelAssistant orchestrator with I/O guardrails
        ├── mcp_server/
        │   └── weroad_mcp.py                    # FastMCP stdio server exposing tour & flight search tools
        ├── observability/
        │   ├── observability.py                 # OpenTelemetry Tempo traces, Loki OTLP logs & Prometheus metrics
        │   ├── tempo.yaml                       # Grafana Tempo OTLP receiver & span-metrics generator config
        │   ├── loki.yaml                        # Grafana Loki native OTLP log ingestion config
        │   ├── prometheus.yaml                  # Prometheus scrape config for FastAPI /metrics & Tempo
        │   ├── grafana-datasources.yaml         # Auto-provisioned Prometheus, Tempo & Loki datasources
        │   ├── grafana-dashboards.yaml          # Grafana dashboard provisioning config
        │   └── grafana-dashboard.yaml           # Pre-built Grafana dashboard definition
        ├── schemas/
        │   └── schemas.py                       # Pydantic v2 models for configs, intents, tours, flights & telemetry
        └── tools/
            ├── weroad_parser.py                 # Async WeRoad.it XML sitemap indexer & HTML/JSON departure parser
            └── flight_search.py                 # IATA resolver, Skyscanner deep-link builder & RapidAPI client
```

---

## Prerequisites & Installation

### 1. System Requirements

- **Python 3.11+**
- **uv** (recommended) or **pip**
- **Docker & Docker Compose** (optional, required only for the local Grafana/Tempo/Loki/Prometheus observability stack)
- Internet access to query LLM providers (OpenRouter / OpenAI), `weroad.it`, and `skyscanner.it`

### 2. Install Dependencies

#### Option A: Using `uv` (Recommended)

```bash
cd travel_assistant
uv sync
```

#### Option B: Using `pip` and a Virtual Environment

```bash
cd travel_assistant
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install pydantic>=2.9.0 pydantic-ai>=0.1.0 mcp>=1.2.0 httpx>=0.27.2 beautifulsoup4>=4.12.3 \
  pyyaml python-dotenv fastapi uvicorn streamlit \
  opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
  opentelemetry-exporter-prometheus prometheus-client
```

---

## Configuration

### 1. Environment Variables (`.env`)

Create or update `.env` in the `travel_assistant/` root directory:

```env
# Required when using default "openrouter:..." models in config/config.yaml
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional: Required only if switching model_name to direct OpenAI models
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Enables live Skyscanner quotes via RapidAPI (Sky-Scrapper API).
# If omitted, the assistant automatically uses route-calibrated EUR price/duration
# estimates alongside exact pre-filled Skyscanner search links.
RAPIDAPI_KEY=your_rapidapi_key_here

# Optional: Override FastAPI backend URL consumed by Streamlit (default: http://localhost:8000)
WEROAD_API_URL=http://localhost:8000
```

### 2. Application Configuration (`config/config.yaml`)

`ConfigManager` loads `config/config.yaml` at startup and validates it into `WeRoadAssistantConfig`:

- **`pipeline`**:
  - `llm_default`, `llm_intent`, `llm_proposal`: Configure `model_name` (default `openrouter:openai/gpt-4o-mini`), `temperature`, and `max_tokens` per stage.
  - `intent_request_limit` (`2`) and `proposal_request_limit` (`8`): Cap LLM/tool round-trips.
  - `require_departure_airport` (`true`): Prompts the user for their origin airport when omitted.
- **`llama_guard`**:
  - `enabled`, `check_input`, `check_output`: Toggle input and output safety moderation.
  - `model_name`: Defaults to `openrouter:meta-llama/llama-guard-4-12b`.
  - `fail_open`: Defaults to `false` (blocks requests if the moderation service fails).
  - `category_map`: Maps safety codes `S1`–`S13` to human-readable descriptions.
- **`scraper`**:
  - `base_url` (`https://www.weroad.it`), `sitemap_url` (`https://www.weroad.it/sitemap.xml`), `cache_path` (`.weroad_catalog_cache.json`), `cache_ttl_seconds` (`21600` = 6 hours), `timeout_seconds` (`20.0`), and `max_concurrent_requests` (`5`).
- **`flight_search`**:
  - `enabled`, `provider` (`"skyscanner"`), `currency` (`"EUR"`), `market` (`"IT"`), `locale` (`"it-IT"`), `skyscanner_base_url` (`https://www.skyscanner.it/trasporti/voli`), and `rapidapi_host`.
- **`server`**:
  - `host` (`"0.0.0.0"`), `port` (`8000`), `api_base_url`, and `cors_origins` (allows Streamlit on port `8501`).
- **`observability`**:
  - `enabled`, `service_name`, `environment`, `otlp_traces_endpoint`, `otlp_logs_endpoint`, `otlp_metrics_endpoint`, `grafana_base_url`, `capture_prompts`, `json_logs`, and `log_level`.
- **`schemas`**:
  - `mcp_config_path`: Path to the MCP server JSON definition (set to `config/mcp_config.json` when running from the project root).
- **`prompts`**:
  - Customizable `intent_system_prompt`, `proposal_system_prompt`, `stage2_user_prompt_template`, and `missing_departure_airport_message`.

### 3. MCP Server Configuration (`config/mcp_config.json`)

`GenericAgent` spawns the WeRoad MCP server over `stdio` using `config/mcp_config.json`. When running from the project root with the `src/` layout, ensure `PYTHONPATH=src` is exported (or configured in `env`) and `mcp_config.json` points to `ai_assistant.mcp_server.weroad_mcp`:

```json
{
  "mcpServers": {
    "weroad": {
      "command": "python",
      "args": ["-m", "ai_assistant.mcp_server.weroad_mcp"],
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

---

## How to Run

### 1. FastAPI Backend Server (`server.py`)

Start the REST API server on port `8000`:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:
- `GET /` — Service metadata and endpoint directory
- `GET /health` — Health check, active model, and guardrail status
- `GET /metrics` — Prometheus metrics exposition (`weroad_llm_tokens`, `weroad_llm_requests`, `weroad_llm_request_duration_seconds`, `weroad_guardrail_checks`)
- `POST /api/v1/chat` — Main guarded travel assistant endpoint
- `GET /docs` — Interactive Swagger UI documentation

Example API request:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to visit Japan in October for 10 days. I am 29.",
    "departure_airport": "MXP",
    "session_id": "demo-session-1"
  }'
```

### 2. Streamlit Web Interface (`streamlit_app.py`)

With the FastAPI server running, launch the Streamlit chat UI in a second terminal:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser to:
- Check FastAPI backend health in the sidebar
- Select a default departure airport (`MXP`, `LIN`, `BGY`, `FCO`, `VCE`, `BLQ`, `NAP`, etc.) or enter a custom airport/IATA code
- Run sample prompts or chat naturally in English or Italian
- Browse expandable WeRoad tour cards, live departure date tables, Skyscanner round-trip flight links, and per-stage token telemetry

### 3. CLI Mode (`main.py`)

#### Interactive Multi-Turn Chat

```bash
PYTHONPATH=src python main.py
```

#### Single-Query Execution

```bash
PYTHONPATH=src python main.py --query "Japan in October for 10 days leaving from Milan Malpensa"
```

In Italian:

```bash
PYTHONPATH=src python main.py -q "Cerco un viaggio WeRoad in Islanda ad agosto partendo da Roma Fiumicino"
```

#### CLI Flags

| Flag | Description |
| :--- | :--- |
| `-q`, `--query "<text>"` | Run a single travel query and exit. |
| `-i`, `--interactive` | Force interactive multi-turn CLI chat mode. |
| `-a`, `--airport <code>` | Pre-set a default departure airport or IATA code (e.g., `-a MXP`, `-a "Rome Fiumicino"`). |
| `-c`, `--config <path>` | Custom path to YAML configuration file (defaults to `config/config.yaml`). |
| `--mcp-config <path>` | Custom path to `mcp_config.json` (e.g., `config/mcp_config.json`). |
| `--model <model_id>` | Override the PydanticAI model for both stages (e.g., `openrouter:openai/gpt-4o-mini`). |
| `--json` | Output the structured `TravelProposalResponse` as formatted JSON. |
| `--log-level <level>` | Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

Examples:

```bash
# Supply default departure airport via CLI flag
PYTHONPATH=src python main.py -q "Morocco in November for 8 days" -a BLQ --mcp-config config/mcp_config.json

# Export raw JSON output
PYTHONPATH=src python main.py -q "Peru in September from MXP" --mcp-config config/mcp_config.json --json
```

### 4. Running the MCP Server Standalone

To inspect or test the MCP tool server directly (e.g., with MCP Inspector):

```bash
PYTHONPATH=src python -m ai_assistant.mcp_server.weroad_mcp
```

*Note: `weroad_mcp.py` routes all application logs strictly to `stderr` so the `stdout` JSON-RPC stream remains clean for MCP clients.*

---

## Observability Stack (Grafana, Tempo, Loki & Prometheus)

The repository includes full OpenTelemetry instrumentation (`src/ai_assistant/observability/observability.py`) and a Docker Compose stack (`docker-compose.yaml`) for monitoring LLM token usage, request latency, distributed traces, correlated JSON logs, and Llama Guard safety checks:

1. **Start Tempo, Loki, Prometheus, and Grafana:**
   ```bash
   docker compose up -d
   ```
   *(Note: If running from the root directory, ensure the volume paths in `docker-compose.yaml` point to `./src/ai_assistant/observability/...`.)*

2. **Access the Monitoring UIs:**
   - **Grafana**: `http://localhost:3000` (default credentials: `admin` / `admin`)
   - **Prometheus**: `http://localhost:9090`
   - **Tempo**: `http://localhost:3200` (OTLP gRPC `4317`, OTLP HTTP `4318`)
   - **Loki**: `http://localhost:3100` (OTLP HTTP `/otlp/v1/logs`)

3. **Trace & Log Correlation:**
   - Every request generates a root trace span (`HTTP POST /api/v1/chat` -> `weroad.handle_query` -> `agent.intent_extraction` -> `agent.proposal_synthesis`) and returns an `X-Trace-Id` response header.
   - Python logs automatically inject `trace_id` and `span_id`, enabling one-click navigation between **Grafana Loki** logs and **Grafana Tempo** waterfall traces, as well as direct trace links inside the Streamlit UI.

---

## Troubleshooting

- **Missing `OPENROUTER_API_KEY`**: Verify `.env` exists in `travel_assistant/` and contains a valid `OPENROUTER_API_KEY`, or pass `--model` with another configured provider.
- **MCP Config Not Found (`mcp_config.json`)**: `ConfigManager` defaults `schemas.mcp_config_path` to `"mcp_config.json"` in the working directory, whereas the file resides at `config/mcp_config.json`. Either pass `--mcp-config config/mcp_config.json` or update `schemas.mcp_config_path` in `config/config.yaml`.
- **Module Import Errors (`No module named 'ai_assistant'` or `'agent'`)**: Run commands from `travel_assistant/` with `PYTHONPATH=src` (or `PYTHONPATH=src:src/ai_assistant/core`) so both package imports (`ai_assistant.*`) and internal module imports resolve cleanly.
- **Plain URLs in `config/config.yaml`**: Ensure `scraper.base_url`, `scraper.sitemap_url`, `flight_search.skyscanner_base_url`, and `server.cors_origins` contain plain URL strings (e.g., `"https://www.weroad.it"`) rather than Markdown-formatted links.
- **Llama Guard Blocking Requests (`MODERATION_UNAVAILABLE`)**: By default, `llama_guard.fail_open` is `false`. If OpenRouter is unreachable during local testing, set `llama_guard.enabled: false` or `llama_guard.fail_open: true` in `config/config.yaml`.
- **Stale WeRoad Tour Catalog**: Delete `.weroad_catalog_cache.json` to force `WeRoadClient` to re-index active `/viaggi/<slug>` itineraries from `weroad.it/sitemap.xml`.