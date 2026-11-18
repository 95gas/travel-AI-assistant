"""
Domain Schemas & Configuration Models (`schemas.py`)
===========================================================
Defines all Pydantic v2 data contracts shared across `agent.py`, the Llama Guard
moderator, `config_management.py`, the WeRoad parser, the MCP tool server,
the two-stage orchestrator, the FastAPI backend, and the web frontend.
"""

from __future__ import annotations

import time
import uuid
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, HttpUrl

ReasoningEffort = Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]


# ---------------------------------------------------------------------------
# Configuration Schemas
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """LLM configuration schema consumed by `GenericAgent` in `agent.py`."""

    model_name: str = Field(
        default="openrouter:openrouter/free",
        description=(
            "PydanticAI model identifier (e.g. 'openrouter:openrouter/free', "
            "'openrouter:nvidia/nemotron-3-nano-30b-a3b:free', 'ollama:qwen2.5:3b')."
        ),
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    provider: Optional[str] = None
    base_url: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    load_in_4bit: bool = False
    reasoning_enabled: Optional[bool] = Field(
        default=None,
        description="Enable or disable reasoning/thinking tokens (e.g. on OpenRouter models).",
    )
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="OpenRouter/OpenAI-style reasoning effort level ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max').",
    )
    reasoning_max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional explicit reasoning token budget (cannot be combined with reasoning_effort).",
    )
    reasoning_exclude: Optional[bool] = Field(
        default=None,
        description="If True, the model reasons internally without returning reasoning text in the response.",
    )


class LlamaGuardConfig(BaseModel):
    """Configuration for input/output safety moderation via Llama Guard."""

    enabled: bool = Field(default=True)
    model_name: str = Field(default="openrouter:openrouter/free")
    provider: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    load_in_4bit: bool = False
    reasoning_enabled: Optional[bool] = Field(
        default=False,
        description="Disabled by default so safety classification returns fast 'safe'/'unsafe' verdicts.",
    )
    reasoning_effort: Optional[ReasoningEffort] = Field(default=None)
    check_input: bool = Field(default=True)
    check_output: bool = Field(default=True)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum tokens for moderation response (set >= 1024 when routing through models with mandatory reasoning tokens).",
    )
    fail_open: bool = Field(
        default=False,
        description="If True, allow traffic when moderation fails with a runtime error; if False, block on error.",
    )
    blocked_input_message: str = Field(
        default="I'm sorry, I cannot process this request because it violates our content safety guidelines."
    )
    blocked_output_message: str = Field(
        default="I'm sorry, the generated response could not be displayed due to content safety policies."
    )
    system_prompt: str = Field(
        default=(
            "You are a content safety classifier. Determine whether the user input or model output "
            "is safe or unsafe according to standard AI safety policies.\n\n"
            "Output format:\n"
            "If safe, respond ONLY with 'safe'.\n"
            "If unsafe, respond with 'unsafe' on the first line, followed by the specific category code "
            "(e.g. S1, S2, S13) on the second line."
        )
    )
    category_map: Dict[str, str] = Field(
        default_factory=lambda: {
            "S1": "Violent Crimes",
            "S2": "Non-Violent Crimes",
            "S3": "Sex Crimes",
            "S4": "Child Sexual Exploitation and Abuse",
            "S5": "Defamation",
            "S6": "Specialized Advice (Medical/Legal/Financial)",
            "S7": "Privacy & Personally Identifiable Information",
            "S8": "Intellectual Property Violation",
            "S9": "Indiscriminate Weapons (CBRN)",
            "S10": "Hate Speech",
            "S11": "Suicide & Self-Harm",
            "S12": "Sexual Content",
            "S13": "Cyberattacks & Malware",
        }
    )


class SafetyCheckResult(BaseModel):
    """Outcome of a Llama Guard input or output moderation check."""

    is_safe: bool
    violating_categories: List[str] = Field(default_factory=list)
    raw_response: str = ""
    checked_target: str = Field(default="input", description="'input' or 'output'")


class WeRoadScraperConfig(BaseModel):
    """Configuration for `WeRoadClient` HTTP fetching and sitemap caching."""

    base_url: str = Field(default="https://www.weroad.it")
    sitemap_url: str = Field(default="https://www.weroad.it/sitemap.xml")
    cache_path: str = Field(default=".weroad_catalog_cache.json")
    cache_ttl_seconds: int = Field(default=6 * 3600, gt=0)
    timeout_seconds: float = Field(default=20.0, gt=0)
    max_concurrent_requests: int = Field(default=5, ge=1)
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
    )


class FlightSearchConfig(BaseModel):
    """Configuration for `FlightSearchClient` Skyscanner lookups and deep links."""

    enabled: bool = Field(default=True)
    provider: str = Field(default="skyscanner")
    currency: str = Field(default="EUR")
    market: str = Field(default="IT")
    locale: str = Field(default="it-IT")
    skyscanner_base_url: str = Field(default="https://www.skyscanner.it/trasporti/voli")
    rapidapi_host: str = Field(default="sky-scrapper.p.rapidapi.com")
    timeout_seconds: float = Field(default=20.0, gt=0)
    max_results_per_date: int = Field(default=3, ge=1)
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
    )


class ServerConfig(BaseModel):
    """Configuration for the FastAPI backend server and web client connection."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = Field(default="http://localhost:8000")
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "null",
        ]
    )


class ObservabilityConfig(BaseModel):
    """Configuration for OpenTelemetry tracing (Tempo), logging (Loki), and metrics (Grafana)."""

    enabled: bool = Field(default=True)
    service_name: str = Field(default="weroad-travel-assistant")
    environment: str = Field(default="development")
    otlp_traces_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="Grafana Tempo OTLP HTTP traces endpoint.",
    )
    otlp_logs_endpoint: str = Field(
        default="http://localhost:3100/otlp/v1/logs",
        description="Grafana Loki native OTLP HTTP logs endpoint.",
    )
    otlp_metrics_endpoint: str = Field(
        default="http://localhost:4318/v1/metrics",
        description="Optional OTLP HTTP metrics endpoint.",
    )
    grafana_base_url: str = Field(
        default="http://localhost:3000",
        description="Base URL of Grafana UI for constructing direct Tempo trace links.",
    )
    capture_prompts: bool = Field(
        default=True,
        description="Whether to include prompt/completion snippets in OpenTelemetry span attributes.",
    )
    json_logs: bool = Field(
        default=True,
        description="Emit structured JSON logs with trace_id and span_id for Loki ingestion.",
    )
    log_level: str = Field(default="INFO")


class TokenUsageSummary(BaseModel):
    """Aggregated LLM token usage and trace metadata for a single assistant request."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    llm_requests: int = Field(default=0, ge=0)
    by_stage: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Per-stage token breakdown (e.g., guardrail_input, intent_extraction, proposal_synthesis, guardrail_output).",
    )
    trace_id: Optional[str] = Field(
        None,
        description="32-character OpenTelemetry trace ID for correlating in Grafana Tempo & Loki.",
    )
    grafana_trace_url: Optional[str] = Field(
        None,
        description="Direct URL to inspect this trace in Grafana Explore (Tempo).",
    )


class WeRoadPipelineConfig(BaseModel):
    """LLM and execution limit settings for the two-stage WeRoad agent pipeline."""

    llm_default: LLMConfig = Field(default_factory=LLMConfig)
    llm_intent: Optional[LLMConfig] = None
    llm_proposal: Optional[LLMConfig] = None
    intent_request_limit: int = Field(default=4, ge=1)
    proposal_request_limit: int = Field(default=25, ge=1)
    tool_retries: int = Field(default=4, ge=1)
    require_departure_airport: bool = Field(default=True)

    @property
    def resolved_intent_llm(self) -> LLMConfig:
        return self.llm_intent or self.llm_default

    @property
    def resolved_proposal_llm(self) -> LLMConfig:
        return self.llm_proposal or self.llm_default


class WeRoadSchemasPathsConfig(BaseModel):
    """External schema/tool configuration file paths."""

    mcp_config_path: str = Field(default="mcp_config.json")


class WeRoadPromptsConfig(BaseModel):
    """System and user prompt templates for the WeRoad orchestrator."""

    intent_system_prompt: str = Field(
        default=(
            "You are a travel query parser specializing in WeRoad.it group tours. "
            "Today's date is {today_str}.\n"
            "Extract the user's input language, target destination, departure airport, date constraints, "
            "trip duration, budget, travel style, and traveler age into the `TravelSearchIntent` schema.\n"
            "CRITICAL:\n"
            "- Always detect the language of the user's input and set `user_language` to its ISO 639-1 code "
            "(e.g., 'it', 'en', 'es', 'fr', 'de').\n"
            "- Always populate `destination_slug_keywords` with lowercase Italian terms "
            "matching WeRoad URL slugs (e.g., 'Japan' -> ['giappone'], 'Iceland' -> ['islanda'], "
            "'Morocco' -> ['marocco'], 'Peru' -> ['peru'], 'Thailand' -> ['thailandia']).\n"
            "- Extract `departure_airport` and `departure_airport_iata` when specified; if omitted, "
            "populate `missing_airport_reply` in the exact same language as the user's input."
        )
    )
    proposal_system_prompt: str = Field(
        default=(
            "You are an expert WeRoad travel advisor. Today's date is {today_str}.\n"
            "CRITICAL LANGUAGE REQUIREMENT:\n"
            "- Write ALL user-facing fields (`assistant_message`, `why_it_fits`, and `alternative_suggestions`) "
            "in the EXACT SAME LANGUAGE as the user's original query (`TravelSearchIntent.user_language`).\n"
            "- Do NOT switch to English or Italian if the user wrote in another language.\n\n"
            "Given the user's travel request and parsed `TravelSearchIntent`:\n"
            "1. Call `search_weroad_tours` using `destination_slug_keywords` and duration/budget filters.\n"
            "2. For the top 2-3 matching itineraries, call `get_tour_departures` (batching tool calls in parallel when possible) "
            "using the tour `slug` and date bounds (`earliest_date`, `latest_date` in YYYY-MM-DD format if specified).\n"
            "3. Prioritize departures with status 'Confermato' or 'Quasi pieno', followed by 'Disponibile'. "
            "Exclude 'Sold out' departures from `matching_dates` unless no other slots exist.\n"
            "4. Call `search_flights` (batching tool calls in parallel when possible) for the top 2-3 available departure dates "
            "and attach results to `matching_flights`.\n"
            "5. Report flight costs in EUR, flight durations, stops, and clickable Skyscanner links (`booking_url`) "
            "in the user's input language."
        )
    )
    stage2_user_prompt_template: str = Field(
        default=(
            "Original user query: {user_query}\n\n"
            "Structured TravelSearchIntent:\n{intent_json}\n\n"
            "REMINDER: Write `assistant_message`, every `why_it_fits`, and `alternative_suggestions` "
            "strictly in the same language as the original user query above (`user_language`)."
        )
    )
    missing_departure_airport_message: str = Field(
        default=(
            "Please specify which airport you would like to depart from "
            "(for example: Milan Malpensa MXP, Rome Fiumicino FCO, Venice VCE, Bologna BLQ, or Naples NAP) "
            "so I can check available WeRoad tour dates and matching round-trip flights on Skyscanner."
        )
    )


class WeRoadAssistantConfig(BaseModel):
    """Root typed configuration for the WeRoad Travel Assistant."""

    pipeline: WeRoadPipelineConfig = Field(default_factory=WeRoadPipelineConfig)
    llama_guard: LlamaGuardConfig = Field(default_factory=LlamaGuardConfig)
    scraper: WeRoadScraperConfig = Field(default_factory=WeRoadScraperConfig)
    flight_search: FlightSearchConfig = Field(default_factory=FlightSearchConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    schemas: WeRoadSchemasPathsConfig = Field(default_factory=WeRoadSchemasPathsConfig)
    prompts: WeRoadPromptsConfig = Field(default_factory=WeRoadPromptsConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "WeRoadAssistantConfig":
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Travel Domain Schemas
# ---------------------------------------------------------------------------


class DepartureStatus(str, Enum):
    """Normalized availability status of a WeRoad departure slot ('turno')."""

    CONFIRMED = "Confermato"
    ALMOST_FULL = "Quasi pieno"
    AVAILABLE = "Disponibile"
    SOLD_OUT = "Sold out"
    UNKNOWN = "Sconosciuto"


class TravelSearchIntent(BaseModel):
    """Structured travel criteria extracted from a user's natural-language query."""

    user_language: str = Field(
        default="en",
        description=(
            "ISO 639-1 language code detected from the user's input query "
            "(e.g., 'it' for Italian, 'en' for English, 'es' for Spanish, 'fr' for French, 'de' for German)."
        ),
    )
    destination: str = Field(
        ...,
        description="Target country, region, or city requested by the user.",
    )
    destination_slug_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Normalized lowercase Italian keywords used to match WeRoad URL slugs and titles "
            "(e.g., ['giappone', 'tokyo'] for Japan, ['islanda'] for Iceland)."
        ),
    )
    departure_airport: Optional[str] = Field(
        None,
        description="Departure airport or city specified by the user (e.g., 'Milan Malpensa', 'Rome Fiumicino').",
    )
    departure_airport_iata: Optional[str] = Field(
        None,
        description="3-letter IATA airport code for the departure airport (e.g., 'MXP', 'FCO', 'VCE').",
    )
    missing_airport_reply: Optional[str] = Field(
        None,
        description=(
            "When `departure_airport` is null, a polite, concise follow-up question written in the EXACT "
            "same language as the user's input (`user_language`) asking which departure airport they prefer."
        ),
    )
    destination_airport_iata: Optional[str] = Field(
        None,
        description="3-letter IATA airport or city code for the main destination hub (e.g., 'TYO', 'KEF', 'RAK').",
    )
    earliest_departure: Optional[date] = Field(
        None,
        description="Earliest acceptable departure date (YYYY-MM-DD).",
    )
    latest_departure: Optional[date] = Field(
        None,
        description="Latest acceptable departure date (YYYY-MM-DD).",
    )
    preferred_months: List[int] = Field(
        default_factory=list,
        description="Preferred departure months as integers 1-12.",
    )
    min_days: Optional[int] = Field(
        None,
        description="Minimum desired trip duration in days.",
    )
    max_days: Optional[int] = Field(
        None,
        description="Maximum desired trip duration in days.",
    )
    max_budget_eur: Optional[float] = Field(
        None,
        description="Maximum price per person in EUR, if specified.",
    )
    travel_style: Optional[str] = Field(
        None,
        description="Optional WeRoad mood/tier: '360°', 'Express', 'Backpack', 'Collection', 'Trekking'.",
    )
    traveler_age: Optional[int] = Field(
        None,
        description="Traveler's age if mentioned, used to match WeRoad age cohorts (25-35, 35-49).",
    )


class DepartureSlot(BaseModel):
    """A single scheduled departure date ('turno') for a WeRoad itinerary."""

    start_date: date
    end_date: date
    price_eur: float = Field(..., description="Current price per person in EUR.")
    original_price_eur: Optional[float] = Field(
        None,
        description="Original undiscounted price if a promotion is active.",
    )
    status: DepartureStatus = Field(default=DepartureStatus.AVAILABLE)
    age_bracket: Optional[str] = Field(
        None,
        description="Age group cohort such as '25-35', '35-49', or 'Tutti'.",
    )


class FlightOption(BaseModel):
    """Round-trip flight option on Skyscanner aligned with a WeRoad departure slot."""

    origin_airport: str = Field(..., description="3-letter origin IATA airport code.")
    destination_airport: str = Field(..., description="3-letter destination IATA airport code.")
    departure_date: date
    return_date: date
    airline: str = Field(..., description="Operating carrier(s) or Skyscanner search summary.")
    price_eur: Optional[float] = Field(
        None,
        description="Round-trip flight price per passenger in EUR.",
    )
    is_estimated_price: bool = Field(
        default=False,
        description="True when the price/duration is a typical route estimate rather than a live API quote.",
    )
    stops: Optional[int] = Field(
        None,
        description="Number of outbound/typical stops (0 = direct).",
    )
    duration: Optional[str] = Field(
        None,
        description="Formatted flight duration (e.g., '13h 45m').",
    )
    booking_url: str = Field(..., description="Direct Skyscanner round-trip search link.")
    source: str = Field(default="skyscanner")


class WeRoadTourSummary(BaseModel):
    """Lightweight catalog entry representing a discovered WeRoad itinerary."""

    title: str
    slug: str
    url: str
    destination_keywords: List[str] = Field(default_factory=list)
    duration_days: Optional[int] = None
    starting_price_eur: Optional[float] = None
    summary: str = ""


class RecommendedTourProposal(BaseModel):
    """A ranked WeRoad tour recommendation tailored to the user's preferences."""

    tour_title: str
    tour_url: HttpUrl
    duration_days: Optional[int] = None
    starting_price_eur: Optional[float] = None
    why_it_fits: str = Field(
        ...,
        description="Concise explanation written in the exact same language as the user's input query explaining why this itinerary matches the request.",
    )
    matching_dates: List[DepartureSlot] = Field(
        default_factory=list,
        description="Available or confirmed departure slots matching the user's timeframe.",
    )
    matching_flights: List[FlightOption] = Field(
        default_factory=list,
        description="Round-trip Skyscanner flight options matching the departure airport and tour dates.",
    )


class TravelProposalResponse(BaseModel):
    """Final structured output returned by the WeRoad Travel Assistant."""

    assistant_message: str = Field(
        ...,
        description="Friendly, helpful response written in the exact same language as the user's input query summarizing the recommended tours, dates, and flights.",
    )
    proposals: List[RecommendedTourProposal] = Field(default_factory=list)
    alternative_suggestions: Optional[str] = Field(
        None,
        description="Helpful fallback advice written in the exact same language as the user's input query if exact dates or budgets had limited availability.",
    )
    needs_departure_airport: bool = Field(
        default=False,
        description="True when the user did not specify a departure airport and must provide one first.",
    )
    guardrail_blocked: bool = Field(
        default=False,
        description="True if the user input or generated response was blocked by Llama Guard.",
    )
    guardrail_violations: List[str] = Field(
        default_factory=list,
        description="Safety categories triggered when `guardrail_blocked` is True.",
    )
    telemetry: Optional[TokenUsageSummary] = Field(
        default=None,
        description="LLM token usage breakdown and Grafana Tempo trace correlation metadata.",
    )


# ---------------------------------------------------------------------------
# Real-Time Agent Streaming Event Schemas
# ---------------------------------------------------------------------------


class StreamStepType(str, Enum):
    """Semantic classification of a real-time agent execution or reasoning event."""

    STATUS = "status"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL = "final"
    ERROR = "error"


class AgentStreamEvent(BaseModel):
    """Single Server-Sent Event (SSE) emitted during live agent execution."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    step_type: StreamStepType
    stage: str = Field(
        ...,
        description="Pipeline stage identifier (e.g., 'guardrail_input', 'intent_extraction', 'proposal_synthesis', 'tool_execution', 'guardrail_output', 'complete').",
    )
    title: str = Field(..., description="Short scannable headline for the UI timeline.")
    detail: str = Field(
        ...,
        description="Human-readable intermediate reasoning or action detail (e.g., 'Looking for round-trip flights from MXP to TYO...').",
    )
    tool_name: Optional[str] = Field(
        None,
        description="Invoked MCP/function tool name when `step_type` is 'tool_call' or 'tool_result'.",
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured arguments passed to the tool.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra metadata (e.g., elapsed_ms, item_count, destination).",
    )
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    response: Optional[TravelProposalResponse] = Field(
        None,
        description="Final structured `TravelProposalResponse` attached when `step_type == 'final'`.",
    )


# ---------------------------------------------------------------------------
# FastAPI Request & Response Schemas
# ---------------------------------------------------------------------------


class ChatQueryRequest(BaseModel):
    """Request payload accepted by `POST /api/v1/chat` and `POST /api/v1/chat/stream`."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language travel query from the user.",
    )
    departure_airport: Optional[str] = Field(
        None,
        description="Optional explicit departure airport name or 3-letter IATA code (e.g., 'MXP', 'Rome Fiumicino').",
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional client session identifier for tracing and log correlation.",
    )


class HealthCheckResponse(BaseModel):
    """Health and capability status returned by `GET /health`."""

    status: str = Field(default="ok")
    service: str = Field(default="weroad-travel-assistant-api")
    guardrail_enabled: bool = Field(default=True)
    flight_provider: str = Field(default="skyscanner")
    default_model: str = Field(default="openrouter:openrouter/free")