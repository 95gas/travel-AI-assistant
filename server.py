"""
FastAPI Backend Server (`server.py`)
====================================
Exposes the guarded two-stage `WeRoadTravelAssistant` over HTTP REST endpoints
for consumption by the Streamlit web interface or any external client.

Endpoints:
- `GET  /`            : Service metadata and documentation link
- `GET  /health`      : Health check and active guardrail/model status
- `GET  /metrics`     : OpenTelemetry / Prometheus metrics exposition
- `POST /api/v1/chat` : Executes a guarded WeRoad tour + Skyscanner flight query

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    # or:
    python server.py
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ai_assistant.core.config_management import ConfigManager
from src.ai_assistant.core.orchestrator import WeRoadTravelAssistant
from src.ai_assistant.observability.observability import (
    get_current_trace_id,
    get_tracer,
    render_prometheus_metrics,
    setup_observability,
)
from src.ai_assistant.schemas.schemas import (
    ChatQueryRequest,
    HealthCheckResponse,
    ServerConfig,
    TravelProposalResponse,
)

logger = logging.getLogger("WeRoadFastAPIServer")
_tracer = get_tracer("weroad.fastapi")

# Global runtime references managed via FastAPI lifespan
_config_manager: Optional[ConfigManager] = None
_assistant: Optional[WeRoadTravelAssistant] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initializes `ConfigManager`, OpenTelemetry observability, and `WeRoadTravelAssistant`."""
    global _config_manager, _assistant
    _config_manager = ConfigManager.get_instance()
    setup_observability(config=_config_manager.observability_config)
    logger.info("Starting WeRoad FastAPI Backend Server...")
    _assistant = WeRoadTravelAssistant(config_manager=_config_manager)
    logger.info(
        f"WeRoadTravelAssistant ready (model={_config_manager.llm_default_config.model_name}, "
        f"guardrail_enabled={_config_manager.llama_guard_config.enabled}, "
        f"observability_enabled={_config_manager.observability_config.enabled})."
    )
    yield
    logger.info("Shutting down WeRoad FastAPI Backend Server...")


# Load initial config for CORS middleware registration
_boot_config = ConfigManager.get_instance()
_boot_server_cfg = getattr(_boot_config, "server_config", ServerConfig())

app = FastAPI(
    title="WeRoad Agentic Travel Assistant API",
    description=(
        "Two-stage PydanticAI travel assistant with Llama Guard I/O moderation, "
        "WeRoad.it catalog & departure scraping, and Skyscanner round-trip flight lookups."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_boot_server_cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)


@app.middleware("http")
async def tracing_middleware(request: Request, call_next) -> Response:
    """Creates a top-level HTTP trace span and exposes `X-Trace-Id` in response headers."""
    span_name = f"HTTP {request.method} {request.url.path}"
    with _tracer.start_as_current_span(span_name) as span:
        if span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.route", request.url.path)
        trace_id = get_current_trace_id()
        response = await call_next(request)
        if span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("http.status_code", response.status_code)
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return response


@app.get("/", tags=["Meta"])
async def root() -> dict[str, str]:
    """Root endpoint providing API navigation links."""
    return {
        "service": "WeRoad Agentic Travel Assistant API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "chat_endpoint": "/api/v1/chat",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check() -> HealthCheckResponse:
    """Returns service health status and active configuration metadata."""
    cfg = _config_manager or ConfigManager.get_instance()
    return HealthCheckResponse(
        status="ok" if _assistant is not None else "initializing",
        guardrail_enabled=cfg.llama_guard_config.enabled,
        flight_provider=cfg.flight_search_config.provider,
        default_model=cfg.llm_default_config.model_name,
    )


@app.get("/metrics", tags=["Observability"])
async def prometheus_metrics() -> Response:
    """Exposes OpenTelemetry / Prometheus metrics (LLM token usage, latency, guardrails)."""
    payload, content_type = render_prometheus_metrics()
    return Response(content=payload, media_type=content_type)


@app.post("/api/v1/chat", response_model=TravelProposalResponse, tags=["Travel Assistant"])
async def chat_query(request: ChatQueryRequest) -> TravelProposalResponse:
    """
    Processes a natural-language travel query through:
    1. Llama Guard input moderation
    2. Stage 1 `TravelSearchIntent` extraction (including departure airport check)
    3. Stage 2 WeRoad tour & departure lookup + Skyscanner round-trip flight search
    4. Llama Guard output moderation
    """
    if _assistant is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="WeRoadTravelAssistant is still initializing. Please retry shortly.",
        )

    effective_query = request.query.strip()
    if request.departure_airport and request.departure_airport.strip():
        effective_query = f"{effective_query} (departing from {request.departure_airport.strip()})"

    logger.info(
        f"Received /api/v1/chat request [session={request.session_id or 'anonymous'}]: "
        f"'{effective_query[:80]}'"
    )

    try:
        response = await _assistant.handle_query(effective_query)
        return response
    except Exception as exc:
        logger.exception(f"Unhandled error during /api/v1/chat execution: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process travel query: {exc}",
        ) from exc


if __name__ == "__main__":
    cfg = getattr(ConfigManager.get_instance(), "server_config", ServerConfig())
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        reload=False,
    )