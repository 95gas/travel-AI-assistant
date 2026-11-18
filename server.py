"""
FastAPI Backend Server (`server.py`)
====================================
Exposes the guarded two-stage `WeRoadTravelAssistant` over HTTP REST and Server-Sent
Events (SSE) endpoints, and serves the modern reactive web interface (`index.html`).

Endpoints:
- `GET  /`                   : Serves the web UI (`index.html`) for browsers, or service metadata JSON
- `GET  /ui`                 : Serves the web UI (`index.html`)
- `GET  /health`             : Health check and active guardrail/model status
- `GET  /metrics`            : OpenTelemetry / Prometheus metrics exposition
- `POST /api/v1/chat`        : Executes a guarded WeRoad tour + Skyscanner flight query (JSON)
- `POST /api/v1/chat/stream` : Streams real-time agent reasoning, tool steps, and final response (SSE)

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    # or:
    python server.py
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

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
    AgentStreamEvent,
    ChatQueryRequest,
    HealthCheckResponse,
    ServerConfig,
    StreamStepType,
    TravelProposalResponse,
)

logger = logging.getLogger("WeRoadFastAPIServer")
_tracer = get_tracer("weroad.fastapi")

FRONTEND_HTML_PATH = PROJECT_ROOT / "index.html"

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
        "WeRoad.it catalog & departure scraping, Skyscanner round-trip flight lookups, "
        "and real-time SSE agent step streaming."
    ),
    version="1.1.0",
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
async def root(request: Request) -> Response:
    """Serves the modern web interface when opened in a browser, or API metadata JSON otherwise."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and FRONTEND_HTML_PATH.exists():
        return FileResponse(FRONTEND_HTML_PATH, media_type="text/html")

    return JSONResponse(
        {
            "service": "WeRoad Agentic Travel Assistant API",
            "ui": "/ui",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "chat_endpoint": "/api/v1/chat",
            "stream_endpoint": "/api/v1/chat/stream",
        }
    )


@app.get("/ui", tags=["UI"])
async def serve_ui() -> FileResponse:
    """Serves the reactive WeRoad AI Travel & Flight Finder frontend (`index.html`)."""
    if not FRONTEND_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend file `index.html` not found in project root.",
        )
    return FileResponse(FRONTEND_HTML_PATH, media_type="text/html")


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


@app.post("/api/v1/chat/stream", tags=["Travel Assistant"])
async def chat_query_stream(request: ChatQueryRequest) -> StreamingResponse:
    """
    Streams real-time Server-Sent Events (`text/event-stream`) representing every step
    and intermediate reasoning stage of the WeRoad Travel Assistant, concluding with
    the final `TravelProposalResponse`.
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
        f"Received /api/v1/chat/stream request [session={request.session_id or 'anonymous'}]: "
        f"'{effective_query[:80]}'"
    )

    async def _sse_generator() -> AsyncIterator[str]:
        assert _assistant is not None
        try:
            async for event in _assistant.stream_query(effective_query):
                payload = event.model_dump_json()
                yield f"event: {event.step_type.value}\ndata: {payload}\n\n"
        except Exception as exc:
            logger.exception(f"Unhandled error during /api/v1/chat/stream execution: {exc}")
            err_event = AgentStreamEvent(
                step_type=StreamStepType.ERROR,
                stage="error",
                title="Execution Error",
                detail=f"Failed to process travel query: {exc}",
            )
            yield f"event: error\ndata: {err_event.model_dump_json()}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    cfg = getattr(ConfigManager.get_instance(), "server_config", ServerConfig())
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        reload=False,
    )