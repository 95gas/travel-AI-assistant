"""
Observability & Telemetry Instrumentation (`observability.py`)
==============================================================
Configures unified OpenTelemetry tracing, structured logging, and metrics for:
- **Grafana Tempo**: Distributed traces via OTLP HTTP (`http://localhost:4318/v1/traces`)
  covering HTTP requests, Llama Guard checks, Stage 1/2 PydanticAI agents, and MCP tools.
- **Grafana Loki**: Native OTLP HTTP log shipping (`http://localhost:3100/otlp/v1/logs`)
  and structured JSON console logs enriched with `trace_id` and `span_id`.
- **Prometheus & Grafana**: LLM token counters (`input`, `output`, `total`), latency
  histograms, and guardrail/tool counters exposed via `/metrics` and Tempo span-metrics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.parse
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.ai_assistant.schemas.schemas import ObservabilityConfig, TokenUsageSummary

logger = logging.getLogger("WeRoad.Observability")

# Optional OpenTelemetry imports with graceful degradation
try:
    from opentelemetry import metrics, trace
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

_INITIALIZED = False
_ACTIVE_CONFIG: Optional[ObservabilityConfig] = None
_TOKEN_COUNTER: Any = None
_LLM_REQUEST_COUNTER: Any = None
_LLM_DURATION_HISTOGRAM: Any = None
_GUARDRAIL_COUNTER: Any = None


class TraceContextLogFilter(logging.Filter):
    """Injects active OpenTelemetry `trace_id` and `span_id` into every Python `LogRecord`."""

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = ""
        span_id = ""
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span is not None:
                ctx = span.get_span_context()
                if ctx and ctx.is_valid:
                    trace_id = f"{ctx.trace_id:032x}"
                    span_id = f"{ctx.span_id:016x}"
        record.trace_id = trace_id  # type: ignore[attr-defined]
        record.span_id = span_id  # type: ignore[attr-defined]
        return True


class JSONLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects compatible with Grafana Loki."""

    def __init__(self, service_name: str, environment: str):
        super().__init__()
        self.service_name = service_name
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "environment": self.environment,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", "")
        span_id = getattr(record, "span_id", "")
        if trace_id:
            payload["trace_id"] = trace_id
        if span_id:
            payload["span_id"] = span_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_observability(
    config: Optional[ObservabilityConfig] = None,
    service_name_override: Optional[str] = None,
) -> None:
    """
    Initializes OpenTelemetry tracing (Tempo), logging (Loki), and metrics (Prometheus/Grafana)
    once per process.
    """
    global _INITIALIZED, _ACTIVE_CONFIG
    global _TOKEN_COUNTER, _LLM_REQUEST_COUNTER, _LLM_DURATION_HISTOGRAM, _GUARDRAIL_COUNTER

    if _INITIALIZED:
        return

    cfg = config or ObservabilityConfig()
    _ACTIVE_CONFIG = cfg
    service_name = service_name_override or os.getenv("OTEL_SERVICE_NAME") or cfg.service_name
    environment = os.getenv("OTEL_ENVIRONMENT") or cfg.environment
    numeric_level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    # Configure root Python logging with trace_id/span_id injection
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    trace_filter = TraceContextLogFilter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.addFilter(trace_filter)
    if cfg.json_logs:
        stream_handler.setFormatter(JSONLogFormatter(service_name=service_name, environment=environment))
    else:
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s [trace_id=%(trace_id)s]: %(message)s"
            )
        )

    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)

    if not cfg.enabled:
        _INITIALIZED = True
        logger.info("Observability telemetry disabled in configuration.")
        return

    if not OTEL_AVAILABLE:
        _INITIALIZED = True
        logger.warning(
            "OpenTelemetry SDK packages not installed; falling back to local structured logging only."
        )
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": environment,
        }
    )

    # 1. Tracing -> Grafana Tempo (OTLP HTTP)
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or cfg.otlp_traces_endpoint
    try:
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=traces_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        logger.info(f"OpenTelemetry Tempo trace exporter initialized -> {traces_endpoint}")
    except Exception as exc:
        logger.warning(f"Failed to initialize Tempo OTLP trace exporter: {exc}")

    # 2. Logging -> Grafana Loki (Native OTLP HTTP)
    logs_endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT") or cfg.otlp_logs_endpoint
    try:
        logger_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(endpoint=logs_endpoint)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logger_provider)

        otel_log_handler = LoggingHandler(level=numeric_level, logger_provider=logger_provider)
        otel_log_handler.addFilter(trace_filter)
        root_logger.addHandler(otel_log_handler)
        logger.info(f"OpenTelemetry Loki log exporter initialized -> {logs_endpoint}")
    except Exception as exc:
        logger.warning(f"Failed to initialize Loki OTLP log exporter: {exc}")

    # 3. Metrics -> Prometheus & Grafana
    try:
        metric_readers = []
        if PROMETHEUS_AVAILABLE:
            metric_readers.append(PrometheusMetricReader())
        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter("weroad.assistant.telemetry", "1.0.0")

        _TOKEN_COUNTER = meter.create_counter(
            name="weroad_llm_tokens",
            unit="1",
            description="Total LLM tokens consumed broken down by stage, model, and token_type.",
        )
        _LLM_REQUEST_COUNTER = meter.create_counter(
            name="weroad_llm_requests",
            unit="1",
            description="Total LLM invocations broken down by stage, model, and status.",
        )
        _LLM_DURATION_HISTOGRAM = meter.create_histogram(
            name="weroad_llm_request_duration_seconds",
            unit="s",
            description="Duration of LLM calls in seconds.",
        )
        _GUARDRAIL_COUNTER = meter.create_counter(
            name="weroad_guardrail_checks",
            unit="1",
            description="Llama Guard safety checks broken down by target and decision.",
        )
    except Exception as exc:
        logger.warning(f"Failed to initialize OpenTelemetry metrics: {exc}")

    _INITIALIZED = True


def get_tracer(name: str = "weroad.assistant") -> Any:
    """Returns an OpenTelemetry tracer or a no-op fallback object."""
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)

    class _NoOpTracer:
        def start_as_current_span(self, *_args: Any, **_kwargs: Any) -> Any:
            return nullcontext()

    return _NoOpTracer()


def get_current_trace_id() -> Optional[str]:
    """Returns the active 32-hex-character OpenTelemetry trace ID if inside a valid span."""
    if not OTEL_AVAILABLE:
        return None
    span = trace.get_current_span()
    if span is None:
        return None
    ctx = span.get_span_context()
    if ctx and ctx.is_valid:
        return f"{ctx.trace_id:032x}"
    return None


def build_grafana_trace_url(
    trace_id: Optional[str],
    config: Optional[ObservabilityConfig] = None,
) -> Optional[str]:
    """Constructs a direct Grafana Explore link to inspect a trace in Tempo."""
    if not trace_id:
        return None
    cfg = config or _ACTIVE_CONFIG or ObservabilityConfig()
    base = cfg.grafana_base_url.rstrip("/")
    left_pane = json.dumps(
        {
            "datasource": "tempo",
            "queries": [{"refId": "A", "datasource": {"type": "tempo", "uid": "tempo"}, "queryType": "traceql", "query": trace_id}],
        }
    )
    encoded = urllib.parse.quote(left_pane, safe="")
    return f"{base}/explore?orgId=1&left={encoded}"


def extract_usage_from_result(result: Any) -> Dict[str, int]:
    """Safely extracts token counts from a PydanticAI `RunResult` object."""
    usage_obj = None
    if hasattr(result, "usage") and callable(result.usage):
        try:
            usage_obj = result.usage()
        except Exception:
            usage_obj = None

    if usage_obj is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 1}

    input_tokens = int(
        getattr(usage_obj, "input_tokens", None)
        or getattr(usage_obj, "request_tokens", None)
        or 0
    )
    output_tokens = int(
        getattr(usage_obj, "output_tokens", None)
        or getattr(usage_obj, "response_tokens", None)
        or 0
    )
    total_tokens = int(getattr(usage_obj, "total_tokens", None) or (input_tokens + output_tokens))
    requests = int(getattr(usage_obj, "requests", None) or 1)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests": requests,
    }


def record_llm_usage(
    stage: str,
    model_name: str,
    usage: Dict[str, int],
    duration_seconds: float,
    status: str = "ok",
) -> None:
    """Attaches GenAI token usage attributes to the active span and increments metrics."""
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    if OTEL_AVAILABLE:
        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("weroad.stage", stage)
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.set_attribute("weroad.llm.duration_seconds", round(duration_seconds, 4))
            if status != "ok":
                span.set_status(Status(StatusCode.ERROR, description=status))

    if _TOKEN_COUNTER is not None:
        if input_tokens > 0:
            _TOKEN_COUNTER.add(
                input_tokens,
                {"stage": stage, "model": model_name, "token_type": "input"},
            )
        if output_tokens > 0:
            _TOKEN_COUNTER.add(
                output_tokens,
                {"stage": stage, "model": model_name, "token_type": "output"},
            )
        if total_tokens > 0:
            _TOKEN_COUNTER.add(
                total_tokens,
                {"stage": stage, "model": model_name, "token_type": "total"},
            )

    if _LLM_REQUEST_COUNTER is not None:
        _LLM_REQUEST_COUNTER.add(1, {"stage": stage, "model": model_name, "status": status})

    if _LLM_DURATION_HISTOGRAM is not None:
        _LLM_DURATION_HISTOGRAM.record(duration_seconds, {"stage": stage, "model": model_name})


def record_guardrail_event(
    target: str,
    is_safe: bool,
    categories: list[str],
    duration_seconds: float,
) -> None:
    """Records Llama Guard moderation span attributes and counters."""
    decision = "safe" if is_safe else "blocked"
    if OTEL_AVAILABLE:
        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("guardrail.target", target)
            span.set_attribute("guardrail.is_safe", is_safe)
            span.set_attribute("guardrail.decision", decision)
            if categories:
                span.set_attribute("guardrail.violations", ", ".join(categories))

    if _GUARDRAIL_COUNTER is not None:
        _GUARDRAIL_COUNTER.add(1, {"target": target, "decision": decision})


def build_token_usage_summary(
    stage_usages: Dict[str, Dict[str, int]],
    config: Optional[ObservabilityConfig] = None,
) -> TokenUsageSummary:
    """Aggregates per-stage token usage dictionaries into a `TokenUsageSummary`."""
    total_in = sum(u.get("input_tokens", 0) for u in stage_usages.values())
    total_out = sum(u.get("output_tokens", 0) for u in stage_usages.values())
    total_all = sum(u.get("total_tokens", 0) for u in stage_usages.values())
    total_reqs = sum(u.get("requests", 0) for u in stage_usages.values())
    trace_id = get_current_trace_id()

    return TokenUsageSummary(
        input_tokens=total_in,
        output_tokens=total_out,
        total_tokens=total_all or (total_in + total_out),
        llm_requests=total_reqs,
        by_stage=stage_usages,
        trace_id=trace_id,
        grafana_trace_url=build_grafana_trace_url(trace_id, config=config),
    )


def render_prometheus_metrics() -> tuple[bytes, str]:
    """Renders Prometheus text exposition metrics for FastAPI `/metrics`."""
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not installed\n", "text/plain; charset=utf-8"
    return generate_latest(), CONTENT_TYPE_LATEST