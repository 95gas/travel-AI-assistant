"""
main.py
=======
CLI & Interactive Entrypoint for the WeRoad Agentic Travel Assistant.

Features:
- Loads centralized configuration via `ConfigManager` (`config/config_weroad.yaml`
  or `config.yaml` fallback) and initializes `WeRoadTravelAssistant`.
- Enforces Llama Guard input and output safety moderation on every turn.
- Supports both single-query execution (`--query`) and multi-turn interactive
  chat mode (`--interactive` or default when no query is provided).
- Automatically prompts for a departure airport if omitted in the initial query
  and re-runs the search so Skyscanner round-trip flight costs, durations,
  and booking links are always included.
- Supports human-readable terminal formatting as well as `--json` output.

Usage Examples:
    Interactive Chat Mode:
        python main.py
        python main.py --interactive

    Single-Query Mode:
        python main.py --query "I want to visit Japan in October for 10 days leaving from Milan Malpensa"
        python main.py -q "Iceland in August" --airport FCO
        python main.py -q "Morocco in November from Bologna" --json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Support both `ai_assistant.*` package structure and flat root module layout
try:
    from ai_assistant.core.config_management import ConfigManager
    from ai_assistant.core.orchestrator import WeRoadTravelAssistant
    from ai_assistant.schemas.schemas import LLMConfig, TravelProposalResponse
except ImportError:
    from config_management import ConfigManager  # type: ignore[no-redef]
    from orchestrator import WeRoadTravelAssistant  # type: ignore[no-redef]
    from common.schemas import LLMConfig, TravelProposalResponse  # type: ignore[no-redef]

from ai_assistant.observability.observability import setup_observability

logger = logging.getLogger("WeRoadCLI")


def setup_logging(log_level: str = "INFO") -> None:
    """Configures console logging with timestamp, severity level, and logger name."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def resolve_config_path(explicit_path: Optional[str]) -> Path:
    """
    Resolves the YAML configuration file path.
    Checks an explicit CLI path first, then standard project locations.
    """
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Configuration file not found at '{candidate.resolve()}'.")
        return candidate

    candidates = [
        Path("config/config_weroad.yaml"),
        Path("config_weroad.yaml"),
        Path("config/config.yaml"),
        Path("config.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return path

    # Fall back to default expected path (ConfigManager will apply schema defaults)
    return Path("config/config_weroad.yaml")


def format_proposal_response(response: TravelProposalResponse) -> str:
    """Formats a `TravelProposalResponse` into a clean, readable terminal report."""
    lines: list[str] = []
    divider = "=" * 72

    lines.append(divider)

    # 1. Guardrail block notice
    if response.guardrail_blocked:
        lines.append("[SAFETY GUARDRAIL BLOCKED]")
        if response.guardrail_violations:
            lines.append(f"Triggered categories: {', '.join(response.guardrail_violations)}")
        lines.append(response.assistant_message)
        lines.append(divider)
        return "\n".join(lines)

    # 2. Primary assistant summary
    lines.append(response.assistant_message)

    # 3. Structured tour proposals, departure dates, and Skyscanner flights
    if response.proposals:
        lines.append("")
        lines.append("-" * 72)
        lines.append("RECOMMENDED WEROAD TOURS & SKYSCANNER FLIGHTS")
        lines.append("-" * 72)

        for idx, tour in enumerate(response.proposals, start=1):
            meta_parts: list[str] = []
            if tour.duration_days is not None:
                meta_parts.append(f"{tour.duration_days} days")
            if tour.starting_price_eur is not None:
                meta_parts.append(f"from EUR {tour.starting_price_eur:.2f}")
            meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""

            lines.append(f"\n{idx}. {tour.tour_title}{meta_suffix}")
            lines.append(f"   Tour URL : {tour.tour_url}")
            lines.append(f"   Why fit  : {tour.why_it_fits}")

            if tour.matching_dates:
                lines.append("   Available WeRoad Departures:")
                for slot in tour.matching_dates:
                    age_info = f" | Age: {slot.age_bracket}" if slot.age_bracket else ""
                    promo_info = (
                        f" (was EUR {slot.original_price_eur:.2f})"
                        if slot.original_price_eur
                        else ""
                    )
                    lines.append(
                        f"     - {slot.start_date} -> {slot.end_date} | "
                        f"EUR {slot.price_eur:.2f}{promo_info} | Status: {slot.status.value}{age_info}"
                    )

            if tour.matching_flights:
                lines.append("   Matching Round-Trip Flights (Skyscanner):")
                for flight in tour.matching_flights:
                    if flight.price_eur is not None:
                        est_label = " (est.)" if flight.is_estimated_price else ""
                        price_text = f"EUR {flight.price_eur:.2f}{est_label}"
                    else:
                        price_text = "Check live fare"

                    duration_text = f" | Duration: {flight.duration}" if flight.duration else ""
                    if flight.stops is not None:
                        stops_text = (
                            " | Direct"
                            if flight.stops == 0
                            else f" | {flight.stops} stop{'s' if flight.stops > 1 else ''}"
                        )
                    else:
                        stops_text = ""

                    lines.append(
                        f"     - {flight.origin_airport} -> {flight.destination_airport} "
                        f"({flight.departure_date} to {flight.return_date}) | "
                        f"{flight.airline} | {price_text}{duration_text}{stops_text}"
                    )
                    lines.append(f"       Skyscanner Link: {flight.booking_url}")

    # 4. Alternative suggestions
    if response.alternative_suggestions:
        lines.append("")
        lines.append(f"Alternative Suggestions: {response.alternative_suggestions}")

    # 5. Token usage & Grafana Tempo trace telemetry
    if response.telemetry:
        t = response.telemetry
        lines.append("")
        lines.append(
            f"Telemetry: Tokens In={t.input_tokens} | Out={t.output_tokens} | "
            f"Total={t.total_tokens} | LLM Calls={t.llm_requests}"
        )
        if t.trace_id:
            lines.append(f"Trace ID : {t.trace_id}")
        if t.grafana_trace_url:
            lines.append(f"Grafana  : {t.grafana_trace_url}")

    lines.append(divider)
    return "\n".join(lines)


async def execute_travel_query(
    assistant: WeRoadTravelAssistant,
    user_query: str,
    default_airport: Optional[str] = None,
    allow_interactive_followup: bool = True,
) -> TravelProposalResponse:
    """
    Executes a single user travel query through the guarded WeRoad orchestrator.
    If the departure airport is missing and `allow_interactive_followup` is enabled,
    prompts the user for their departure airport and re-runs the request.
    """
    effective_query = user_query.strip()
    if default_airport:
        effective_query = f"{effective_query} (departing from {default_airport.strip()})"

    response = await assistant.handle_query(effective_query)

    # Handle missing departure airport follow-up seamlessly in terminal sessions
    if response.needs_departure_airport and allow_interactive_followup and sys.stdin.isatty():
        print(f"\nAssistant > {response.assistant_message}")
        try:
            airport_reply = input("Departure Airport (e.g., MXP, FCO, Venice) > ").strip()
        except (EOFError, KeyboardInterrupt):
            return response

        if airport_reply:
            enriched_query = f"{user_query.strip()} (departing from {airport_reply})"
            response = await assistant.handle_query(enriched_query)

    return response


async def run_interactive_session(
    assistant: WeRoadTravelAssistant,
    default_airport: Optional[str] = None,
    output_json: bool = False,
) -> None:
    """Runs an interactive multi-turn CLI chat loop for the WeRoad Travel Assistant."""
    print("=" * 72)
    print("  WeRoad Agentic Travel Assistant (with Skyscanner Flights & Guardrails)")
    print("  Type your travel request below, or type 'exit' / 'quit' to stop.")
    print("=" * 72)

    active_airport = default_airport

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting WeRoad Travel Assistant. Safe travels!")
            break

        if not user_input:
            continue

        lower_cmd = user_input.lower()
        if lower_cmd in {"exit", "quit", "/exit", "/quit"}:
            print("Exiting WeRoad Travel Assistant. Safe travels!")
            break

        if lower_cmd in {"/help", "help"}:
            print(
                "Commands:\n"
                "  - Ask any WeRoad travel question (include destination, dates/month, and departure airport).\n"
                "  - Type 'exit' or 'quit' to close the session."
            )
            continue

        try:
            response = await execute_travel_query(
                assistant=assistant,
                user_query=user_input,
                default_airport=active_airport,
                allow_interactive_followup=True,
            )
            if output_json:
                print(response.model_dump_json(indent=2))
            else:
                print(f"\n{format_proposal_response(response)}")
        except Exception as exc:
            logger.exception(f"Error handling travel query: {exc}")
            print(f"\n[Error] Failed to process your request: {exc}")


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for single-query or interactive assistant execution."""
    parser = argparse.ArgumentParser(
        description="WeRoad Agentic Travel Assistant CLI (Tours, Departure Dates & Skyscanner Flights)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Single travel query to execute (if omitted, starts interactive chat mode).",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive multi-turn CLI chat mode.",
    )
    parser.add_argument(
        "-a",
        "--airport",
        type=str,
        default=None,
        help="Optional default departure airport name or IATA code (e.g., 'MXP', 'Rome Fiumicino').",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to config/config_weroad.yaml or config.yaml).",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Optional override path to mcp_config.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional PydanticAI model override (e.g., 'openrouter:openai/gpt-4o-mini').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the structured `TravelProposalResponse` as formatted JSON.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )
    return parser.parse_args()


async def async_main() -> int:
    """Initializes configuration and runs either single-query or interactive mode."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config_path = resolve_config_path(args.config)
        config_manager = ConfigManager.get_instance(config_path=config_path)
        setup_observability(
            config=config_manager.observability_config,
            service_name_override="weroad-travel-assistant-cli",
        )
        setup_logging(args.log_level)

        # Ensure flight_search_config is attached even if an older ConfigManager instance is loaded
        if not hasattr(config_manager, "flight_search_config"):
            config_manager.flight_search_config = config_manager.weroad_config.flight_search

        llm_override: Optional[LLMConfig] = None
        if args.model:
            base_llm = config_manager.llm_default_config
            llm_override = LLMConfig(
                model_name=args.model,
                temperature=base_llm.temperature,
                max_tokens=base_llm.max_tokens,
            )

        assistant = WeRoadTravelAssistant(
            config_manager=config_manager,
            llm_config=llm_override,
            mcp_config_path=args.mcp_config,
        )

        # Single-query mode when --query is supplied and --interactive is not forced
        if args.query and not args.interactive:
            response = await execute_travel_query(
                assistant=assistant,
                user_query=args.query,
                default_airport=args.airport,
                allow_interactive_followup=not args.json,
            )
            if args.json:
                print(response.model_dump_json(indent=2))
            else:
                print(format_proposal_response(response))
            return 1 if response.guardrail_blocked else 0

        # Interactive chat mode
        await run_interactive_session(
            assistant=assistant,
            default_airport=args.airport,
            output_json=args.json,
        )
        return 0

    except FileNotFoundError as exc:
        logger.error(f"Configuration file error: {exc}")
        return 2
    except Exception as exc:
        logger.exception(f"Fatal startup/execution error: {exc}")
        return 1


def main() -> None:
    """Synchronous CLI entrypoint."""
    try:
        exit_code = asyncio.run(async_main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()