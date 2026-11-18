"""
WeRoad MCP Tool Server (`mcp_server/weroad_mcp.py`)
===================================================
Exposes `search_weroad_tours` and `get_tour_departures` as Model Context Protocol
(MCP) tools over stdio using centralized settings from `ConfigManager`.
"""

from __future__ import annotations
import os
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastmcp.exceptions import ToolError

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(REPO_ROOT)

from fastmcp import FastMCP
from ai_assistant.core.config_management import ConfigManager
from ai_assistant.tools.flight_search import FlightSearchClient
from ai_assistant.tools.weroad_parser import WeRoadClient

# Log exclusively to stderr so MCP stdio JSON-RPC messages on stdout stay uncorrupted
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("WeRoadMCP")

_config_manager = ConfigManager.get_instance()
mcp = FastMCP("weroad-tools")
_client = WeRoadClient(config=_config_manager.scraper_config)
_flight_client = FlightSearchClient(config=_config_manager.flight_search_config)


@mcp.tool()
async def search_weroad_tours(
    keywords: List[str],
    min_days: Optional[int] = None,
    max_days: Optional[int] = None,
    max_budget_eur: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search WeRoad.it group itineraries matching Italian destination keywords,
    optional trip duration (in days), and maximum budget (in EUR).
    """
    logger.info(
        f"MCP tool `search_weroad_tours` called with keywords={keywords}, "
        f"min_days={min_days}, max_days={max_days}, max_budget_eur={max_budget_eur}"
    )
    tours = await _client.search_catalog(
        keywords=keywords,
        min_days=min_days,
        max_days=max_days,
        max_budget_eur=max_budget_eur,
    )
    return [tour.model_dump(mode="json") for tour in tours]


@mcp.tool()
async def get_tour_departures(
    slug: str,
    earliest_date: Optional[str] = None,
    latest_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch live scheduled departure dates ('turni'), prices in EUR, age brackets,
    and availability statuses ('Confermato', 'Quasi pieno', 'Disponibile', 'Sold out')
    for a WeRoad tour slug. The slug MUST be copied exactly from the `slug` field
    of a result returned by `search_weroad_tours`; never guess or invent slugs.
    """
    logger.info(...)
    try:
        slots = await _client.fetch_departures(
            slug=slug, earliest_date=earliest_date, latest_date=latest_date,
        )
        return [slot.model_dump(mode="json") for slot in slots]
    except Exception as exc:
        logger.warning(f"get_tour_departures failed for slug='{slug}': {exc}")
        raise ToolError(
            f"Could not fetch departures for slug '{slug}': {exc}. "
            "Call `search_weroad_tours` and use the exact slug it returns, or try another tour."
        ) from exc

@mcp.tool()
async def search_flights(
    origin_airport: str,
    destination: str,
    departure_date: str,
    return_date: str,
    adults: int = 1,
    max_stops: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search Skyscanner round-trip flights from `origin_airport` (city or 3-letter IATA code,
    e.g., 'MXP', 'FCO') to `destination` (city, country, or IATA code, e.g., 'TYO', 'Japan', 'KEF')
    matching the available WeRoad tour `departure_date` and `return_date` (YYYY-MM-DD),
    returning round-trip prices in EUR, flight durations, stops, and Skyscanner links.
    """
    logger.info(
        f"MCP tool `search_flights` (Skyscanner) called: origin='{origin_airport}', "
        f"destination='{destination}', dates=[{departure_date} .. {return_date}], adults={adults}"
    )
    flights = await _flight_client.search_round_trip_flights(
        origin_airport=origin_airport,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        adults=adults,
        max_stops=max_stops,
    )
    return [flight.model_dump(mode="json") for flight in flights]


if __name__ == "__main__":
    mcp.run()