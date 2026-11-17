"""
WeRoad Catalog & Departure Parser (`tools/weroad_parser.py`)
============================================================
Asynchronous HTTP client and HTML/JSON parser for `weroad.it`.
Configured via `ConfigManager` / `WeRoadScraperConfig`.
Handles:
- Indexing `/viaggi/<slug>` URLs from WeRoad XML sitemaps with local TTL caching.
- Fallback discovery via `/destinazioni/<keyword>` hub pages.
- Parsing itinerary metadata and live departure dates ('turni') from embedded
  Next.js hydration JSON (`__NEXT_DATA__`), JSON-LD blocks, and DOM fallbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup

from ai_assistant.schemas.schemas import (
    DepartureSlot,
    DepartureStatus,
    WeRoadScraperConfig,
    WeRoadTourSummary,
)
from ai_assistant.core.config_management import ConfigManager

logger = logging.getLogger("WeRoadParser")

ITALIAN_MONTHS: Dict[str, int] = {
    "gen": 1,
    "gennaio": 1,
    "feb": 2,
    "febbraio": 2,
    "mar": 3,
    "marzo": 3,
    "apr": 4,
    "aprile": 4,
    "mag": 5,
    "maggio": 5,
    "giu": 6,
    "giugno": 6,
    "lug": 7,
    "luglio": 7,
    "ago": 8,
    "agosto": 8,
    "set": 9,
    "settembre": 9,
    "ott": 10,
    "ottobre": 10,
    "nov": 11,
    "novembre": 11,
    "dic": 12,
    "dicembre": 12,
}


class WeRoadClient:
    """Asynchronous scraper and structured parser for WeRoad.it itineraries."""

    def __init__(
        self,
        config: Optional[WeRoadScraperConfig] = None,
        cache_path: Optional[Path] = None,
        timeout: Optional[float] = None,
    ):
        self.config = config or ConfigManager.get_instance().scraper_config
        self.base_url = self.config.base_url.rstrip("/")
        self.sitemap_url = self.config.sitemap_url
        self.cache_ttl_seconds = self.config.cache_ttl_seconds
        self.max_concurrent_requests = self.config.max_concurrent_requests
        self.cache_path = cache_path or Path(self.config.cache_path)
        self.timeout = timeout if timeout is not None else self.config.timeout_seconds

        self._headers = {
            "User-Agent": self.config.user_agent,
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    # -----------------------------------------------------------------------
    # Sitemap Caching & Catalog Discovery
    # -----------------------------------------------------------------------

    def _load_cached_slugs(self) -> Optional[List[str]]:
        """Returns cached `/viaggi/<slug>` entries if within TTL."""
        if not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if time.time() - payload.get("timestamp", 0) < self.cache_ttl_seconds:
                return payload.get("slugs", [])
        except Exception as exc:
            logger.warning(f"Failed to read WeRoad catalog cache: {exc}")
        return None

    def _save_cached_slugs(self, slugs: List[str]) -> None:
        """Persists discovered tour slugs to local JSON cache."""
        try:
            payload = {"timestamp": time.time(), "slugs": sorted(set(slugs))}
            self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to write WeRoad catalog cache: {exc}")

    @staticmethod
    def _extract_viaggi_slugs_from_xml(xml_text: str) -> Tuple[List[str], List[str]]:
        """Extracts `/viaggi/<slug>` identifiers and nested sitemap URLs from XML."""
        tour_slugs: List[str] = []
        child_sitemaps: List[str] = []
        try:
            root = ET.fromstring(xml_text)
            for elem in root.iter():
                tag = elem.tag.split("}")[-1]
                if tag == "loc" and elem.text:
                    loc = elem.text.strip()
                    if loc.endswith(".xml"):
                        child_sitemaps.append(loc)
                    elif "/viaggi/" in loc:
                        slug = loc.rstrip("/").split("/viaggi/")[-1]
                        if slug and "/" not in slug:
                            tour_slugs.append(slug)
        except ET.ParseError as exc:
            logger.debug(f"XML parse error in sitemap: {exc}")
        return tour_slugs, child_sitemaps

    async def get_all_tour_slugs(self, client: httpx.AsyncClient) -> List[str]:
        """Retrieves all active `/viaggi/<slug>` identifiers from cache or WeRoad sitemaps."""
        cached = self._load_cached_slugs()
        if cached:
            return cached

        discovered_slugs: List[str] = []
        try:
            resp = await client.get(self.sitemap_url)
            resp.raise_for_status()
            slugs, child_sitemaps = self._extract_viaggi_slugs_from_xml(resp.text)
            discovered_slugs.extend(slugs)

            relevant_sitemaps = [
                sm for sm in child_sitemaps if any(k in sm.lower() for k in ("viaggi", "tour", "trip", "it"))
            ] or child_sitemaps[:6]

            for sm_url in relevant_sitemaps:
                try:
                    sm_resp = await client.get(sm_url)
                    if sm_resp.status_code == 200:
                        sub_slugs, _ = self._extract_viaggi_slugs_from_xml(sm_resp.text)
                        discovered_slugs.extend(sub_slugs)
                except Exception as sub_exc:
                    logger.debug(f"Skipping child sitemap {sm_url}: {sub_exc}")
        except Exception as exc:
            logger.warning(f"Sitemap index fetch failed ({exc}); using destination fallback.")

        unique_slugs = sorted(set(discovered_slugs))
        if unique_slugs:
            self._save_cached_slugs(unique_slugs)
        return unique_slugs

    async def _discover_slugs_from_destination_page(
        self, client: httpx.AsyncClient, keyword: str
    ) -> List[str]:
        """Fallback: scrapes `/destinazioni/<keyword>` for `/viaggi/<slug>` links."""
        slug_clean = re.sub(r"[^a-z0-9-]", "", keyword.lower().replace(" ", "-"))
        url = f"{self.base_url}/destinazioni/{slug_clean}"
        found: List[str] = []
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "/viaggi/" in href:
                        slug = href.rstrip("/").split("/viaggi/")[-1].split("?")[0]
                        if slug and "/" not in slug:
                            found.append(slug)
        except Exception as exc:
            logger.debug(f"Destination fallback failed for {url}: {exc}")
        return sorted(set(found))

    async def search_catalog(
        self,
        keywords: List[str],
        min_days: Optional[int] = None,
        max_days: Optional[int] = None,
        max_budget_eur: Optional[float] = None,
        limit: int = 8,
    ) -> List[WeRoadTourSummary]:
        """
        Searches WeRoad itineraries matching destination keywords and optional duration/budget filters.
        """
        normalized_kw = [k.lower().strip() for k in keywords if k.strip()]
        if not normalized_kw:
            return []

        async with httpx.AsyncClient(
            headers=self._headers, timeout=self.timeout, follow_redirects=True
        ) as client:
            all_slugs = await self.get_all_tour_slugs(client)
            matched_slugs = [
                slug for slug in all_slugs if any(kw in slug.lower() for kw in normalized_kw)
            ]

            if not matched_slugs:
                for kw in normalized_kw[:2]:
                    matched_slugs.extend(await self._discover_slugs_from_destination_page(client, kw))
                matched_slugs = sorted(set(matched_slugs))

            candidates = matched_slugs[: max(limit * 2, 10)]
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def _fetch_summary(slug: str) -> Optional[WeRoadTourSummary]:
                async with semaphore:
                    try:
                        tour_url = f"{self.base_url}/viaggi/{slug}"
                        resp = await client.get(tour_url)
                        if resp.status_code != 200:
                            return None
                        title, days, price, summary, _ = self.parse_tour_html(resp.text, slug)
                        return WeRoadTourSummary(
                            title=title,
                            slug=slug,
                            url=tour_url,
                            destination_keywords=normalized_kw,
                            duration_days=days,
                            starting_price_eur=price,
                            summary=summary,
                        )
                    except Exception as exc:
                        logger.debug(f"Failed fetching summary for {slug}: {exc}")
                        return None

            raw_results = await asyncio.gather(*[_fetch_summary(s) for s in candidates])
            summaries: List[WeRoadTourSummary] = []

            for item in raw_results:
                if item is None:
                    continue
                if item.duration_days is not None:
                    if min_days is not None and item.duration_days < max(1, min_days - 2):
                        continue
                    if max_days is not None and item.duration_days > max_days + 2:
                        continue
                if (
                    max_budget_eur is not None
                    and item.starting_price_eur is not None
                    and item.starting_price_eur > max_budget_eur * 1.1
                ):
                    continue
                summaries.append(item)

            return summaries[:limit]

    async def fetch_departures(
        self,
        slug: str,
        earliest_date: Optional[str] = None,
        latest_date: Optional[str] = None,
    ) -> List[DepartureSlot]:
        """
        Fetches a WeRoad itinerary page and extracts scheduled departure dates ('turni'),
        optionally bounded by [earliest_date, latest_date] (YYYY-MM-DD).
        """
        clean_slug = slug.rstrip("/").split("/viaggi/")[-1]
        url = f"{self.base_url}/viaggi/{clean_slug}"

        earliest_dt = date.fromisoformat(earliest_date) if earliest_date else None
        latest_dt = date.fromisoformat(latest_date) if latest_date else None

        async with httpx.AsyncClient(
            headers=self._headers, timeout=self.timeout, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            _, _, _, _, slots = self.parse_tour_html(resp.text, clean_slug)

        filtered: List[DepartureSlot] = []
        for slot in slots:
            if earliest_dt and slot.start_date < earliest_dt:
                continue
            if latest_dt and slot.start_date > latest_dt:
                continue
            filtered.append(slot)

        return sorted(filtered, key=lambda s: s.start_date)

    # -----------------------------------------------------------------------
    # HTML & Embedded JSON Parsing Helpers
    # -----------------------------------------------------------------------

    def parse_tour_html(
        self, html: str, slug: str
    ) -> Tuple[str, Optional[int], Optional[float], str, List[DepartureSlot]]:
        """
        Extracts `(title, duration_days, starting_price_eur, summary, departure_slots)`
        from a WeRoad tour HTML page.
        """
        soup = BeautifulSoup(html, "html.parser")

        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else slug.replace("-", " ").title()
        meta_desc = soup.find("meta", attrs={"name": "description"})
        summary = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else ""

        duration_days: Optional[int] = None
        full_text = soup.get_text(" ", strip=True)
        dur_match = re.search(r"\b(\d{1,2})\s*giorni\b", full_text, flags=re.IGNORECASE)
        if dur_match:
            duration_days = int(dur_match.group(1))

        slots: List[DepartureSlot] = []
        starting_price: Optional[float] = None

        next_data_tag = soup.find("script", id="__NEXT_DATA__")
        if next_data_tag and next_data_tag.string:
            try:
                next_json = json.loads(next_data_tag.string)
                slots.extend(self._extract_slots_from_json_tree(next_json))
            except json.JSONDecodeError:
                pass

        for ld_tag in soup.find_all("script", type="application/ld+json"):
            if not ld_tag.string:
                continue
            try:
                ld_json = json.loads(ld_tag.string)
                slots.extend(self._extract_slots_from_json_tree(ld_json))
                if starting_price is None:
                    starting_price = self._extract_min_price_from_json(ld_json)
            except json.JSONDecodeError:
                continue

        if not slots:
            slots.extend(self._extract_slots_from_dom(soup))

        deduped: Dict[Tuple[date, date, Optional[str]], DepartureSlot] = {}
        for slot in slots:
            key = (slot.start_date, slot.end_date, slot.age_bracket)
            if key not in deduped:
                deduped[key] = slot
        unique_slots = sorted(deduped.values(), key=lambda s: s.start_date)

        if unique_slots:
            starting_price = min(s.price_eur for s in unique_slots)
        elif starting_price is None:
            price_match = re.search(
                r"(?:da|a partire da)?\s*€\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)",
                full_text,
                flags=re.IGNORECASE,
            )
            if price_match:
                starting_price = self._parse_eur_amount(price_match.group(1))

        return title, duration_days, starting_price, summary, unique_slots

    def _extract_slots_from_json_tree(self, node: Any) -> List[DepartureSlot]:
        """Recursively traverses JSON trees to locate departure ('turno') objects."""
        found: List[DepartureSlot] = []

        if isinstance(node, dict):
            start_raw = (
                node.get("startDate")
                or node.get("start_date")
                or node.get("departureDate")
                or node.get("dateFrom")
            )
            end_raw = (
                node.get("endDate")
                or node.get("end_date")
                or node.get("returnDate")
                or node.get("dateTo")
            )
            price_raw = (
                node.get("price")
                or node.get("finalPrice")
                or node.get("discountedPrice")
                or node.get("amount")
            )

            if isinstance(start_raw, str) and isinstance(end_raw, str) and price_raw is not None:
                start_dt = self._parse_iso_date(start_raw)
                end_dt = self._parse_iso_date(end_raw)
                price_val = self._coerce_price(price_raw)
                if start_dt and end_dt and price_val and price_val > 50:
                    orig_raw = node.get("originalPrice") or node.get("fullPrice") or node.get("listPrice")
                    orig_val = self._coerce_price(orig_raw) if orig_raw is not None else None
                    status_raw = str(
                        node.get("status")
                        or node.get("availability")
                        or node.get("label")
                        or "Disponibile"
                    )
                    age_raw = node.get("ageRange") or node.get("ageGroup") or node.get("targetAge")
                    found.append(
                        DepartureSlot(
                            start_date=start_dt,
                            end_date=end_dt,
                            price_eur=price_val,
                            original_price_eur=orig_val if orig_val and orig_val > price_val else None,
                            status=self._normalize_status(status_raw),
                            age_bracket=str(age_raw) if age_raw else None,
                        )
                    )

            for value in node.values():
                found.extend(self._extract_slots_from_json_tree(value))

        elif isinstance(node, list):
            for item in node:
                found.extend(self._extract_slots_from_json_tree(item))

        return found

    def _extract_slots_from_dom(self, soup: BeautifulSoup) -> List[DepartureSlot]:
        """Fallback parser scanning DOM elements for Italian date ranges (e.g. '12 Ott - 22 Ott 2026')."""
        slots: List[DepartureSlot] = []
        current_year = date.today().year

        date_range_re = re.compile(
            r"\b(\d{1,2})\s+([a-zA-Z]{3,9})(?:\s+(\d{4}))?\s*[-–]\s*"
            r"(\d{1,2})\s+([a-zA-Z]{3,9})(?:\s+(\d{4}))?\b"
        )

        for container in soup.find_all(["li", "div", "article", "tr"]):
            text = container.get_text(" ", strip=True)
            if len(text) > 260 or "€" not in text:
                continue

            m = date_range_re.search(text)
            if not m:
                continue

            d1, m1_str, y1_str, d2, m2_str, y2_str = m.groups()
            m1 = ITALIAN_MONTHS.get(m1_str.lower())
            m2 = ITALIAN_MONTHS.get(m2_str.lower())
            if not m1 or not m2:
                continue

            end_year = int(y2_str) if y2_str else (int(y1_str) if y1_str else current_year)
            start_year = int(y1_str) if y1_str else (end_year - 1 if m1 > m2 else end_year)

            try:
                start_dt = date(start_year, m1, int(d1))
                end_dt = date(end_year, m2, int(d2))
            except ValueError:
                continue

            price_matches = re.findall(r"€\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)", text)
            if not price_matches:
                continue
            prices = [p for raw in price_matches if (p := self._parse_eur_amount(raw)) is not None]
            if not prices:
                continue

            final_price = min(prices)
            orig_price = max(prices) if len(prices) > 1 and max(prices) > final_price else None

            age_match = re.search(r"\b(18-24|25-35|35-49|25-49|Tutti)\b", text, flags=re.IGNORECASE)
            age_bracket = age_match.group(1) if age_match else None

            slots.append(
                DepartureSlot(
                    start_date=start_dt,
                    end_date=end_dt,
                    price_eur=final_price,
                    original_price_eur=orig_price,
                    status=self._normalize_status(text),
                    age_bracket=age_bracket,
                )
            )

        return slots

    @staticmethod
    def _parse_iso_date(raw: str) -> Optional[date]:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(raw[:10])
            except ValueError:
                return None

    @staticmethod
    def _parse_eur_amount(raw: str) -> Optional[float]:
        cleaned = raw.replace(".", "").replace(",", ".").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _coerce_price(self, raw: Any) -> Optional[float]:
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            return self._parse_eur_amount(raw.replace("€", ""))
        if isinstance(raw, dict):
            for key in ("amount", "value", "price", "lowPrice"):
                if key in raw:
                    return self._coerce_price(raw[key])
        return None

    def _extract_min_price_from_json(self, node: Any) -> Optional[float]:
        if isinstance(node, dict):
            offers = node.get("offers")
            if isinstance(offers, dict):
                return self._coerce_price(offers.get("lowPrice") or offers.get("price"))
        return None

    @staticmethod
    def _normalize_status(raw_text: str) -> DepartureStatus:
        lower = raw_text.lower()
        if any(k in lower for k in ("sold out", "esaurito", "completo", "non disponibile")):
            return DepartureStatus.SOLD_OUT
        if any(k in lower for k in ("quasi pieno", "ultimi posti", "almost full", "few spots")):
            return DepartureStatus.ALMOST_FULL
        if any(k in lower for k in ("confermato", "confirmed")):
            return DepartureStatus.CONFIRMED
        if any(k in lower for k in ("disponibile", "available", "aperto")):
            return DepartureStatus.AVAILABLE
        return DepartureStatus.UNKNOWN