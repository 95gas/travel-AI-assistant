"""
Streamlit Client Interface (`streamlit_app.py`)
===============================================
Modern conversational web UI that communicates with the FastAPI backend (`server.py`)
to search WeRoad.it group itineraries, live departure dates, and round-trip
Skyscanner flight costs, durations, and booking links.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("WEROAD_API_URL", "http://localhost:8000")
ENABLE_OBSERVABILITY_UI = os.getenv("ENABLE_OBSERVABILITY_UI", "false").lower() == "true"

# WeRoad.it Brand Palette
WEROAD_RED = "#FF4758"        # Radical Red (Primary brand color)
WEROAD_RED_DARK = "#D80027"   # Monza (Hover / active state)
WEROAD_DARK = "#1A1A1A"       # Cod Gray (Headings & primary ink)
WEROAD_MUTED = "#666670"      # Secondary muted text
WEROAD_BG = "#FFFFFF"         # Pure white canvas
WEROAD_SURFACE = "#F7F7F9"    # Warm light surface for bubbles & cards
WEROAD_BORDER = "#EBEBF0"     # Subtle hairline border

POPULAR_AIRPORTS: Dict[str, Optional[str]] = {
    "Any / Specify in chat": None,
    "Milan Malpensa (MXP)": "MXP",
    "Milan Linate (LIN)": "LIN",
    "Bergamo Orio al Serio (BGY)": "BGY",
    "Rome Fiumicino (FCO)": "FCO",
    "Venice Marco Polo (VCE)": "VCE",
    "Bologna (BLQ)": "BLQ",
    "Naples (NAP)": "NAP",
    "Turin (TRN)": "TRN",
    "Florence (FLR)": "FLR",
    "Catania (CTA)": "CTA",
    "Palermo (PMO)": "PMO",
}

STARTER_PROMPTS: List[Dict[str, str]] = [
    {
        "tag": "🇯🇵 Asia • 10 Days",
        "title": "Japan in October from Milan",
        "prompt": "I want to visit Japan in October for 10 days leaving from Milan Malpensa. I'm 29.",
    },
    {
        "tag": "🇮🇸 Europe • August",
        "title": "Islanda ad agosto da Roma",
        "prompt": "Cerco un viaggio WeRoad in Islanda ad agosto da Roma Fiumicino",
    },
    {
        "tag": "🇲🇦 Adventure • Under €900",
        "title": "Morocco in November from Bologna",
        "prompt": "Morocco in November for 8 days under €900 from Bologna",
    },
    {
        "tag": "🇵🇪 South America • 12 Days",
        "title": "Peru expedition in September",
        "prompt": "Peru in September for 12 days",
    },
]


def apply_modern_weroad_theme() -> None:
    """Injects custom CSS for a sidebar-free, modern conversational UI in WeRoad's palette."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Hide sidebar & sidebar toggle completely */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        /* Global typography & light canvas */
        html, body, [class*="css"], .stApp {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {WEROAD_DARK};
            background-color: {WEROAD_BG};
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: {WEROAD_BG};
        }}

        /* Minimalist Streamlit top chrome */
        [data-testid="stHeader"] {{
            background-color: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
        }}

        /* Centered modern chat column */
        .main .block-container {{
            max-width: 840px !important;
            padding-top: 1.25rem !important;
            padding-bottom: 6.5rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }}

        /* Bold geometric headings */
        h1, h2, h3, h4 {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 800 !important;
            color: {WEROAD_DARK} !important;
            letter-spacing: -0.025em;
        }}

        /* Top Navigation Bar */
        .weroad-nav {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0 14px 0;
            margin-bottom: 12px;
            border-bottom: 1px solid {WEROAD_BORDER};
        }}
        .weroad-nav-left {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .weroad-wordmark {{
            font-size: 1.35rem;
            font-weight: 800;
            color: {WEROAD_RED};
            letter-spacing: -0.03em;
            line-height: 1;
        }}
        .weroad-wordmark span {{
            color: {WEROAD_DARK};
        }}
        .weroad-nav-badge {{
            background-color: {WEROAD_SURFACE};
            color: {WEROAD_MUTED};
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid {WEROAD_BORDER};
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 999px;
        }}
        .status-online {{
            background-color: #ECFDF3;
            color: #027A48;
            border: 1px solid #ABEFC6;
        }}
        .status-offline {{
            background-color: #FEF3F2;
            color: #B42318;
            border: 1px solid #FECDCA;
        }}
        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }}

        /* Empty-state Hero Section */
        .weroad-hero {{
            text-align: center;
            padding: 2.5rem 1rem 1.75rem 1rem;
        }}
        .weroad-hero-pill {{
            display: inline-block;
            background-color: #FFF1F2;
            color: {WEROAD_RED};
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            padding: 6px 14px;
            border-radius: 999px;
            margin-bottom: 14px;
        }}
        .weroad-hero-title {{
            font-size: 2.15rem;
            font-weight: 800;
            color: {WEROAD_DARK};
            letter-spacing: -0.03em;
            line-height: 1.18;
            margin: 0 0 10px 0;
        }}
        .weroad-hero-sub {{
            font-size: 1rem;
            color: {WEROAD_MUTED};
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.55;
        }}

        /* Primary CTA & Link Buttons -> WeRoad Radical Red */
        .stLinkButton > a,
        [data-testid="stBaseButton-primary"] {{
            background-color: {WEROAD_RED} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            font-size: 0.85rem !important;
            padding: 0.45rem 1.1rem !important;
            transition: background-color 0.18s ease, transform 0.12s ease, box-shadow 0.18s ease;
            box-shadow: 0 4px 12px rgba(255, 71, 88, 0.24);
        }}

        .stLinkButton > a:hover,
        [data-testid="stBaseButton-primary"]:hover {{
            background-color: {WEROAD_RED_DARK} !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(216, 0, 39, 0.30);
        }}

        /* Starter Prompt Cards & Secondary Pill Buttons */
        .stButton > button,
        [data-testid="stBaseButton-secondary"] {{
            background-color: #FFFFFF !important;
            color: {WEROAD_DARK} !important;
            border: 1px solid {WEROAD_BORDER} !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 0.65rem 0.95rem !important;
            transition: all 0.18s ease;
            box-shadow: 0 2px 6px rgba(26, 26, 26, 0.03);
        }}

        .stButton > button:hover,
        [data-testid="stBaseButton-secondary"]:hover {{
            border-color: {WEROAD_RED} !important;
            color: {WEROAD_RED} !important;
            background-color: #FFF8F8 !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(255, 71, 88, 0.10);
        }}

        /* Popover button styling */
        [data-testid="stPopover"] > button {{
            border-radius: 999px !important;
            border: 1px solid {WEROAD_BORDER} !important;
            background-color: {WEROAD_SURFACE} !important;
            color: {WEROAD_DARK} !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            padding: 0.3rem 0.85rem !important;
        }}
        [data-testid="stPopover"] > button:hover {{
            border-color: {WEROAD_RED} !important;
            color: {WEROAD_RED} !important;
        }}

        /* Modern Chat Bubbles */
        [data-testid="stChatMessage"] {{
            background-color: transparent !important;
            border: none !important;
            padding: 0.5rem 0 !important;
            margin-bottom: 0.35rem !important;
            gap: 0.85rem !important;
        }}

        /* Assistant Message Bubble */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {{
            background-color: {WEROAD_SURFACE};
            border: 1px solid {WEROAD_BORDER};
            border-radius: 4px 20px 20px 20px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 8px rgba(26, 26, 26, 0.03);
        }}

        /* User Message Bubble (Right-aligned sleek pill) */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
            flex-direction: row-reverse !important;
        }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
            background-color: {WEROAD_DARK};
            color: #FFFFFF !important;
            border-radius: 20px 4px 20px 20px;
            padding: 0.8rem 1.15rem;
            max-width: 78%;
            margin-left: auto;
            box-shadow: 0 4px 12px rgba(26, 26, 26, 0.10);
        }}
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {{
            color: #FFFFFF !important;
            margin: 0 !important;
        }}

        /* Avatar Badges */
        [data-testid="stChatMessageAvatarAssistant"] {{
            background-color: {WEROAD_RED} !important;
            color: #FFFFFF !important;
            border-radius: 50% !important;
        }}
        [data-testid="stChatMessageAvatarUser"] {{
            background-color: #FFF1F2 !important;
            color: {WEROAD_RED} !important;
            border: 1px solid #FFD6DA !important;
            border-radius: 50% !important;
        }}

        /* Tour Cards (Expanders) */
        [data-testid="stExpander"] {{
            background-color: #FFFFFF;
            border: 1px solid {WEROAD_BORDER} !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 14px rgba(26, 26, 26, 0.04);
            margin-top: 10px;
            margin-bottom: 10px;
            overflow: hidden;
        }}

        [data-testid="stExpander"] summary {{
            font-weight: 700 !important;
            color: {WEROAD_DARK} !important;
            padding: 0.75rem 1rem !important;
        }}

        [data-testid="stExpander"] summary:hover {{
            color: {WEROAD_RED} !important;
        }}

        /* Floating Pill Chat Input */
        [data-testid="stChatInput"] {{
            max-width: 840px !important;
            margin: 0 auto !important;
        }}
        [data-testid="stChatInput"] > div {{
            border-radius: 999px !important;
            border: 1.5px solid {WEROAD_BORDER} !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 8px 24px rgba(26, 26, 26, 0.08) !important;
            padding-left: 8px !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        [data-testid="stChatInput"] > div:focus-within {{
            border-color: {WEROAD_RED} !important;
            box-shadow: 0 8px 28px rgba(255, 71, 88, 0.16) !important;
        }}
        [data-testid="stChatInputSubmitButton"] {{
            color: {WEROAD_RED} !important;
        }}

        /* Metrics & Dividers */
        [data-testid="stMetricValue"] {{
            color: {WEROAD_RED} !important;
            font-weight: 800 !important;
        }}
        hr {{
            border-color: {WEROAD_BORDER} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initializes Streamlit session state variables across reruns."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = None
    if "selected_airport_label" not in st.session_state:
        st.session_state.selected_airport_label = "Any / Specify in chat"
    if "custom_airport" not in st.session_state:
        st.session_state.custom_airport = ""


@st.cache_data(ttl=15, show_spinner=False)
def check_backend_health(api_url: str) -> bool:
    """Checks `GET /health` on the FastAPI backend with short caching."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{api_url.rstrip('/')}/health")
            return resp.status_code == 200
    except Exception:
        return False


def send_chat_request(
    api_url: str,
    query: str,
    departure_airport: Optional[str],
    session_id: str,
) -> Dict[str, Any]:
    """Sends a travel query to `POST /api/v1/chat` and returns the JSON payload."""
    payload = {
        "query": query,
        "departure_airport": departure_airport,
        "session_id": session_id,
    }
    timeout = httpx.Timeout(600.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{api_url.rstrip('/')}/api/v1/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


def render_top_bar(api_url: str) -> Optional[str]:
    """Renders a compact, modern top header with airport selector and reset button."""
    is_online = check_backend_health(api_url)
    status_cls = "status-online" if is_online else "status-offline"
    dot_color = "#12B76A" if is_online else "#F04438"
    status_label = "Live" if is_online else "API Offline"

    col_brand, col_airport, col_reset = st.columns([5.2, 2.5, 1.3], vertical_alignment="center")

    with col_brand:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;padding:4px 0;">
                <span class="weroad-wordmark">WeRoad<span>.it</span></span>
                <span class="weroad-nav-badge">AI Trip & Flight Finder</span>
                <span class="status-pill {status_cls}">
                    <span class="status-dot" style="background:{dot_color};"></span>
                    {status_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_airport:
        current_custom = st.session_state.custom_airport.strip()
        current_label = st.session_state.selected_airport_label
        airport_btn_text = (
            f"✈️ {current_custom}"
            if current_custom
            else (
                f"✈️ {POPULAR_AIRPORTS[current_label]}"
                if POPULAR_AIRPORTS.get(current_label)
                else "✈️ Origin Airport"
            )
        )

        if hasattr(st, "popover"):
            with st.popover(airport_btn_text, use_container_width=True):
                st.caption("Default Departure Airport")
                st.session_state.selected_airport_label = st.selectbox(
                    "Popular Italian Airports",
                    options=list(POPULAR_AIRPORTS.keys()),
                    index=list(POPULAR_AIRPORTS.keys()).index(st.session_state.selected_airport_label),
                    label_visibility="collapsed",
                )
                st.session_state.custom_airport = st.text_input(
                    "Or type custom IATA / city",
                    value=st.session_state.custom_airport,
                    placeholder="e.g., VRN, Pisa, London",
                )
        else:
            st.session_state.selected_airport_label = st.selectbox(
                "Origin Airport",
                options=list(POPULAR_AIRPORTS.keys()),
                index=list(POPULAR_AIRPORTS.keys()).index(st.session_state.selected_airport_label),
                label_visibility="collapsed",
            )

    with col_reset:
        if st.button("↺ New", use_container_width=True, help="Start a new conversation"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.session_state.queued_prompt = None
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    st.markdown(
        f'<hr style="margin: 6px 0 14px 0; border: none; border-top: 1px solid {WEROAD_BORDER};">',
        unsafe_allow_html=True,
    )

    custom_val = st.session_state.custom_airport.strip()
    if custom_val:
        return custom_val
    return POPULAR_AIRPORTS.get(st.session_state.selected_airport_label)


def render_welcome_hero() -> None:
    """Renders a clean centered greeting and 2x2 starter prompt cards when chat is empty."""
    st.markdown(
        """
        <div class="weroad-hero">
            <span class="weroad-hero-pill">Group Adventures + Live Flights</span>
            <h1 class="weroad-hero-title">Where do you want to travel next?</h1>
            <p class="weroad-hero-sub">
                Tell me your dream destination, travel month, duration, and departure airport—I'll match
                you with real-time <strong>WeRoad</strong> departures and <strong>Skyscanner</strong> round-trip flights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    for idx, item in enumerate(STARTER_PROMPTS):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            button_label = f"{item['tag']}  ·  {item['title']}\n\n{item['prompt']}"
            if st.button(button_label, key=f"starter_{idx}", use_container_width=True):
                st.session_state.queued_prompt = item["prompt"]
                st.rerun()


def render_proposal_payload(payload: Dict[str, Any]) -> None:
    """Renders structured WeRoad tour proposals, departure dates, and Skyscanner flights."""
    # 1. Llama Guard safety block banner
    if payload.get("guardrail_blocked"):
        violations = payload.get("guardrail_violations") or []
        violation_str = f" ({', '.join(violations)})" if violations else ""
        st.error(
            f"**Safety Guardrail Triggered{violation_str}**\n\n"
            f"{payload.get('assistant_message', 'Request blocked by safety policies.')}"
        )
        return

    # 2. Departure airport prompt highlight
    if payload.get("needs_departure_airport"):
        st.info(payload.get("assistant_message", "Please specify your departure airport."))
        return

    # 3. Main assistant response
    st.markdown(payload.get("assistant_message", ""))

    # 4. Recommended WeRoad tours + departure dates + Skyscanner flights
    proposals = payload.get("proposals") or []
    if proposals:
        st.markdown("#### Recommended WeRoad Tours & Flights")

        for idx, tour in enumerate(proposals, start=1):
            duration = tour.get("duration_days")
            start_price = tour.get("starting_price_eur")
            meta_badges = []
            if duration:
                meta_badges.append(f"{duration} days")
            if start_price is not None:
                meta_badges.append(f"from €{start_price:,.2f}")
            header_suffix = f" — {' · '.join(meta_badges)}" if meta_badges else ""

            with st.expander(f"{idx}. {tour.get('tour_title', 'WeRoad Tour')}{header_suffix}", expanded=True):
                col_info, col_link = st.columns([3.8, 1.2], vertical_alignment="center")
                with col_info:
                    st.markdown(f"**Why it fits:** {tour.get('why_it_fits', '')}")
                with col_link:
                    if tour.get("tour_url"):
                        st.link_button("View Tour ↗", tour["tour_url"], use_container_width=True)

                # Available WeRoad departure dates ('turni')
                matching_dates = tour.get("matching_dates") or []
                if matching_dates:
                    st.markdown("**Available Departure Dates**")
                    date_rows = []
                    for slot in matching_dates:
                        orig = slot.get("original_price_eur")
                        price_display = f"€{slot['price_eur']:,.2f}"
                        if orig:
                            price_display += f" (was €{orig:,.2f})"
                        date_rows.append(
                            {
                                "Start": slot.get("start_date"),
                                "End": slot.get("end_date"),
                                "Price": price_display,
                                "Status": slot.get("status", "Disponibile"),
                                "Age Group": slot.get("age_bracket") or "Tutti",
                            }
                        )
                    st.dataframe(date_rows, use_container_width=True, hide_index=True)

                # Matching Skyscanner round-trip flights
                matching_flights = tour.get("matching_flights") or []
                if matching_flights:
                    st.markdown("**Round-Trip Flights (Skyscanner)**")
                    for flight in matching_flights:
                        f_col1, f_col2, f_col3, f_col4 = st.columns(
                            [2.2, 1.5, 1.5, 1.3], vertical_alignment="center"
                        )
                        route_str = (
                            f"**{flight.get('origin_airport')} → {flight.get('destination_airport')}**\n\n"
                            f"{flight.get('departure_date')} – {flight.get('return_date')}"
                        )
                        price_val = flight.get("price_eur")
                        if price_val is not None:
                            est_tag = " *(est.)*" if flight.get("is_estimated_price") else ""
                            cost_str = f"**€{price_val:,.2f}**{est_tag}\n\n{flight.get('airline', 'Skyscanner')}"
                        else:
                            cost_str = f"Live fare\n\n{flight.get('airline', 'Skyscanner')}"

                        stops = flight.get("stops")
                        stops_str = (
                            "Direct"
                            if stops == 0
                            else (f"{stops} stop{'s' if stops and stops > 1 else ''}" if stops is not None else "—")
                        )
                        dur_str = f"⏱ {flight.get('duration') or 'N/A'}\n\n{stops_str}"

                        with f_col1:
                            st.markdown(route_str)
                        with f_col2:
                            st.markdown(cost_str)
                        with f_col3:
                            st.markdown(dur_str)
                        with f_col4:
                            if flight.get("booking_url"):
                                st.link_button(
                                    "Flights ↗",
                                    flight["booking_url"],
                                    use_container_width=True,
                                )

    # 5. Alternative suggestions
    if payload.get("alternative_suggestions"):
        st.info(f"**Tip:** {payload['alternative_suggestions']}")

    # 6. Optional Observability Telemetry (hidden unless ENABLE_OBSERVABILITY_UI=true)
    telemetry = payload.get("telemetry")
    if (
        ENABLE_OBSERVABILITY_UI
        and isinstance(telemetry, dict)
        and (telemetry.get("total_tokens") or telemetry.get("trace_id"))
    ):
        with st.expander("Observability: Token Usage & Tempo Trace", expanded=False):
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("Input Tokens", f"{telemetry.get('input_tokens', 0):,}")
            t_col2.metric("Output Tokens", f"{telemetry.get('output_tokens', 0):,}")
            t_col3.metric("Total Tokens", f"{telemetry.get('total_tokens', 0):,}")
            t_col4.metric("LLM Calls", f"{telemetry.get('llm_requests', 0)}")

            trace_id = telemetry.get("trace_id")
            grafana_url = telemetry.get("grafana_trace_url")
            if trace_id:
                st.caption(f"OpenTelemetry Trace ID: `{trace_id}`")
            if grafana_url:
                st.link_button("Inspect Trace in Grafana Tempo ↗", grafana_url)


def main() -> None:
    """Entry point for the WeRoad & Skyscanner modern chatbot interface."""
    st.set_page_config(
        page_title="WeRoad AI Travel Assistant",
        page_icon="🌍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    apply_modern_weroad_theme()
    init_session_state()

    api_url = DEFAULT_API_URL
    active_airport = render_top_bar(api_url)

    # Show empty-state welcome hero when conversation hasn't started yet
    if not st.session_state.messages:
        render_welcome_hero()

    # Render conversation history
    for msg in st.session_state.messages:
        avatar = "🧭" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and isinstance(msg.get("payload"), dict):
                render_proposal_payload(msg["payload"])
            else:
                st.markdown(msg["content"])

    # Accept input from either bottom pill chat box or starter prompt cards
    chat_placeholder = (
        "Enter your departure airport (e.g., Milan Malpensa, FCO, Venice)..."
        if st.session_state.pending_query
        else "Ask about destinations, dates, budget, or flights..."
    )
    user_input = st.chat_input(chat_placeholder)
    if not user_input and st.session_state.queued_prompt:
        user_input = st.session_state.queued_prompt
        st.session_state.queued_prompt = None

    if user_input:
        # Display and store user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Combine with pending query if user is answering the missing departure airport prompt
        if st.session_state.pending_query:
            combined_query = f"{st.session_state.pending_query} (departing from {user_input.strip()})"
            airport_for_request = None
        else:
            combined_query = user_input
            airport_for_request = active_airport

        with st.chat_message("assistant", avatar="🧭"):
            with st.spinner("Finding matching WeRoad itineraries and Skyscanner flights..."):
                try:
                    response_payload = send_chat_request(
                        api_url=api_url,
                        query=combined_query,
                        departure_airport=airport_for_request,
                        session_id=st.session_state.session_id,
                    )

                    if response_payload.get("needs_departure_airport"):
                        st.session_state.pending_query = user_input
                    else:
                        st.session_state.pending_query = None

                    render_proposal_payload(response_payload)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_payload.get("assistant_message", ""),
                            "payload": response_payload,
                        }
                    )
                except httpx.HTTPStatusError as http_err:
                    err_detail = http_err.response.text
                    st.error(f"Backend API error ({http_err.response.status_code}): {err_detail}")
                except httpx.RequestError as conn_err:
                    st.error(
                        f"Could not reach the FastAPI server at `{api_url}` ({conn_err}). "
                        "Make sure `uvicorn server:app --reload` is running."
                    )
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()