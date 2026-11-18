"""
Two-Stage Agentic Travel Orchestrator (`orchestrator.py`)
=========================================================
Coordinates input/output Llama Guard moderation and two specialized instances
of `GenericAgent` configured via `ConfigManager`:
0. Input Guardrail: Moderates the incoming user prompt via `LlamaGuardModerator`.
1. `intent_agent`: Extracts a structured `TravelSearchIntent` from free-form text.
2. `proposal_agent`: Uses WeRoad MCP tools (`mcp_config.json`) to search tours,
   inspect live departure dates, and synthesize a `TravelProposalResponse`.
3. Output Guardrail: Moderates the synthesized response before returning it.

Supports both standard request-response (`handle_query`) and real-time Server-Sent
Events streaming (`stream_query`) of intermediate reasoning and tool invocations.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import AsyncIterator, Dict, Optional, Union

from src.ai_assistant.core.agent import GenericAgent
from src.ai_assistant.schemas.schemas import (
    AgentStreamEvent,
    LLMConfig,
    StreamStepType,
    TravelProposalResponse,
    TravelSearchIntent,
)
from src.ai_assistant.core.config_management import ConfigManager
from src.ai_assistant.core.guardrail import LlamaGuardModerator
from src.ai_assistant.observability.observability import build_token_usage_summary, get_tracer

logger = logging.getLogger("WeRoadOrchestrator")
_tracer = get_tracer("weroad.orchestrator")


class WeRoadTravelAssistant:
    """Two-stage PydanticAI travel assistant with centralized config, I/O guardrails, and live step streaming."""

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        llm_config: Optional[LLMConfig] = None,
        mcp_config_path: Optional[Union[str, Path]] = None,
    ):
        self.config_manager = config_manager or ConfigManager.get_instance()
        self.weroad_cfg = self.config_manager.weroad_config

        intent_llm = llm_config or self.config_manager.llm_intent_config
        proposal_llm = llm_config or self.config_manager.llm_proposal_config

        resolved_mcp_path = (
            Path(mcp_config_path)
            if mcp_config_path is not None
            else self.config_manager.mcp_config_path
        )
        if not resolved_mcp_path.exists():
            raise FileNotFoundError(
                f"MCP config file not found at '{resolved_mcp_path.resolve()}'. "
                "Ensure `mcp_config.json` is present or update `schemas.mcp_config_path` in YAML."
            )
        self.mcp_config_path = resolved_mcp_path

        today_str = date.today().isoformat()
        prompts = self.config_manager.prompts_config

        intent_system_prompt = prompts.intent_system_prompt.format(today_str=today_str)
        proposal_system_prompt = prompts.proposal_system_prompt.format(today_str=today_str)

        # Input & Output Safety Guardrail
        self.guardrail = LlamaGuardModerator(config=self.config_manager.llama_guard_config)

        # Stage 1: Natural-language query -> TravelSearchIntent
        self.intent_agent = GenericAgent(
            llm_settings=intent_llm,
            system_prompt=intent_system_prompt,
            output_type=TravelSearchIntent,
            request_limit=self.config_manager.pipeline_config.intent_request_limit,
            stage_name="intent_extraction",
        )

        # Stage 2: Tool-equipped search & date verification -> TravelProposalResponse
        self.proposal_agent = GenericAgent(
            llm_settings=proposal_llm,
            system_prompt=proposal_system_prompt,
            output_type=TravelProposalResponse,
            mcp_config_path=self.mcp_config_path,
            request_limit=self.config_manager.pipeline_config.proposal_request_limit,
            stage_name="proposal_synthesis",
        )

    @staticmethod
    def _serialize_proposal_for_moderation(proposal: TravelProposalResponse) -> str:
        """Builds a plain-text representation of the final response for output moderation."""
        sections = [proposal.assistant_message]
        for idx, item in enumerate(proposal.proposals, start=1):
            sections.append(f"{idx}. {item.tour_title} ({item.tour_url}): {item.why_it_fits}")
            for flight in item.matching_flights:
                if flight.price_eur is not None:
                    est_tag = " est." if flight.is_estimated_price else ""
                    price_str = f"€{flight.price_eur:.2f}{est_tag}"
                else:
                    price_str = "live fare on Skyscanner"
                dur_str = f", {flight.duration}" if flight.duration else ""
                sections.append(
                    f"   Skyscanner {flight.origin_airport}->{flight.destination_airport} "
                    f"({flight.departure_date} to {flight.return_date}): {flight.airline} "
                    f"({price_str}{dur_str}) - {flight.booking_url}"
                )
        if proposal.alternative_suggestions:
            sections.append(f"Alternative suggestions: {proposal.alternative_suggestions}")
        return "\n".join(sections)

    @staticmethod
    def _summarize_intent_reasoning(intent: TravelSearchIntent) -> str:
        """Formats extracted `TravelSearchIntent` fields into a clear intermediate reasoning explanation."""
        parts = [f"Destination: **{intent.destination}**"]
        if intent.destination_slug_keywords:
            parts.append(f"WeRoad keywords: `{', '.join(intent.destination_slug_keywords)}`")

        if intent.departure_airport or intent.departure_airport_iata:
            origin_disp = (
                f"{intent.departure_airport} ({intent.departure_airport_iata})"
                if intent.departure_airport and intent.departure_airport_iata
                else (intent.departure_airport or intent.departure_airport_iata)
            )
            parts.append(f"Departure airport: **{origin_disp}**")
        else:
            parts.append("Departure airport: *not specified*")

        if intent.earliest_departure and intent.latest_departure:
            parts.append(f"Travel window: {intent.earliest_departure} to {intent.latest_departure}")
        elif intent.earliest_departure:
            parts.append(f"Departing from: {intent.earliest_departure}")
        elif intent.latest_departure:
            parts.append(f"Departing before: {intent.latest_departure}")
        elif intent.preferred_months:
            months_str = ", ".join(str(m) for m in intent.preferred_months)
            parts.append(f"Preferred months: {months_str}")

        if intent.min_days and intent.max_days:
            parts.append(f"Duration: {intent.min_days}–{intent.max_days} days")
        elif intent.min_days:
            parts.append(f"Duration: ≥ {intent.min_days} days")
        elif intent.max_days:
            parts.append(f"Duration: ≤ {intent.max_days} days")

        if intent.max_budget_eur:
            parts.append(f"Max budget: €{intent.max_budget_eur:,.0f}")
        if intent.traveler_age:
            parts.append(f"Traveler age: {intent.traveler_age} yrs")
        if intent.travel_style:
            parts.append(f"Style: {intent.travel_style}")

        return "Extracted search parameters → " + " · ".join(parts) + "."

    async def handle_query(self, user_query: str) -> TravelProposalResponse:
        """Runs the guarded two-stage agentic pipeline and returns the final `TravelProposalResponse`."""
        final_response: Optional[TravelProposalResponse] = None
        async for event in self.stream_query(user_query):
            if event.step_type == StreamStepType.FINAL and event.response is not None:
                final_response = event.response

        if final_response is None:
            raise RuntimeError("Orchestrator stream ended without producing a final TravelProposalResponse.")
        return final_response

    async def stream_query(self, user_query: str) -> AsyncIterator[AgentStreamEvent]:
        """
        Executes the guarded two-stage agentic pipeline while yielding real-time
        `AgentStreamEvent` items for every safety check, intent reasoning step,
        WeRoad catalog/departure search, and Skyscanner flight search.
        """
        stage_usages: Dict[str, Dict[str, int]] = {}
        obs_cfg = self.config_manager.observability_config
        pipeline_start = time.perf_counter()

        with _tracer.start_as_current_span("weroad.handle_query") as root_span:
            if root_span is not None and hasattr(root_span, "set_attribute") and obs_cfg.capture_prompts:
                root_span.set_attribute("weroad.user_query", user_query[:500])

            # ---------------------------------------------------------------
            # Step 0: Input Guardrail
            # ---------------------------------------------------------------
            if self.config_manager.llama_guard_config.enabled and self.config_manager.llama_guard_config.check_input:
                yield AgentStreamEvent(
                    step_type=StreamStepType.STATUS,
                    stage="guardrail_input",
                    title="Safety Policy Check",
                    detail="Verifying request against content safety guidelines...",
                )

            input_safety = await self.guardrail.check_input(user_query)
            if self.guardrail.last_input_usage:
                stage_usages["guardrail_input"] = self.guardrail.last_input_usage

            if not input_safety.is_safe:
                logger.warning(
                    f"Blocking unsafe user input. Categories: {input_safety.violating_categories}"
                )
                telemetry = build_token_usage_summary(stage_usages, config=obs_cfg)
                blocked_resp = TravelProposalResponse(
                    assistant_message=self.config_manager.llama_guard_config.blocked_input_message,
                    proposals=[],
                    guardrail_blocked=True,
                    guardrail_violations=input_safety.violating_categories,
                    telemetry=telemetry,
                )
                yield AgentStreamEvent(
                    step_type=StreamStepType.FINAL,
                    stage="complete",
                    title="Request Blocked by Safety Guardrail",
                    detail="The request triggered a content safety policy.",
                    response=blocked_resp,
                )
                return

            # ---------------------------------------------------------------
            # Stage 1: Intent Extraction
            # ---------------------------------------------------------------
            yield AgentStreamEvent(
                step_type=StreamStepType.STATUS,
                stage="intent_extraction",
                title="Understanding Travel Preferences",
                detail="Parsing destination, departure airport, dates, trip duration, and budget...",
            )

            logger.info("Stage 1: Extracting TravelSearchIntent...")
            intent: TravelSearchIntent = await self.intent_agent.run(user_query)
            stage_usages["intent_extraction"] = dict(self.intent_agent.last_usage)
            logger.info(f"Parsed intent: {intent.model_dump_json()}")

            if root_span is not None and hasattr(root_span, "set_attribute"):
                root_span.set_attribute("weroad.intent.destination", intent.destination)
                if intent.departure_airport_iata:
                    root_span.set_attribute("weroad.intent.origin_iata", intent.departure_airport_iata)

            yield AgentStreamEvent(
                step_type=StreamStepType.REASONING,
                stage="intent_extraction",
                title=f"Trip Intent Parsed: {intent.destination}",
                detail=self._summarize_intent_reasoning(intent),
                metadata={
                    "destination": intent.destination,
                    "origin_iata": intent.departure_airport_iata,
                    "destination_iata": intent.destination_airport_iata,
                },
            )

            # Enforce departure airport presence when required by configuration
            if (
                self.config_manager.pipeline_config.require_departure_airport
                and not intent.departure_airport
                and not intent.departure_airport_iata
            ):
                logger.info("Departure airport missing from user query; prompting user for origin airport.")
                yield AgentStreamEvent(
                    step_type=StreamStepType.REASONING,
                    stage="intent_extraction",
                    title="Departure Airport Needed",
                    detail="No departure airport was detected in the query—asking user for an origin airport so round-trip Skyscanner flights can be matched.",
                )
                missing_msg = self.config_manager.prompts_config.missing_departure_airport_message.strip()
                output_safety = await self.guardrail.check_output(user_query, missing_msg)
                if self.guardrail.last_output_usage:
                    stage_usages["guardrail_output"] = self.guardrail.last_output_usage

                telemetry = build_token_usage_summary(stage_usages, config=obs_cfg)
                if not output_safety.is_safe:
                    blocked_out = TravelProposalResponse(
                        assistant_message=self.config_manager.llama_guard_config.blocked_output_message,
                        proposals=[],
                        guardrail_blocked=True,
                        guardrail_violations=output_safety.violating_categories,
                        telemetry=telemetry,
                    )
                    yield AgentStreamEvent(
                        step_type=StreamStepType.FINAL,
                        stage="complete",
                        title="Response Blocked by Safety Guardrail",
                        detail="Output moderation blocked the response.",
                        response=blocked_out,
                    )
                    return

                prompt_airport_resp = TravelProposalResponse(
                    assistant_message=missing_msg,
                    proposals=[],
                    needs_departure_airport=True,
                    telemetry=telemetry,
                )
                yield AgentStreamEvent(
                    step_type=StreamStepType.FINAL,
                    stage="complete",
                    title="Awaiting Departure Airport",
                    detail="Prompted user to specify departure airport.",
                    response=prompt_airport_resp,
                )
                return

            # ---------------------------------------------------------------
            # Stage 2: Catalog Search, Departure Verification & Flight Search
            # ---------------------------------------------------------------
            logger.info("Stage 2: Streaming WeRoad catalog, departure dates, and matching flights...")
            stage2_prompt = self.config_manager.prompts_config.stage2_user_prompt_template.format(
                user_query=user_query,
                intent_json=intent.model_dump_json(indent=2),
            )

            async for step_event in self.proposal_agent.run_stream(stage2_prompt):
                yield step_event

            proposal: Optional[TravelProposalResponse] = self.proposal_agent.last_output
            if proposal is None:
                raise RuntimeError("Stage 2 proposal agent did not return a TravelProposalResponse.")
            stage_usages["proposal_synthesis"] = dict(self.proposal_agent.last_usage)

            # ---------------------------------------------------------------
            # Step 3: Output Guardrail
            # ---------------------------------------------------------------
            if self.config_manager.llama_guard_config.enabled and self.config_manager.llama_guard_config.check_output:
                yield AgentStreamEvent(
                    step_type=StreamStepType.STATUS,
                    stage="guardrail_output",
                    title="Verifying Final Recommendations",
                    detail="Running final safety & policy check on generated travel proposals...",
                )

            rendered_output = self._serialize_proposal_for_moderation(proposal)
            output_safety = await self.guardrail.check_output(user_query, rendered_output)
            if self.guardrail.last_output_usage:
                stage_usages["guardrail_output"] = self.guardrail.last_output_usage

            telemetry = build_token_usage_summary(stage_usages, config=obs_cfg)
            if root_span is not None and hasattr(root_span, "set_attribute"):
                root_span.set_attribute("weroad.tokens.input", telemetry.input_tokens)
                root_span.set_attribute("weroad.tokens.output", telemetry.output_tokens)
                root_span.set_attribute("weroad.tokens.total", telemetry.total_tokens)
                root_span.set_attribute("weroad.proposals.count", len(proposal.proposals))

            total_elapsed_ms = int((time.perf_counter() - pipeline_start) * 1000)
            logger.info(
                f"Request complete in {total_elapsed_ms}ms | total_tokens={telemetry.total_tokens} "
                f"(in={telemetry.input_tokens}, out={telemetry.output_tokens}) | trace_id={telemetry.trace_id}"
            )

            if not output_safety.is_safe:
                logger.warning(
                    f"Blocking unsafe assistant output. Categories: {output_safety.violating_categories}"
                )
                blocked_final = TravelProposalResponse(
                    assistant_message=self.config_manager.llama_guard_config.blocked_output_message,
                    proposals=[],
                    alternative_suggestions=None,
                    guardrail_blocked=True,
                    guardrail_violations=output_safety.violating_categories,
                    telemetry=telemetry,
                )
                yield AgentStreamEvent(
                    step_type=StreamStepType.FINAL,
                    stage="complete",
                    title="Response Blocked by Safety Guardrail",
                    detail="Output safety check blocked the generated response.",
                    metadata={"elapsed_ms": total_elapsed_ms},
                    response=blocked_final,
                )
                return

            proposal.telemetry = telemetry
            yield AgentStreamEvent(
                step_type=StreamStepType.FINAL,
                stage="complete",
                title="Recommendations Ready",
                detail=f"Prepared {len(proposal.proposals)} WeRoad itinerary recommendation(s) in {total_elapsed_ms / 1000:.1f}s.",
                metadata={"elapsed_ms": total_elapsed_ms},
                response=proposal,
            )