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
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Union

from src.ai_assistant.core.agent import GenericAgent
from src.ai_assistant.schemas.schemas import (
    LLMConfig,
    TravelProposalResponse,
    TravelSearchIntent,
)
from src.ai_assistant.core.config_management import ConfigManager
from src.ai_assistant.core.guardrail import LlamaGuardModerator
from src.ai_assistant.observability.observability import build_token_usage_summary, get_tracer

logger = logging.getLogger("WeRoadOrchestrator")
_tracer = get_tracer("weroad.orchestrator")


class WeRoadTravelAssistant:
    """Two-stage PydanticAI travel assistant with centralized config and I/O guardrails."""

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

    async def handle_query(self, user_query: str) -> TravelProposalResponse:
        """Runs the guarded two-stage agentic pipeline with OpenTelemetry tracing & token tracking."""
        stage_usages: Dict[str, Dict[str, int]] = {}
        obs_cfg = self.config_manager.observability_config

        with _tracer.start_as_current_span("weroad.handle_query") as root_span:
            if root_span is not None and hasattr(root_span, "set_attribute") and obs_cfg.capture_prompts:
                root_span.set_attribute("weroad.user_query", user_query[:500])

            # Step 0: Input Guardrail
            input_safety = await self.guardrail.check_input(user_query)
            if self.guardrail.last_input_usage:
                stage_usages["guardrail_input"] = self.guardrail.last_input_usage

            if not input_safety.is_safe:
                logger.warning(
                    f"Blocking unsafe user input. Categories: {input_safety.violating_categories}"
                )
                telemetry = build_token_usage_summary(stage_usages, config=obs_cfg)
                return TravelProposalResponse(
                    assistant_message=self.config_manager.llama_guard_config.blocked_input_message,
                    proposals=[],
                    guardrail_blocked=True,
                    guardrail_violations=input_safety.violating_categories,
                    telemetry=telemetry,
                )

            # Stage 1: Intent Extraction
            logger.info("Stage 1: Extracting TravelSearchIntent...")
            intent: TravelSearchIntent = await self.intent_agent.run(user_query)
            stage_usages["intent_extraction"] = dict(self.intent_agent.last_usage)
            logger.info(f"Parsed intent: {intent.model_dump_json()}")

            if root_span is not None and hasattr(root_span, "set_attribute"):
                root_span.set_attribute("weroad.intent.destination", intent.destination)
                if intent.departure_airport_iata:
                    root_span.set_attribute("weroad.intent.origin_iata", intent.departure_airport_iata)

            # Enforce departure airport presence when required by configuration
            if (
                self.config_manager.pipeline_config.require_departure_airport
                and not intent.departure_airport
                and not intent.departure_airport_iata
            ):
                logger.info("Departure airport missing from user query; prompting user for origin airport.")
                missing_msg = self.config_manager.prompts_config.missing_departure_airport_message.strip()
                output_safety = await self.guardrail.check_output(user_query, missing_msg)
                if self.guardrail.last_output_usage:
                    stage_usages["guardrail_output"] = self.guardrail.last_output_usage

                telemetry = build_token_usage_summary(stage_usages, config=obs_cfg)
                if not output_safety.is_safe:
                    return TravelProposalResponse(
                        assistant_message=self.config_manager.llama_guard_config.blocked_output_message,
                        proposals=[],
                        guardrail_blocked=True,
                        guardrail_violations=output_safety.violating_categories,
                        telemetry=telemetry,
                    )
                return TravelProposalResponse(
                    assistant_message=missing_msg,
                    proposals=[],
                    needs_departure_airport=True,
                    telemetry=telemetry,
                )

            # Stage 2: Catalog Search, Departure Verification & Flight Search
            logger.info("Stage 2: Searching WeRoad catalog, departure dates, and matching flights...")
            stage2_prompt = self.config_manager.prompts_config.stage2_user_prompt_template.format(
                user_query=user_query,
                intent_json=intent.model_dump_json(indent=2),
            )
            proposal: TravelProposalResponse = await self.proposal_agent.run(stage2_prompt)
            stage_usages["proposal_synthesis"] = dict(self.proposal_agent.last_usage)

            # Step 3: Output Guardrail
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

            logger.info(
                f"Request complete | total_tokens={telemetry.total_tokens} "
                f"(in={telemetry.input_tokens}, out={telemetry.output_tokens}) | trace_id={telemetry.trace_id}"
            )

            if not output_safety.is_safe:
                logger.warning(
                    f"Blocking unsafe assistant output. Categories: {output_safety.violating_categories}"
                )
                return TravelProposalResponse(
                    assistant_message=self.config_manager.llama_guard_config.blocked_output_message,
                    proposals=[],
                    alternative_suggestions=None,
                    guardrail_blocked=True,
                    guardrail_violations=output_safety.violating_categories,
                    telemetry=telemetry,
                )

            proposal.telemetry = telemetry
            return proposal