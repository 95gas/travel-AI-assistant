"""
Llama Guard Moderation Agent (`guardrail.py`)
=============================================
Integrates Meta Llama Guard via OpenRouter, local Ollama, or local `transformers`
to evaluate safety for incoming user queries and synthesized final agent responses
using `ConfigManager`.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic_ai import ModelSettings

from src.ai_assistant.core.agent import GenericAgent
from src.ai_assistant.core.config_management import ConfigManager
from src.ai_assistant.schemas.schemas import LLMConfig, LlamaGuardConfig, SafetyCheckResult
from src.ai_assistant.observability.observability import record_guardrail_event


logger = logging.getLogger("WeRoad.LlamaGuard")


class LlamaGuardModerator:
    """Evaluates input and output safety policies using Llama Guard."""

    def __init__(self, config: Optional[LlamaGuardConfig] = None):
        self.config = config or ConfigManager.get_instance().llama_guard_config
        self.system_prompt = self.config.system_prompt
        self.category_map = self.config.category_map

        # OpenRouter's `openrouter/free` router can select reasoning models that consume
        # internal thinking tokens before emitting 'safe'/'unsafe', so ensure >= 1024 tokens.
        effective_max_tokens = self.config.max_tokens
        if self.config.model_name.startswith("openrouter:") and effective_max_tokens < 1024:
            effective_max_tokens = 1024

        self.llm_settings = LLMConfig(
            model_name=self.config.model_name,
            provider=getattr(self.config, "provider", None),
            device=getattr(self.config, "device", None),
            torch_dtype=getattr(self.config, "torch_dtype", None),
            load_in_4bit=bool(getattr(self.config, "load_in_4bit", False)),
            reasoning_enabled=getattr(self.config, "reasoning_enabled", False),
            reasoning_effort=getattr(self.config, "reasoning_effort", None),
            temperature=self.config.temperature,
            max_tokens=effective_max_tokens,
        )
        self._agent: Optional[GenericAgent] = None
        self.last_input_usage: dict[str, int] = {}
        self.last_output_usage: dict[str, int] = {}

    def _get_agent(self) -> GenericAgent:
        """Lazily initializes and reuses a single `GenericAgent` instance."""
        if self._agent is None:
            self._agent = GenericAgent(
                llm_settings=self.llm_settings,
                system_prompt=self.system_prompt,
                request_limit=1,
                stage_name="guardrail",
            )
        return self._agent

    async def _run_with_token_retry(self, prompt: str) -> str:
        """Runs the moderation agent and retries once with higher `max_tokens` if thinking exhausted the budget."""
        agent = self._get_agent()
        try:
            res = await agent.run(prompt)
            return str(res or "")
        except Exception as exc:
            if "Model token limit" in str(exc):
                logger.warning(
                    f"[LlamaGuard] Token limit hit on {self.config.model_name}; "
                    "retrying once with max_tokens=2048..."
                )
                res = await agent.run(
                    prompt,
                    model_settings=ModelSettings(
                        temperature=self.config.temperature,
                        max_tokens=2048,
                    ),
                )
                return str(res or "")
            raise

    def _parse_llama_guard_response(self, raw_output: str, checked_target: str) -> SafetyCheckResult:
        """Parses Llama Guard output format ('safe' vs 'unsafe\\nS1')."""
        clean_text = raw_output.strip()
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]

        if not lines:
            return SafetyCheckResult(
                is_safe=True,
                checked_target=checked_target,
                raw_response=raw_output,
            )

        first_line = lines[0].lower()
        if "unsafe" in first_line:
            categories = []
            if len(lines) > 1:
                raw_codes = [c.strip().upper() for c in lines[1].split(",") if c.strip()]
                for cat_code in raw_codes:
                    cat_description = self.category_map.get(cat_code, cat_code)
                    categories.append(f"{cat_code}: {cat_description}")
            return SafetyCheckResult(
                is_safe=False,
                violating_categories=categories,
                raw_response=clean_text,
                checked_target=checked_target,
            )

        return SafetyCheckResult(
            is_safe=True,
            violating_categories=[],
            raw_response=clean_text,
            checked_target=checked_target,
        )

    def _handle_moderation_error(self, exc: Exception, checked_target: str) -> SafetyCheckResult:
        """Applies configured fail-open vs fail-closed policy when moderation errors occur."""
        if self.config.fail_open:
            logger.warning(
                f"[LlamaGuard] {checked_target} moderation failed ({exc}); failing open."
            )
            return SafetyCheckResult(
                is_safe=True,
                checked_target=checked_target,
                raw_response=f"Error (fail_open=True): {exc}",
            )
        logger.error(
            f"[LlamaGuard] {checked_target} moderation failed ({exc}); failing closed."
        )
        return SafetyCheckResult(
            is_safe=False,
            violating_categories=["MODERATION_UNAVAILABLE"],
            checked_target=checked_target,
            raw_response=f"Error (fail_open=False): {exc}",
        )

    async def check_input(self, user_query: str) -> SafetyCheckResult:
        """Moderates incoming user query prior to LLM/tool execution."""

        self.last_input_usage = {}
        if not self.config.enabled or not self.config.check_input:
            return SafetyCheckResult(is_safe=True, checked_target="input")

        logger.info(f"[LlamaGuard] Moderating user query using {self.config.model_name}...")
        formatted_prompt = f"Evaluate the safety of this user message:\n\n{user_query}"

        start = time.perf_counter()

        try:
            raw_res = await self._run_with_token_retry(formatted_prompt)
            agent = self._get_agent()
            self.last_input_usage = dict(agent.last_usage)
            result = self._parse_llama_guard_response(raw_res, checked_target="input")

            record_guardrail_event(
                target="input",
                is_safe=result.is_safe,
                categories=result.violating_categories,
                duration_seconds=time.perf_counter() - start,
            )

            if not result.is_safe:
                logger.warning(
                    f"[LlamaGuard] Input query flagged unsafe: {result.violating_categories}"
                )
            else:
                logger.info("[LlamaGuard] Input query passed safety check.")
            return result
        except Exception as exc:
            logger.exception(f"[LlamaGuard] Input moderation error: {exc}")
            result = self._handle_moderation_error(exc, checked_target="input")
            record_guardrail_event(
                target="input",
                is_safe=result.is_safe,
                categories=result.violating_categories,
                duration_seconds=time.perf_counter() - start,
            )
            return result

    async def check_output(self, user_query: str, synthesized_answer: str) -> SafetyCheckResult:
        """Moderates final synthesized response prior to returning it to the user."""

        self.last_output_usage = {}
        if not self.config.enabled or not self.config.check_output:
            return SafetyCheckResult(is_safe=True, checked_target="output")

        logger.info(f"[LlamaGuard] Moderating synthesized answer using {self.config.model_name}...")
        formatted_prompt = (
            f"Evaluate the safety of this assistant response to the user.\n\n"
            f"User message:\n{user_query}\n\n"
            f"Assistant response:\n{synthesized_answer}"
        )

        start = time.perf_counter()

        try:
            raw_res = await self._run_with_token_retry(formatted_prompt)
            agent = self._get_agent()
            self.last_output_usage = dict(agent.last_usage)
            result = self._parse_llama_guard_response(raw_res, checked_target="output")

            record_guardrail_event(
                target="output",
                is_safe=result.is_safe,
                categories=result.violating_categories,
                duration_seconds=time.perf_counter() - start,
            )

            if not result.is_safe:
                logger.warning(
                    f"[LlamaGuard] Final output flagged unsafe: {result.violating_categories}"
                )
            else:
                logger.info("[LlamaGuard] Final output passed safety check.")
            return result
        except Exception as exc:
            logger.exception(f"[LlamaGuard] Output moderation error: {exc}")
            result = self._handle_moderation_error(exc, checked_target="output")
            record_guardrail_event(
                target="output",
                is_safe=result.is_safe,
                categories=result.violating_categories,
                duration_seconds=time.perf_counter() - start,
            )
            return result