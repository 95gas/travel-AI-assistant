"""
Llama Guard Moderation Agent (`guardrail.py`)
=============================================
Integrates Meta Llama Guard via OpenRouter to evaluate safety for incoming
user queries and synthesized final agent responses using `ConfigManager`.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.ai_assistant.core.agent import GenericAgent
from src.ai_assistant.core.config_management import ConfigManager
from src.ai_assistant.schemas.schemas import LLMConfig, LlamaGuardConfig, SafetyCheckResult

logger = logging.getLogger("WeRoad.LlamaGuard")


class LlamaGuardModerator:
    """Evaluates input and output safety policies using Llama Guard."""

    def __init__(self, config: Optional[LlamaGuardConfig] = None):
        self.config = config or ConfigManager.get_instance().llama_guard_config
        self.system_prompt = self.config.system_prompt
        self.category_map = self.config.category_map

        self.llm_settings = LLMConfig(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
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
        formatted_prompt = f"<|user|>\n{user_query}\n<|end|>\n<|assistant|>"

        try:
            agent = self._get_agent()
            res = await agent.run(formatted_prompt)
            self.last_input_usage = dict(agent.last_usage)
            result = self._parse_llama_guard_response(str(res or ""), checked_target="input")

            if not result.is_safe:
                logger.warning(
                    f"[LlamaGuard] Input query flagged unsafe: {result.violating_categories}"
                )
            else:
                logger.info("[LlamaGuard] Input query passed safety check.")
            return result
        except Exception as exc:
            logger.exception(f"[LlamaGuard] Input moderation error: {exc}")
            return self._handle_moderation_error(exc, checked_target="input")

    async def check_output(self, user_query: str, synthesized_answer: str) -> SafetyCheckResult:
        """Moderates final synthesized response prior to returning it to the user."""
        self.last_output_usage = {}
        if not self.config.enabled or not self.config.check_output:
            return SafetyCheckResult(is_safe=True, checked_target="output")

        logger.info(f"[LlamaGuard] Moderating synthesized answer using {self.config.model_name}...")
        formatted_prompt = (
            f"<|user|>\n{user_query}\n<|end|>\n"
            f"<|assistant|>\n{synthesized_answer}\n<|end|>"
        )

        try:
            agent = self._get_agent()
            res = await agent.run(formatted_prompt)
            self.last_output_usage = dict(agent.last_usage)
            result = self._parse_llama_guard_response(str(res or ""), checked_target="output")

            if not result.is_safe:
                logger.warning(
                    f"[LlamaGuard] Final output flagged unsafe: {result.violating_categories}"
                )
            else:
                logger.info("[LlamaGuard] Final output passed safety check.")
            return result
        except Exception as exc:
            logger.exception(f"[LlamaGuard] Output moderation error: {exc}")
            return self._handle_moderation_error(exc, checked_target="output")