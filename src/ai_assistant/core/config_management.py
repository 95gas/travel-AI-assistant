"""
Centralized Configuration Manager (`config_management.py`)
==========================================================
Loads `.env` environment variables and YAML configuration files once upon startup,
exposing strongly-typed Pydantic sub-configurations for LLMs, Llama Guard,
WeRoad scraping, prompts, and MCP tooling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.ai_assistant.schemas.schemas import (
    FlightSearchConfig,
    LLMConfig,
    LlamaGuardConfig,
    ObservabilityConfig,
    ServerConfig,
    WeRoadAssistantConfig,
    WeRoadPipelineConfig,
    WeRoadPromptsConfig,
    WeRoadScraperConfig,
)

logger = logging.getLogger("ConfigManager")


class ConfigManager:
    """
    Singleton Configuration Manager.

    Parses `config/config_weroad.yaml` (or root `config_weroad.yaml` fallback) and
    optional `config/config.yaml` into strongly-typed Pydantic models.
    """

    _instance: Optional["ConfigManager"] = None

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
    ):
        load_dotenv()

        self.config_path = self._resolve_config_path(
            config_path, fallback_name="config.yaml"
        )

        # Raw general YAML contents
        self._general_raw: Dict[str, Any] = {}

        # High-level directory attributes
        self.dataset_path: Path = Path("./data")
        self.debug_dir: Path = Path("./debug")

        # Strongly-typed WeRoad Assistant configuration & shortcuts
        self.weroad_config: WeRoadAssistantConfig
        self.pipeline_config: WeRoadPipelineConfig
        self.llm_default_config: LLMConfig
        self.llm_intent_config: LLMConfig
        self.llm_proposal_config: LLMConfig
        self.llama_guard_config: LlamaGuardConfig
        self.scraper_config: WeRoadScraperConfig
        self.flight_search_config: FlightSearchConfig
        self.server_config: ServerConfig
        self.observability_config: ObservabilityConfig
        self.prompts_config: WeRoadPromptsConfig
        self.mcp_config_path: Path

        self._load_and_parse()

    @staticmethod
    def _resolve_config_path(primary_path: str | Path, fallback_name: str) -> Path:
        """Resolves a config path, checking project root fallback if `config/<file>` is absent."""
        candidate = Path(primary_path)
        if candidate.exists():
            return candidate
        root_fallback = Path(fallback_name)
        if root_fallback.exists():
            return root_fallback
        return candidate

    def _ensure_directories_exist(self) -> None:
        """Creates runtime directories if they do not exist."""
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_parse(self) -> None:
        """Loads YAML files from disk and validates them into Pydantic schemas."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._general_raw = yaml.safe_load(f) or {}

        self.dataset_path = Path(self._general_raw.get("dataset_path", "./data"))
        self.debug_dir = Path(self._general_raw.get("debug", "./debug"))
        self._ensure_directories_exist()

        self.weroad_config = WeRoadAssistantConfig.load_from_yaml(self.config_path)
        self.pipeline_config = self.weroad_config.pipeline
        self.llm_default_config = self.pipeline_config.llm_default
        self.llm_intent_config = self.pipeline_config.resolved_intent_llm
        self.llm_proposal_config = self.pipeline_config.resolved_proposal_llm
        self.llama_guard_config = self.weroad_config.llama_guard
        self.scraper_config = self.weroad_config.scraper
        self.flight_search_config = self.weroad_config.flight_search
        self.server_config = self.weroad_config.server
        self.observability_config = self.weroad_config.observability
        self.prompts_config = self.weroad_config.prompts

        configured_mcp = Path(self.weroad_config.schemas.mcp_config_path)
        if configured_mcp.exists():
            self.mcp_config_path = configured_mcp
        elif Path("mcp_config.json").exists():
            self.mcp_config_path = Path("mcp_config.json")
        else:
            self.mcp_config_path = configured_mcp

    @classmethod
    def get_instance(
        cls,
        config_path: str | Path = "config/config.yaml",
    ) -> "ConfigManager":
        """Singleton accessor ensuring configs load only once per runtime."""
        if cls._instance is None:
            load_dotenv()
            cls._instance = cls(
                config_path=config_path,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton instance (useful for tests or hot-reloading configs)."""
        cls._instance = None