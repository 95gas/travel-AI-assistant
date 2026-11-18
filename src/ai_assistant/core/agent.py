import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence, Union

from pydantic_ai import Agent, ModelSettings, UsageLimits
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

try:
    from pydantic_ai.messages import ThinkingPart
except ImportError:  # pragma: no cover
    ThinkingPart = None  # type: ignore[assignment,misc]

try:
    from pydantic_ai.profiles import ModelProfile
except ImportError:  # pragma: no cover
    ModelProfile = None  # type: ignore[assignment,misc]

from src.ai_assistant.schemas.schemas import (
    AgentStreamEvent,
    LLMConfig,
    StreamStepType,
)
from src.ai_assistant.observability.observability import (
    extract_usage_from_result,
    get_tracer,
    record_llm_usage,
)

_tracer = get_tracer("weroad.agent")
_TRANSFORMERS_CACHE: dict[tuple[str, str, str, bool], tuple[Any, Any]] = {}
_TRANSFORMERS_CACHE_LOCK = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("GenericAgent")


class GenericAgent:
    """
    A generic, reusable wrapper around Pydantic AI's Agent class.

    Supports:
    - OpenRouter Free Models Router & `:free` variants (`openrouter:openrouter/free`, etc.)
      with unified reasoning tokens (`reasoning_enabled`, `reasoning_effort`, `reasoning_max_tokens`)
    - Local Ollama servers (`ollama:qwen2.5:3b`, `ollama:qwen2.5:1.5b`, etc.)
    - Local Hugging Face `transformers` checkpoints (`transformers:google/gemma-3-1b-it`, etc.)
    - Remote Hugging Face Inference Providers (`huggingface:google/gemma-3-27b-it`, etc.)
    - Structured Pydantic output types, MCP toolsets, and OpenTelemetry instrumentation
    - Real-time node-by-node streaming (`run_stream`) of intermediate reasoning and tool calls
    """

    def __init__(
        self,
        llm_settings: LLMConfig,
        system_prompt: str = "",
        output_type: Optional[Any] = None,
        mcp_config_path: Optional[Union[str, Path]] = None,
        builtin_tools: Optional[Sequence[Any]] = None,
        request_limit: int = 5,
        stage_name: str = "generic_agent",
    ):
        self.model_name = llm_settings.model_name
        self.provider_name = getattr(llm_settings, "provider", None)
        self.base_url = getattr(llm_settings, "base_url", None)
        self.device = getattr(llm_settings, "device", None) or "auto"
        self.torch_dtype = getattr(llm_settings, "torch_dtype", None) or "auto"
        self.load_in_4bit = bool(getattr(llm_settings, "load_in_4bit", False))
        self.temperature = llm_settings.temperature
        self.max_tokens = llm_settings.max_tokens
        self.reasoning_enabled: Optional[bool] = getattr(llm_settings, "reasoning_enabled", None)
        self.reasoning_effort: Optional[str] = getattr(llm_settings, "reasoning_effort", None)
        self.reasoning_max_tokens: Optional[int] = getattr(llm_settings, "reasoning_max_tokens", None)
        self.reasoning_exclude: Optional[bool] = getattr(llm_settings, "reasoning_exclude", None)
        self.system_prompt = system_prompt
        self.output_type = output_type
        self.mcp_config_path = Path(mcp_config_path) if mcp_config_path else None
        self.request_limit = request_limit
        self.stage_name = stage_name
        self.last_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        self.last_output: Any = None

        self.toolsets = None
        if self.mcp_config_path:
            if not self.mcp_config_path.exists():
                raise FileNotFoundError(f"MCP config path not found: {self.mcp_config_path}")
            self.toolsets = self._build_mcp_toolsets()

        resolved_model = self._resolve_model(
            model_name=self.model_name,
            provider_name=self.provider_name,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            device=self.device,
            torch_dtype=self.torch_dtype,
            load_in_4bit=self.load_in_4bit,
        )

        model_settings = self._build_model_settings()

        agent_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "system_prompt": self.system_prompt,
            "model_settings": model_settings,
        }
        if self.output_type is not None:
            agent_kwargs["output_type"] = self.output_type
        if self.toolsets is not None:
            agent_kwargs["toolsets"] = self.toolsets
        if builtin_tools:
            agent_kwargs["builtin_tools"] = list(builtin_tools)

        try:
            self.agent = Agent(**agent_kwargs, instrument=True)
        except TypeError:
            self.agent = Agent(**agent_kwargs)

    def _build_openrouter_reasoning_config(self) -> Optional[dict[str, Any]]:
        """
        Builds OpenRouter's unified `reasoning` config object.
        OpenRouter accepts either `effort` or `max_tokens` (not both simultaneously),
        or `enabled: True/False`, plus optional `exclude: True/False`.
        For `openrouter:` models, reasoning defaults to enabled unless explicitly disabled.
        """
        reasoning_cfg: dict[str, Any] = {}

        if self.reasoning_enabled is False:
            reasoning_cfg["enabled"] = False
        elif self.reasoning_effort is not None:
            reasoning_cfg["effort"] = self.reasoning_effort
        elif self.reasoning_max_tokens is not None:
            reasoning_cfg["max_tokens"] = self.reasoning_max_tokens
        elif self.reasoning_enabled is True or self.model_name.startswith("openrouter:"):
            reasoning_cfg["enabled"] = True

        if self.reasoning_exclude is not None:
            reasoning_cfg["exclude"] = self.reasoning_exclude

        return reasoning_cfg or None

    def _build_model_settings(self) -> ModelSettings:
        """Constructs provider-appropriate `ModelSettings` (including OpenRouter reasoning & streaming usage)."""
        if self.model_name.startswith("openrouter:"):
            reasoning_cfg = self._build_openrouter_reasoning_config()
            extra_body: dict[str, Any] = {
                "usage": {"include": True},
                "stream_options": {"include_usage": True},
            }
            if reasoning_cfg is not None:
                extra_body["reasoning"] = reasoning_cfg

            try:
                from pydantic_ai.models.openrouter import OpenRouterModelSettings

                or_kwargs: dict[str, Any] = {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "openrouter_usage": {"include": True},
                    "extra_body": {"stream_options": {"include_usage": True}},
                }
                if reasoning_cfg is not None:
                    or_kwargs["openrouter_reasoning"] = reasoning_cfg
                return OpenRouterModelSettings(**or_kwargs)
            except (ImportError, TypeError):
                return ModelSettings(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body,
                )

        # For local Ollama models on 4 GB GPUs, cap `num_ctx` to 3072 (shared across all stages)
        # so Ollama keeps a single loaded runner and avoids KV-cache VRAM OOM crashes (0xc0000409).
        if self.model_name.startswith("ollama:"):
            ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "3072"))
            return ModelSettings(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"options": {"num_ctx": ollama_num_ctx}},
            )

        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @staticmethod
    def _extract_usage(result: Any) -> dict[str, int]:
        """
        Extracts token usage from a PydanticAI `AgentRunResult`, falling back to
        inspecting `result.usage()` / `result.usage` and `ModelResponse.usage` parts
        if `extract_usage_from_result` returns zeros.
        """
        usage = dict(extract_usage_from_result(result))
        if usage.get("total_tokens", 0) > 0 or result is None:
            return usage

        raw_usage = getattr(result, "usage", None)
        if callable(raw_usage):
            try:
                raw_usage = raw_usage()
            except Exception:
                raw_usage = None

        if raw_usage is not None:
            in_tok = int(
                getattr(raw_usage, "input_tokens", None)
                or getattr(raw_usage, "request_tokens", None)
                or getattr(raw_usage, "prompt_tokens", None)
                or 0
            )
            out_tok = int(
                getattr(raw_usage, "output_tokens", None)
                or getattr(raw_usage, "response_tokens", None)
                or getattr(raw_usage, "completion_tokens", None)
                or 0
            )
            tot_tok = int(getattr(raw_usage, "total_tokens", None) or (in_tok + out_tok))
            reqs = int(getattr(raw_usage, "requests", None) or usage.get("requests", 1) or 1)
            if tot_tok > 0 or in_tok > 0 or out_tok > 0:
                return {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok or (in_tok + out_tok),
                    "requests": reqs,
                }

        messages_fn = getattr(result, "new_messages", None) or getattr(result, "all_messages", None)
        if callable(messages_fn):
            try:
                in_sum = 0
                out_sum = 0
                resp_count = 0
                for msg in messages_fn():
                    msg_usage = getattr(msg, "usage", None)
                    if msg_usage is not None:
                        resp_count += 1
                        in_sum += int(
                            getattr(msg_usage, "input_tokens", None)
                            or getattr(msg_usage, "request_tokens", None)
                            or 0
                        )
                        out_sum += int(
                            getattr(msg_usage, "output_tokens", None)
                            or getattr(msg_usage, "response_tokens", None)
                            or 0
                        )
                if (in_sum + out_sum) > 0:
                    return {
                        "input_tokens": in_sum,
                        "output_tokens": out_sum,
                        "total_tokens": in_sum + out_sum,
                        "requests": max(resp_count, 1),
                    }
            except Exception:
                pass

        return usage

    def _build_mcp_toolsets(self) -> Any:
        """
        Loads MCP toolsets from `self.mcp_config_path`, normalizing `python`/`python3`
        commands to `sys.executable` and injecting `PYTHONPATH` so stdio MCP subprocesses
        run inside the active virtual environment without failing on startup.
        """
        assert self.mcp_config_path is not None
        try:
            raw_config = json.loads(self.mcp_config_path.read_text(encoding="utf-8"))
            servers = raw_config.get("mcpServers") if isinstance(raw_config, dict) else None
            config_to_load = self.mcp_config_path

            if isinstance(servers, dict):
                project_root = str(Path(__file__).resolve().parents[3])
                modified = False
                for server_name, server_cfg in servers.items():
                    if not isinstance(server_cfg, dict):
                        continue
                    cmd = server_cfg.get("command")
                    if isinstance(cmd, str) and cmd.lower() in {"python", "python3", "python.exe"}:
                        server_cfg["command"] = sys.executable
                        modified = True

                    if "command" in server_cfg:
                        env = dict(os.environ)
                        if isinstance(server_cfg.get("env"), dict):
                            env.update({str(k): str(v) for k, v in server_cfg["env"].items()})
                        existing_pp = env.get("PYTHONPATH", "")
                        if project_root not in existing_pp.split(os.pathsep):
                            env["PYTHONPATH"] = (
                                f"{project_root}{os.pathsep}{existing_pp}" if existing_pp else project_root
                            )
                        env.setdefault("PYTHONUNBUFFERED", "1")
                        server_cfg["env"] = env
                        modified = True

                if modified:
                    tmp_path = Path(tempfile.gettempdir()) / f"weroad_resolved_{self.mcp_config_path.name}"
                    tmp_path.write_text(json.dumps(raw_config, indent=2), encoding="utf-8")
                    config_to_load = tmp_path

            toolsets = load_mcp_toolsets(config_to_load)
            for ts in toolsets:
                client = getattr(getattr(ts, "wrapped", ts), "client", None)
                if client:
                    client._init_timeout = 60.0
            logger.info(f"Loaded MCP toolsets from {self.mcp_config_path}")
            return toolsets
        except Exception as e:
            logger.error(f"Failed to load MCP toolsets from {self.mcp_config_path}: {e}")
            raise

    @classmethod
    def preload(
        cls,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Preloads a local `transformers:` checkpoint into the shared in-memory cache
        (e.g., during FastAPI lifespan startup) so the first user request does not time out.
        """
        if model_name.startswith("transformers:"):
            repo_id = model_name.split(":", 1)[1]
            cls._load_transformers_pipeline(
                repo_id=repo_id,
                device=device or "auto",
                torch_dtype=torch_dtype or "auto",
                load_in_4bit=load_in_4bit,
            )

    @staticmethod
    def _load_transformers_pipeline(
        repo_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_4bit: bool = False,
    ) -> tuple[Any, Any]:
        """Loads and caches a Hugging Face `transformers` tokenizer and causal LM."""
        norm_device = device or "auto"
        norm_dtype = torch_dtype or "auto"
        norm_4bit = bool(load_in_4bit)
        cache_key = (repo_id, norm_device, norm_dtype, norm_4bit)

        with _TRANSFORMERS_CACHE_LOCK:
            if cache_key in _TRANSFORMERS_CACHE:
                return _TRANSFORMERS_CACHE[cache_key]

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Local `transformers` support requires `transformers` and `torch`. "
                    "Install them via: pip install transformers torch"
                ) from exc

            resolved_dtype: Any = "auto"
            if norm_dtype != "auto":
                resolved_dtype = getattr(torch, norm_dtype, "auto")

            quantization_config = None
            if norm_4bit:
                if torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes") is not None:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    logger.warning(
                        "load_in_4bit=True requested, but CUDA + `bitsandbytes` is unavailable; "
                        "falling back to standard precision."
                    )

            has_accelerate = importlib.util.find_spec("accelerate") is not None
            logger.info(
                f"Loading local Hugging Face transformers model: {repo_id} "
                f"(device={norm_device}, dtype={norm_dtype}, 4bit={quantization_config is not None}, accelerate={has_accelerate})..."
            )
            load_start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(repo_id)

            if quantization_config is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    quantization_config=quantization_config,
                    device_map=norm_device,
                )
            elif has_accelerate:
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=resolved_dtype,
                    device_map=norm_device,
                )
            else:
                if norm_device != "auto":
                    target_device = norm_device
                elif torch.cuda.is_available():
                    target_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"

                load_kwargs: dict[str, Any] = {}
                if resolved_dtype != "auto" and target_device != "cpu":
                    load_kwargs["torch_dtype"] = resolved_dtype

                model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
                model = model.to(target_device)

            model.eval()
            logger.info(f"Loaded {repo_id} in {time.perf_counter() - load_start:.2f}s")

            _TRANSFORMERS_CACHE[cache_key] = (tokenizer, model)
            return tokenizer, model

    @staticmethod
    def _extract_json_payload(raw_text: str) -> Optional[dict[str, Any]]:
        """Extracts the first valid JSON object from raw model text (handling markdown code fences)."""
        cleaned = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @classmethod
    def _schema_to_example_json(
        cls,
        schema: dict[str, Any],
        root_schema: Optional[dict[str, Any]] = None,
        depth: int = 0,
    ) -> Any:
        """
        Converts a verbose JSON Schema into a compact concrete JSON template so smaller
        local models populate field values instead of echoing `"properties": {...}`.
        """
        if root_schema is None:
            root_schema = schema
        if depth > 3 or not isinstance(schema, dict):
            return None

        if "$ref" in schema and isinstance(schema["$ref"], str):
            ref_path = schema["$ref"].lstrip("#/").split("/")
            target: Any = root_schema
            for part in ref_path:
                if isinstance(target, dict):
                    target = target.get(part)
                else:
                    target = None
                    break
            if isinstance(target, dict):
                return cls._schema_to_example_json(target, root_schema, depth + 1)

        for union_key in ("anyOf", "oneOf"):
            if union_key in schema and isinstance(schema[union_key], list):
                non_null = [
                    branch
                    for branch in schema[union_key]
                    if isinstance(branch, dict) and branch.get("type") != "null"
                ]
                if non_null:
                    return cls._schema_to_example_json(non_null[0], root_schema, depth + 1)
                return None

        schema_type = schema.get("type")
        if schema_type == "object" or "properties" in schema:
            props = schema.get("properties", {})
            if not isinstance(props, dict):
                return {}
            return {
                key: cls._schema_to_example_json(prop_schema, root_schema, depth + 1)
                for key, prop_schema in props.items()
            }

        if schema_type == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                return [cls._schema_to_example_json(items, root_schema, depth + 1)]
            return []

        if schema_type == "string":
            if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
                return " | ".join(str(v) for v in schema["enum"])
            if schema.get("format") == "date":
                return "YYYY-MM-DD"
            return "<string>"

        if schema_type == "integer":
            return 0

        if schema_type == "number":
            return 0.0

        if schema_type == "boolean":
            return False

        return None

    @classmethod
    def _build_transformers_model(
        cls,
        repo_id: str,
        temperature: float,
        max_tokens: int,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_4bit: bool = False,
    ) -> FunctionModel:
        """
        Wraps a local Hugging Face `transformers` causal LM in PydanticAI's `FunctionModel`
        so it supports multi-turn messages, MCP/function tool calling, and structured outputs.
        """

        async def _run_local_transformers(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tokenizer, model = await asyncio.to_thread(
                cls._load_transformers_pipeline, repo_id, device, torch_dtype, load_in_4bit
            )

            system_chunks: list[str] = []
            if getattr(info, "instructions", None):
                system_chunks.append(str(info.instructions))

            chat_messages: list[dict[str, str]] = []
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, SystemPromptPart) and part.content:
                            system_chunks.append(part.content)
                        elif isinstance(part, UserPromptPart):
                            if isinstance(part.content, str):
                                text_content = part.content
                            else:
                                text_content = "\n".join(
                                    item for item in part.content if isinstance(item, str)
                                )
                            chat_messages.append({"role": "user", "content": text_content})
                        elif isinstance(part, ToolReturnPart):
                            payload = (
                                part.content
                                if isinstance(part.content, str)
                                else json.dumps(part.content, ensure_ascii=False, default=str)
                            )
                            chat_messages.append(
                                {
                                    "role": "user",
                                    "content": f"[Tool Result for `{part.tool_name}`]:\n{payload}",
                                }
                            )
                        elif isinstance(part, RetryPromptPart):
                            chat_messages.append(
                                {
                                    "role": "user",
                                    "content": f"Validation error—please correct your previous response:\n{part.model_response()}",
                                }
                            )
                elif isinstance(msg, ModelResponse):
                    assistant_segments: list[str] = []
                    for part in msg.parts:
                        if isinstance(part, TextPart) and part.content:
                            assistant_segments.append(part.content)
                        elif isinstance(part, ToolCallPart):
                            if isinstance(part.args, str):
                                try:
                                    parsed_args = json.loads(part.args) if part.args else {}
                                except json.JSONDecodeError:
                                    parsed_args = {"raw": part.args}
                            else:
                                parsed_args = part.args or {}
                            assistant_segments.append(
                                json.dumps(
                                    {"tool_call": {"name": part.tool_name, "arguments": parsed_args}},
                                    ensure_ascii=False,
                                )
                            )
                    if assistant_segments:
                        chat_messages.append({"role": "assistant", "content": "\n".join(assistant_segments)})

            function_tools = list(getattr(info, "function_tools", None) or [])
            output_tools = list(getattr(info, "output_tools", None) or [])
            allow_text_output = getattr(info, "allow_text_output", True)

            if function_tools:
                tools_spec = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters_json_schema,
                    }
                    for t in function_tools
                ]
                system_chunks.append(
                    "You have access to the following tools:\n"
                    f"{json.dumps(tools_spec, ensure_ascii=False, indent=2)}\n"
                    "To call a tool, respond ONLY with a JSON object in this exact format:\n"
                    '{"tool_call": {"name": "<tool_name>", "arguments": {<arguments>}}}'
                )

            if output_tools:
                target_tool = output_tools[0]
                example_template = cls._schema_to_example_json(target_tool.parameters_json_schema)
                system_chunks.append(
                    "You MUST respond ONLY with a single valid JSON object filled with the extracted values "
                    "(do NOT output a JSON schema with 'properties' or 'type', no markdown fences, no commentary).\n"
                    "Use this exact JSON structure (use null for unknown optional fields):\n"
                    f"{json.dumps(example_template, ensure_ascii=False, indent=2)}"
                )

            if system_chunks:
                combined_system = "\n\n".join(system_chunks).strip()
                if chat_messages and chat_messages[0]["role"] == "user":
                    chat_messages[0]["content"] = f"{combined_system}\n\n{chat_messages[0]['content']}"
                else:
                    chat_messages.insert(0, {"role": "user", "content": combined_system})

            merged_messages: list[dict[str, str]] = []
            for entry in chat_messages:
                if merged_messages and merged_messages[-1]["role"] == entry["role"]:
                    merged_messages[-1]["content"] = (
                        f"{merged_messages[-1]['content']}\n\n{entry['content']}"
                    )
                else:
                    merged_messages.append(dict(entry))

            def _generate_sync() -> tuple[str, int, int]:
                import torch

                prompt_text = tokenizer.apply_chat_template(
                    merged_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_device = model.device
                if getattr(input_device, "type", None) == "meta":
                    input_device = next(
                        (p.device for p in model.parameters() if p.device.type != "meta"),
                        torch.device("cpu"),
                    )

                inputs = tokenizer(prompt_text, return_tensors="pt").to(input_device)
                input_len = int(inputs["input_ids"].shape[-1])

                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if temperature > 0.0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                else:
                    gen_kwargs["do_sample"] = False

                with torch.inference_mode():
                    output_ids = model.generate(**inputs, **gen_kwargs)

                new_tokens = output_ids[0][input_len:]
                output_len = int(new_tokens.shape[-1])
                decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                return decoded, input_len, output_len

            decoded_text, in_tokens, out_tokens = await asyncio.to_thread(_generate_sync)
            usage = RequestUsage(input_tokens=in_tokens, output_tokens=out_tokens)

            parsed_json = cls._extract_json_payload(decoded_text)
            if parsed_json is not None:
                if "tool_call" in parsed_json and isinstance(parsed_json["tool_call"], dict) and function_tools:
                    tc = parsed_json["tool_call"]
                    tc_name = str(tc.get("name", ""))
                    tc_args = tc.get("arguments", {})
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name=tc_name,
                                args=tc_args,
                                tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
                            )
                        ],
                        usage=usage,
                        model_name=f"transformers:{repo_id}",
                    )

                if output_tools and not allow_text_output:
                    target_tool = output_tools[0]
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name=target_tool.name,
                                args=parsed_json,
                                tool_call_id=f"out_{uuid.uuid4().hex[:12]}",
                            )
                        ],
                        usage=usage,
                        model_name=f"transformers:{repo_id}",
                    )

                return ModelResponse(
                    parts=[TextPart(content=json.dumps(parsed_json, ensure_ascii=False))],
                    usage=usage,
                    model_name=f"transformers:{repo_id}",
                )

            return ModelResponse(
                parts=[TextPart(content=decoded_text)],
                usage=usage,
                model_name=f"transformers:{repo_id}",
            )

        if ModelProfile is not None:
            try:
                return FunctionModel(
                    _run_local_transformers,
                    model_name=f"transformers:{repo_id}",
                    profile=ModelProfile(default_structured_output_mode="tool"),
                )
            except TypeError:
                pass

        return FunctionModel(_run_local_transformers, model_name=f"transformers:{repo_id}")

    @classmethod
    def _resolve_model(
        cls,
        model_name: str,
        provider_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_4bit: bool = False,
    ) -> Any:
        if model_name.startswith("openrouter:"):
            openrouter_model_id = model_name.split(":", 1)[1]
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENROUTER_API_KEY environment variable is not set; OpenRouter requests may fail."
                )

            app_url = os.getenv("OPENROUTER_SITE_URL") or os.getenv("HTTP_REFERER")
            app_title = (
                os.getenv("OPENROUTER_APP_NAME")
                or os.getenv("X_TITLE")
                or "WeRoad Travel Assistant"
            )

            try:
                from pydantic_ai.models.openrouter import OpenRouterModel
                from pydantic_ai.providers.openrouter import OpenRouterProvider

                provider_kwargs: dict[str, Any] = {}
                if api_key:
                    provider_kwargs["api_key"] = api_key
                if app_url:
                    provider_kwargs["app_url"] = app_url
                if app_title:
                    provider_kwargs["app_title"] = app_title

                try:
                    provider = OpenRouterProvider(**provider_kwargs)
                except TypeError:
                    provider = (
                        OpenRouterProvider(api_key=api_key)
                        if api_key
                        else OpenRouterProvider()
                    )

                return OpenRouterModel(openrouter_model_id, provider=provider)
            except ImportError:
                try:
                    from pydantic_ai.models.openai import OpenAIChatModel
                    from pydantic_ai.providers.openai import OpenAIProvider

                    return OpenAIChatModel(
                        openrouter_model_id,
                        provider=OpenAIProvider(
                            base_url=base_url or "https://openrouter.ai/api/v1",
                            api_key=api_key or "",
                        ),
                    )
                except ImportError:
                    return model_name

        if model_name.startswith("transformers:"):
            repo_id = model_name.split(":", 1)[1]
            return cls._build_transformers_model(
                repo_id=repo_id,
                temperature=temperature,
                max_tokens=max_tokens,
                device=device,
                torch_dtype=torch_dtype,
                load_in_4bit=load_in_4bit,
            )

        if model_name.startswith("ollama:"):
            try:
                from pydantic_ai.models.ollama import OllamaModel
                from pydantic_ai.providers.ollama import OllamaProvider
            except ImportError as exc:
                raise ImportError(
                    "Local Ollama support requires `pydantic-ai-slim[openai]`. "
                    "Install it via: pip install 'pydantic-ai-slim[openai]'"
                ) from exc

            ollama_model_name = model_name.split(":", 1)[1]
            resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            return OllamaModel(
                ollama_model_name,
                provider=OllamaProvider(base_url=resolved_base_url),
            )

        if model_name.startswith("huggingface:"):
            if not os.getenv("HF_TOKEN"):
                logger.warning(
                    "HF_TOKEN environment variable is not set; Hugging Face Inference requests may fail."
                )
            if provider_name:
                try:
                    from pydantic_ai.models.huggingface import HuggingFaceModel
                    from pydantic_ai.providers.huggingface import HuggingFaceProvider
                except ImportError as exc:
                    raise ImportError(
                        "Hugging Face support requires `pydantic-ai-slim[huggingface]`. "
                        "Install it via: pip install 'pydantic-ai-slim[huggingface]'"
                    ) from exc

                hf_repo_id = model_name.split(":", 1)[1]
                return HuggingFaceModel(
                    hf_repo_id,
                    provider=HuggingFaceProvider(provider_name=provider_name),
                )

        return model_name

    @staticmethod
    def _format_log_snippet(user_prompt: Union[str, Sequence[Any]]) -> str:
        """Safely builds a short debug preview for either text or multimodal prompts."""
        if isinstance(user_prompt, str):
            return user_prompt[:50]
        parts_summary = []
        for part in user_prompt:
            if isinstance(part, str):
                parts_summary.append(f"text('{part[:35]}...')")
            else:
                parts_summary.append(type(part).__name__)
        return f"[{', '.join(parts_summary)}]"

    # -----------------------------------------------------------------------
    # Human-Readable Tool Call & Tool Result Formatters
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_tool_args(part: ToolCallPart) -> dict[str, Any]:
        """Safely extracts a dictionary of arguments from a PydanticAI `ToolCallPart`."""
        if hasattr(part, "args_as_dict"):
            try:
                res = part.args_as_dict()
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        if isinstance(part.args, dict):
            return part.args
        if isinstance(part.args, str) and part.args.strip():
            try:
                parsed = json.loads(part.args)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"raw": part.args}
        return {}

    @staticmethod
    def format_tool_call_message(tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
        """Converts a raw tool invocation into a clear `(title, detail)` pair for the UI stream."""
        lower_name = tool_name.lower()

        if "flight" in lower_name:
            origin = args.get("origin_airport") or args.get("origin") or "origin airport"
            dest = args.get("destination") or args.get("destination_airport") or "destination"
            dep = args.get("departure_date") or args.get("start_date")
            ret = args.get("return_date") or args.get("end_date")
            adults = args.get("adults", 1)
            traveler_label = "1 passenger" if adults == 1 else f"{adults} passengers"
            date_clause = f" ({dep} → {ret})" if dep and ret else (f" departing {dep}" if dep else "")
            stops_clause = (
                f", max {args['max_stops']} stop(s)" if args.get("max_stops") is not None else ""
            )
            return (
                f"Searching Flights: {origin} → {dest}",
                f"Looking for round-trip flights for {traveler_label} from {origin} to {dest}{date_clause}{stops_clause}...",
            )

        if "tour" in lower_name and "departure" not in lower_name:
            keywords = args.get("keywords") or args.get("destination_slug_keywords") or []
            kw_str = ", ".join(str(k) for k in keywords) if isinstance(keywords, list) else str(keywords)
            filters = []
            if args.get("min_days") and args.get("max_days"):
                filters.append(f"{args['min_days']}-{args['max_days']} days")
            elif args.get("min_days"):
                filters.append(f"at least {args['min_days']} days")
            elif args.get("max_days"):
                filters.append(f"up to {args['max_days']} days")
            if args.get("max_budget_eur"):
                filters.append(f"budget ≤ €{args['max_budget_eur']}")
            filter_suffix = f" ({', '.join(filters)})" if filters else ""
            target_desc = f"'{kw_str}'" if kw_str else "matching criteria"
            return (
                "Searching WeRoad Itineraries",
                f"Looking for WeRoad group tours matching {target_desc}{filter_suffix}...",
            )

        if "departure" in lower_name:
            slug = args.get("slug") or args.get("tour_slug") or "itinerary"
            readable_slug = str(slug).replace("-", " ").title()
            earliest = args.get("earliest_date")
            latest = args.get("latest_date")
            window = ""
            if earliest and latest:
                window = f" between {earliest} and {latest}"
            elif earliest:
                window = f" from {earliest}"
            elif latest:
                window = f" up to {latest}"
            return (
                f"Checking Dates: {readable_slug}",
                f"Checking live WeRoad departure slots ('turni') for '{readable_slug}'{window}...",
            )

        compact_args = ", ".join(f"{k}={v}" for k, v in list(args.items())[:4])
        return (
            f"Running Tool: {tool_name}",
            f"Executing `{tool_name}` ({compact_args})..." if compact_args else f"Executing `{tool_name}`...",
        )

    @staticmethod
    def format_tool_result_message(tool_name: str, content: Any) -> tuple[str, str]:
        """Builds a concise human-readable summary from a `ToolReturnPart` payload."""
        lower_name = tool_name.lower()
        parsed: Any = content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = content

        if isinstance(parsed, list):
            count = len(parsed)
            if "flight" in lower_name:
                prices = [
                    item.get("price_eur")
                    for item in parsed
                    if isinstance(item, dict) and item.get("price_eur") is not None
                ]
                price_note = f" starting from €{min(prices):,.0f}" if prices else ""
                return (
                    "Skyscanner Flights Retrieved",
                    f"Found {count} round-trip flight option{'s' if count != 1 else ''}{price_note}.",
                )
            if "departure" in lower_name:
                available = [
                    item
                    for item in parsed
                    if isinstance(item, dict) and str(item.get("status", "")).lower() != "sold out"
                ]
                return (
                    "Departure Dates Retrieved",
                    f"Found {count} scheduled date{'s' if count != 1 else ''} ({len(available)} open/confirmed).",
                )
            if "tour" in lower_name:
                titles = [
                    item.get("title")
                    for item in parsed[:3]
                    if isinstance(item, dict) and item.get("title")
                ]
                preview = f": {', '.join(titles)}" if titles else ""
                return (
                    "WeRoad Catalog Matches Found",
                    f"Discovered {count} matching WeRoad itinerar{'ies' if count != 1 else 'y'}{preview}.",
                )

        snippet = str(parsed).strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        return (
            f"Completed: {tool_name}",
            snippet or f"Tool `{tool_name}` finished.",
        )

    async def run(self, user_prompt: Union[str, Sequence[Any]], **kwargs) -> Any:
        """
        Executes the agent run, records OpenTelemetry span & LLM token usage metrics,
        and extracts the typed or multimodal output payload.
        """
        snippet = self._format_log_snippet(user_prompt)
        logger.debug(f"Executing agent [{self.model_name}] ({self.stage_name}) with prompt: {snippet}")
        limits = kwargs.pop("usage_limits", UsageLimits(request_limit=self.request_limit))

        start_time = time.perf_counter()
        with _tracer.start_as_current_span(f"agent.{self.stage_name}") as span:
            if span is not None and hasattr(span, "set_attribute"):
                span.set_attribute("gen_ai.request.model", self.model_name)
                if self.provider_name:
                    span.set_attribute("gen_ai.request.provider", self.provider_name)
                span.set_attribute("weroad.stage", self.stage_name)
                span.set_attribute("weroad.prompt.preview", snippet)
            try:
                try:
                    result = await self.agent.run(user_prompt, usage_limits=limits, **kwargs)
                except Exception as run_exc:
                    if self.mcp_config_path and "Connection closed" in str(run_exc):
                        logger.warning(
                            f"MCP connection closed during [{self.stage_name}] "
                            f"(config={self.mcp_config_path}); reloading MCP toolsets and retrying once..."
                        )
                        self.toolsets = self._build_mcp_toolsets()
                        self.agent._user_toolsets = self.toolsets  # type: ignore[attr-defined]
                        result = await self.agent.run(user_prompt, usage_limits=limits, **kwargs)
                    else:
                        raise

                duration = time.perf_counter() - start_time
                usage = self._extract_usage(result)
                self.last_usage = usage
                record_llm_usage(
                    stage=self.stage_name,
                    model_name=self.model_name,
                    usage=usage,
                    duration_seconds=duration,
                    status="ok",
                )
                logger.info(
                    f"Agent [{self.stage_name}] completed in {duration:.2f}s | "
                    f"tokens(in={usage['input_tokens']}, out={usage['output_tokens']}, total={usage['total_tokens']})"
                )
                if hasattr(result, "output"):
                    self.last_output = result.output
                    return result.output
                return None
            except Exception as exc:
                duration = time.perf_counter() - start_time
                record_llm_usage(
                    stage=self.stage_name,
                    model_name=self.model_name,
                    usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 1},
                    duration_seconds=duration,
                    status=f"error: {exc}",
                )
                raise

    async def run_stream(
        self,
        user_prompt: Union[str, Sequence[Any]],
        **kwargs,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        Executes the agent via PydanticAI's `agent.iter()` graph runner and yields
        real-time `AgentStreamEvent` objects for intermediate model thoughts,
        tool calls, and tool results. Stores final output in `self.last_output`.
        """
        snippet = self._format_log_snippet(user_prompt)
        limits = kwargs.pop("usage_limits", UsageLimits(request_limit=self.request_limit))
        self.last_output = None

        start_time = time.perf_counter()
        output_tool_names = {"final_result"}

        with _tracer.start_as_current_span(f"agent.{self.stage_name}.stream") as span:
            if span is not None and hasattr(span, "set_attribute"):
                span.set_attribute("gen_ai.request.model", self.model_name)
                span.set_attribute("weroad.stage", self.stage_name)
                span.set_attribute("weroad.prompt.preview", snippet)

            try:
                step_counter = 0

                async def _iterate_once() -> AsyncIterator[AgentStreamEvent]:
                    nonlocal step_counter
                    async with self.agent.iter(user_prompt, usage_limits=limits, **kwargs) as agent_run:
                        async for node in agent_run:
                            if Agent.is_model_request_node(node):
                                req = getattr(node, "request", None)
                                req_parts = getattr(req, "parts", []) if req is not None else []
                                had_tool_return = False
                                for part in req_parts:
                                    if isinstance(part, ToolReturnPart):
                                        if part.tool_name in output_tool_names:
                                            continue
                                        had_tool_return = True
                                        res_title, res_detail = self.format_tool_result_message(
                                            part.tool_name, part.content
                                        )
                                        yield AgentStreamEvent(
                                            step_type=StreamStepType.TOOL_RESULT,
                                            stage=self.stage_name,
                                            title=res_title,
                                            detail=res_detail,
                                            tool_name=part.tool_name,
                                        )

                                step_counter += 1
                                if had_tool_return:
                                    yield AgentStreamEvent(
                                        step_type=StreamStepType.REASONING,
                                        stage=self.stage_name,
                                        title="Analyzing Tool Results",
                                        detail="Evaluating retrieved tour dates and flight options to decide the next step...",
                                    )
                                elif step_counter == 1:
                                    yield AgentStreamEvent(
                                        step_type=StreamStepType.REASONING,
                                        stage=self.stage_name,
                                        title="Planning Search Strategy",
                                        detail="Determining which WeRoad itineraries, departure slots, and Skyscanner routes to query...",
                                    )

                            elif Agent.is_call_tools_node(node):
                                model_resp = getattr(node, "model_response", None)
                                resp_parts = getattr(model_resp, "parts", []) if model_resp is not None else []
                                for part in resp_parts:
                                    if ThinkingPart is not None and isinstance(part, ThinkingPart):
                                        thought_text = (part.content or "").strip()
                                        if thought_text:
                                            yield AgentStreamEvent(
                                                step_type=StreamStepType.REASONING,
                                                stage=self.stage_name,
                                                title="Agent Reasoning",
                                                detail=thought_text,
                                            )
                                    elif isinstance(part, TextPart):
                                        text_thought = (part.content or "").strip()
                                        if text_thought and not text_thought.startswith("{"):
                                            yield AgentStreamEvent(
                                                step_type=StreamStepType.REASONING,
                                                stage=self.stage_name,
                                                title="Agent Reasoning",
                                                detail=text_thought,
                                            )
                                    elif isinstance(part, ToolCallPart):
                                        if part.tool_name in output_tool_names:
                                            yield AgentStreamEvent(
                                                step_type=StreamStepType.REASONING,
                                                stage=self.stage_name,
                                                title="Synthesizing Final Proposal",
                                                detail="Organizing matching WeRoad tours, verified departure dates, and Skyscanner flights into your itinerary...",
                                            )
                                            continue

                                        args_dict = self._extract_tool_args(part)
                                        call_title, call_detail = self.format_tool_call_message(
                                            part.tool_name, args_dict
                                        )
                                        yield AgentStreamEvent(
                                            step_type=StreamStepType.TOOL_CALL,
                                            stage=self.stage_name,
                                            title=call_title,
                                            detail=call_detail,
                                            tool_name=part.tool_name,
                                            tool_args=args_dict,
                                        )

                        run_result = agent_run.result
                        duration = time.perf_counter() - start_time
                        usage = self._extract_usage(run_result)
                        self.last_usage = usage
                        record_llm_usage(
                            stage=self.stage_name,
                            model_name=self.model_name,
                            usage=usage,
                            duration_seconds=duration,
                            status="ok",
                        )
                        if run_result is not None and hasattr(run_result, "output"):
                            self.last_output = run_result.output

                try:
                    async for ev in _iterate_once():
                        yield ev
                except Exception as run_exc:
                    if self.mcp_config_path and "Connection closed" in str(run_exc):
                        logger.warning(
                            f"MCP connection closed during streaming [{self.stage_name}]; reloading MCP toolsets and retrying once..."
                        )
                        yield AgentStreamEvent(
                            step_type=StreamStepType.STATUS,
                            stage=self.stage_name,
                            title="Reconnecting Tool Server",
                            detail="Re-establishing connection to WeRoad & Skyscanner tool server...",
                        )
                        self.toolsets = self._build_mcp_toolsets()
                        self.agent._user_toolsets = self.toolsets  # type: ignore[attr-defined]
                        async for ev in _iterate_once():
                            yield ev
                    else:
                        raise

            except Exception as exc:
                duration = time.perf_counter() - start_time
                record_llm_usage(
                    stage=self.stage_name,
                    model_name=self.model_name,
                    usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 1},
                    duration_seconds=duration,
                    status=f"error: {exc}",
                )
                raise