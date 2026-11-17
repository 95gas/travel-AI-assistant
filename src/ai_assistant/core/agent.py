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
from typing import Optional, Any, Sequence, Union

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
    from pydantic_ai.profiles import ModelProfile
except ImportError:  # pragma: no cover
    ModelProfile = None  # type: ignore[assignment,misc]

from src.ai_assistant.schemas.schemas import LLMConfig
from src.ai_assistant.observability.observability import extract_usage_from_result, get_tracer, record_llm_usage

_tracer = get_tracer("weroad.agent")
_TRANSFORMERS_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}
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
    - Local Hugging Face `transformers` checkpoints (`transformers:google/gemma-3-1b-it`, etc.)
    - Local Ollama servers (`ollama:gemma4:12b`, `ollama:gemma4:e4b`, etc.)
    - Remote Hugging Face Inference Providers (`huggingface:google/gemma-3-27b-it`, etc.)
    - Structured Pydantic output types, MCP toolsets, and OpenTelemetry instrumentation
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
        self.temperature = llm_settings.temperature
        self.max_tokens = llm_settings.max_tokens
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
        )

        agent_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "system_prompt": self.system_prompt,
            "model_settings": ModelSettings(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
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

    def _build_mcp_toolsets(self) -> Any:
        """
        Loads MCP toolsets from `self.mcp_config_path`, normalizing `python`/`python3`
        commands to `sys.executable` and injecting `PYTHONPATH` so stdio MCP subprocesses
        on Windows run inside the active virtual environment without failing on startup.
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
            )

    @staticmethod
    def _load_transformers_pipeline(
        repo_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> tuple[Any, Any]:
        """Loads and caches a Hugging Face `transformers` tokenizer and causal LM."""
        norm_device = device or "auto"
        norm_dtype = torch_dtype or "auto"
        cache_key = (repo_id, norm_device, norm_dtype)

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

            has_accelerate = importlib.util.find_spec("accelerate") is not None
            logger.info(
                f"Loading local Hugging Face transformers model: {repo_id} "
                f"(device={norm_device}, dtype={norm_dtype}, accelerate={has_accelerate})..."
            )
            load_start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(repo_id)

            if has_accelerate:
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
        local models (e.g., Gemma 1B/4B) populate field values instead of echoing `"properties": {...}`.
        """
        if root_schema is None:
            root_schema = schema
        if depth > 3 or not isinstance(schema, dict):
            return None

        # Resolve local $ref pointers (e.g. "#/$defs/MySubModel")
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

        # Unwrap Optional[...] / Union[...] branches to pick the concrete non-null schema
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
    ) -> FunctionModel:
        """
        Wraps a local Hugging Face `transformers` causal LM in PydanticAI's `FunctionModel`
        so it supports multi-turn messages, MCP/function tool calling, and structured outputs.
        """

        async def _run_local_transformers(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tokenizer, model = await asyncio.to_thread(
                cls._load_transformers_pipeline, repo_id, device, torch_dtype
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

            # Gemma chat templates require strict user/assistant alternation without a `system` role;
            # prepend combined system instructions to the first user message.
            if system_chunks:
                combined_system = "\n\n".join(system_chunks).strip()
                if chat_messages and chat_messages[0]["role"] == "user":
                    chat_messages[0]["content"] = f"{combined_system}\n\n{chat_messages[0]['content']}"
                else:
                    chat_messages.insert(0, {"role": "user", "content": combined_system})

            # Merge consecutive messages of the same role so Gemma's chat template alternation never fails
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
                # 1. Check if the model emitted a function/MCP tool call
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

                # 2. Check if PydanticAI expects a structured Pydantic output tool
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

                # 3. If PydanticAI is using PromptedOutput (`allow_text_output=True`), return clean JSON text
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
    ) -> Any:
        if model_name.startswith("transformers:"):
            repo_id = model_name.split(":", 1)[1]
            return cls._build_transformers_model(
                repo_id=repo_id,
                temperature=temperature,
                max_tokens=max_tokens,
                device=device,
                torch_dtype=torch_dtype,
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
                usage = extract_usage_from_result(result)
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