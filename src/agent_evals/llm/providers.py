"""Model providers: Gemini (google-genai SDK) and Ollama (/api/chat)."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from .types import (
    LLMError,
    LLMResponse,
    Message,
    RateLimitError,
    ToolCall,
    ToolSpec,
    TransientError,
    Usage,
)

_UNSUPPORTED_KEYS = {"additionalProperties", "$schema", "title", "$id"}
_DATA_KEYS = {"default", "enum", "const", "examples"}
_NAMED_SCHEMA_KEYS = ("properties", "$defs", "definitions")


def clean_schema(schema: Any) -> Any:
    """Recursively drop JSON-schema keys Gemini rejects."""
    if isinstance(schema, dict):
        out = {
            k: v if k in _DATA_KEYS else clean_schema(v)
            for k, v in schema.items()
            if k not in _UNSUPPORTED_KEYS
        }
        for key in _NAMED_SCHEMA_KEYS:  # names under these are user-defined, not keywords
            if isinstance(schema.get(key), dict):
                out[key] = {k: clean_schema(v) for k, v in schema[key].items()}
        return out
    if isinstance(schema, list):
        return [clean_schema(v) for v in schema]
    return schema


def _all_responses(content: types.Content) -> bool:
    return all(p.function_response is not None for p in content.parts or [])


def to_gemini_contents(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
    """Return (system_instruction, contents). Consecutive tool results share one turn."""
    system: list[str] = []
    contents: list[types.Content] = []
    call_names: dict[str, str] = {}
    for m in messages:
        if m.role == "system":
            system.append(m.content)
        elif m.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
        elif m.role == "assistant":
            parts = [types.Part(text=m.content)] if m.content else []
            call_names.update({c.id: c.name for c in m.tool_calls})
            parts += [
                types.Part(
                    function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments),
                    thought_signature=base64.b64decode(c.signature) if c.signature else None,
                )
                for c in m.tool_calls
            ]
            contents.append(types.Content(role="model", parts=parts))
        else:
            name = m.name or call_names.get(m.tool_call_id or "")
            if not name:
                raise LLMError(f"cannot resolve tool name for tool_call_id {m.tool_call_id!r}")
            part = types.Part(
                function_response=types.FunctionResponse(
                    id=m.tool_call_id, name=name, response={"result": m.content}
                )
            )
            if contents and contents[-1].role == "user" and _all_responses(contents[-1]):
                contents[-1].parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
    return ("\n\n".join(system) or None), contents


def to_gemini_tools(tools: list[ToolSpec] | None) -> list[types.Tool] | None:
    if not tools:
        return None
    decls = [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters_json_schema=clean_schema(t.input_schema),
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=decls)]


def parse_gemini_response(resp: Any, model: str, latency_s: float) -> LLMResponse:
    """Convert a GenerateContentResponse; raise on blocked, truncated or empty output."""
    block = resp.prompt_feedback.block_reason if resp.prompt_feedback else None
    if block:
        raise LLMError(f"prompt blocked: {block.name}")
    if not resp.candidates:
        raise TransientError("Gemini returned no candidates")
    cand = resp.candidates[0]
    reason = cand.finish_reason.name if cand.finish_reason else None
    if reason in ("MALFORMED_FUNCTION_CALL", "OTHER"):
        raise TransientError(f"Gemini stopped with finish_reason {reason}")
    if reason == "MAX_TOKENS":
        raise LLMError("Gemini response truncated (MAX_TOKENS); raise the output limit")
    if reason not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        raise LLMError(f"Gemini stopped with finish_reason {reason}")
    text: list[str] = []
    calls: list[ToolCall] = []
    for p in (cand.content.parts if cand.content else None) or []:
        if p.function_call:
            fc = p.function_call
            call_id = fc.id or f"call_{len(calls)}_{fc.name}"
            sig = base64.b64encode(p.thought_signature).decode() if p.thought_signature else None
            calls.append(ToolCall(call_id, fc.name, dict(fc.args or {}), sig))
        elif p.text and not p.thought:
            text.append(p.text)
    if not text and not calls:
        raise TransientError("Gemini returned an empty response")
    um = resp.usage_metadata
    usage = Usage()
    if um:
        usage = Usage(
            um.prompt_token_count or 0,
            (um.candidates_token_count or 0) + (um.thoughts_token_count or 0),
        )
    return LLMResponse("".join(text), calls, usage, model, latency_s)


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        thinking_level: str = "low",
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("Gemini API key missing: set GEMINI_API_KEY or GOOGLE_API_KEY")
        self.model = model
        self.timeout_s = timeout_s
        self.thinking_level = thinking_level
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        system, contents = to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            tools=to_gemini_tools(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level.upper()),
        )
        start = time.perf_counter()
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.APIError as e:
            if e.code == 429:
                raise RateLimitError(str(e)) from e
            if e.code >= 500 or e.code == 408:
                raise TransientError(str(e)) from e
            raise LLMError(str(e)) from e
        except (httpx.TransportError, TimeoutError) as e:
            raise TransientError(str(e)) from e
        return parse_gemini_response(resp, self.model, time.perf_counter() - start)


def to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
            ]
        if m.role == "tool" and m.name:
            d["tool_name"] = m.name
        out.append(d)
    return out


def to_ollama_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools or []
    ]


def parse_ollama_response(data: dict[str, Any], model: str, latency_s: float) -> LLMResponse:
    msg = data.get("message", {})
    calls = []
    for i, c in enumerate(msg.get("tool_calls") or []):
        fn = c["function"]
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError as e:
                raise TransientError(f"ollama tool arguments are not valid JSON: {args!r}") from e
        if not isinstance(args, dict):
            raise TransientError(f"ollama tool arguments are not an object: {args!r}")
        calls.append(ToolCall(c.get("id") or f"call_{i}_{fn['name']}", fn["name"], args))
    text = msg.get("content", "")
    if data.get("done_reason") == "length":
        raise LLMError("ollama response truncated (done_reason=length)")
    if not text and not calls:
        raise TransientError("ollama returned an empty response")
    usage = Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
    return LLMResponse(text, calls, usage, model, latency_s)


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 300.0,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self._http = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout_s)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama_messages(messages),
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if tools:
            body["tools"] = to_ollama_tools(tools)
        start = time.perf_counter()
        try:
            r = await self._http.post("/api/chat", json=body)
        except httpx.HTTPError as e:
            raise TransientError(f"ollama request failed: {e!r}") from e
        if r.status_code == 429:
            raise RateLimitError(r.text)
        if r.status_code >= 500:
            raise TransientError(f"ollama {r.status_code}: {r.text}")
        if r.status_code >= 400:
            raise LLMError(f"ollama {r.status_code}: {r.text}")
        return parse_ollama_response(r.json(), self.model, time.perf_counter() - start)
