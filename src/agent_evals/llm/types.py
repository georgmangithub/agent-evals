"""Provider-neutral message and response types, the client protocol, and errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolSpec:
        return cls(d["name"], d["description"], d["input_schema"])


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(d["id"], d["name"], d.get("arguments", {}))


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            d["role"],
            d.get("content", ""),
            [ToolCall.from_dict(c) for c in d.get("tool_calls", [])],
            d.get("tool_call_id"),
            d.get("name"),
        )


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Usage:
        return cls(d.get("input_tokens", 0), d.get("output_tokens", 0))


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    latency_s: float
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "usage": self.usage.to_dict(),
            "model": self.model,
            "latency_s": self.latency_s,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMResponse:
        return cls(
            d["text"],
            [ToolCall.from_dict(c) for c in d.get("tool_calls", [])],
            Usage.from_dict(d.get("usage", {})),
            d["model"],
            d["latency_s"],
            d.get("cached", False),
        )


class LLMError(Exception):
    """Base class for LLM client failures."""


class RateLimitError(LLMError):
    """Provider rate limit or quota hit (HTTP 429)."""


class TransientError(LLMError):
    """Retryable failure: 5xx, network error or timeout."""


class LLMClient(Protocol):
    model: str

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.0,
    ) -> LLMResponse: ...
