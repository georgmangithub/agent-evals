"""On-disk response cache and make_client(), which builds a client from a spec."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from .providers import GeminiClient, OllamaClient
from .types import LLMClient, LLMError, LLMResponse, Message, ToolSpec

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_cache_dir() -> Path:
    """AGENT_EVALS_CACHE_DIR if set, else <repo>/.cache/llm."""
    env = os.environ.get("AGENT_EVALS_CACHE_DIR")
    return Path(env) if env else _PROJECT_ROOT / ".cache" / "llm"


def cache_key(
    model: str, messages: list[Message], tools: list[ToolSpec] | None, temperature: float
) -> str:
    payload = {
        "model": model,
        "messages": [m.to_dict() for m in messages],
        "tools": [t.to_dict() for t in tools or []],
        "temperature": float(temperature),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedClient:
    def __init__(self, inner: LLMClient, cache_dir: str | Path | None = None):
        self.inner = inner
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()

    @property
    def model(self) -> str:
        return self.inner.model

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        path = self.cache_dir / f"{cache_key(self.model, messages, tools, temperature)}.json"
        start = time.perf_counter()
        try:
            resp = LLMResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        else:
            resp.cached = True
            resp.latency_s = time.perf_counter() - start
            return resp
        resp = await self.inner.complete(messages, tools, temperature=temperature)
        try:
            self._write(path, resp)
        except OSError as e:
            log.warning("LLM cache write failed for %s: %s", path.name, e)
        return resp

    def _write(self, path: Path, resp: LLMResponse) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(resp.to_dict(), f)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


DEFAULT_MODEL = "gemini:gemini-3.5-flash"


def _build(spec: str) -> LLMClient:
    provider, _, model = spec.partition(":")
    if provider == "gemini":
        return GeminiClient(model or "gemini-3.5-flash")
    if provider == "ollama":
        return OllamaClient(model or "qwen2.5:3b")
    raise LLMError(f"unknown provider {provider!r} in spec {spec!r}")


def make_client(spec: str | None = None, *, cache: bool = True) -> LLMClient:
    """Return the client for `spec` (default: env AGENT_EVALS_MODEL), wrapped in a cache."""
    load_dotenv()
    spec = spec or os.environ.get("AGENT_EVALS_MODEL", DEFAULT_MODEL)
    client = _build(spec)
    return CachedClient(client) if cache else client
