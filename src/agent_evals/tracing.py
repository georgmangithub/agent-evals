"""Tracing: record every model and tool call in a run, with tokens, time, and estimated cost."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Self

# List prices in USD per 1M tokens (input, output). Free-tier use costs nothing; these estimate
# what a run would cost on a paid plan. Checked 2026-11-01 on Google's pricing pages: requests to
# gemini-3.5-flash are routed to Gemini 3.6 Flash, so prices may change. Update when they do.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (1.50, 7.50),
}
LOCAL_MODEL_PREFIXES = ("qwen", "llama", "mistral", "gemma", "phi")  # Ollama models: free


@dataclass
class Step:
    kind: Literal["model", "tool"]
    name: str  # the model name, or the tool name
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # model replies read from the cache cost nothing and use no quota
    error: bool = False  # the tool returned an error (including a guardrail denial)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated list-price cost, 0 for local models, None if we don't know the price."""
    if model.startswith(LOCAL_MODEL_PREFIXES):
        return 0.0
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    @property
    def model_calls(self) -> int:
        return sum(s.kind == "model" for s in self.steps)

    @property
    def api_requests(self) -> int:
        """Model calls that really went to the provider (not served from the cache)."""
        return sum(s.kind == "model" and not s.cached for s in self.steps)

    @property
    def tool_calls(self) -> int:
        return sum(s.kind == "tool" for s in self.steps)

    @property
    def tokens(self) -> tuple[int, int]:
        """(input, output) tokens over all model calls, cached or not."""
        models = [s for s in self.steps if s.kind == "model"]
        return sum(s.input_tokens for s in models), sum(s.output_tokens for s in models)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    def cost(self) -> float | None:
        """Estimated cost of the API requests actually made; cached replies are free."""
        total = 0.0
        for s in self.steps:
            if s.kind == "model" and not s.cached:
                c = cost_usd(s.name, s.input_tokens, s.output_tokens)
                if c is None:
                    return None
                total += c
        return total

    def summary(self) -> str:
        tokens_in, tokens_out = self.tokens
        cost = self.cost()
        return (
            f"{self.model_calls} model calls ({self.api_requests} API requests), "
            f"{self.tool_calls} tool calls, {tokens_in} in / {tokens_out} out tokens, "
            f"{self.seconds:.1f}s, " + (f"${cost:.4f}" if cost is not None else "cost unknown")
        )


class Timer:
    """Measure a block: `with Timer() as t: ...` then read t.seconds."""

    def __enter__(self) -> Self:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self.start


def write_trace(path: Path, record: dict, trace: Trace) -> None:
    """Append one run as a JSON line: who/what ran (record) plus every step and the totals."""
    tokens_in, tokens_out = trace.tokens
    line = {
        **record,
        "api_requests": trace.api_requests,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "seconds": round(trace.seconds, 3),
        "cost_usd": trace.cost(),
        "steps": [asdict(s) for s in trace.steps],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
