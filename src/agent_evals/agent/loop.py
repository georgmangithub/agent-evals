"""The agent loop: ask the model, run the tools it requests, repeat until it answers."""

import asyncio
from collections.abc import Callable

from agent_evals.llm import LLMClient, Message, ToolSpec, make_client


async def run_agent(
    llm: LLMClient,
    task: str,
    tools: list[ToolSpec],
    run_tool: Callable[[str, dict], str],
    max_steps: int = 8,
) -> tuple[str | None, list[Message]]:
    """Return (final answer, full conversation). The answer is None if max_steps ran out."""
    messages = [Message(role="user", content=task)]
    for _ in range(max_steps):
        resp = await llm.complete(messages, tools=tools)
        messages.append(Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls))
        if not resp.tool_calls:
            return resp.text, messages
        for call in resp.tool_calls:
            result = run_tool(call.name, call.arguments)
            messages.append(
                Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
            )
    return None, messages


# --- Demo: two math tools, and a task that needs them one after the other ---


def _number_tool(name: str, description: str) -> ToolSpec:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }
    return ToolSpec(name=name, description=description, input_schema=schema)


DEMO_TOOLS = [
    _number_tool("add", "Add two numbers."),
    _number_tool("multiply", "Multiply two numbers."),
]


def demo_run_tool(name: str, arguments: dict) -> str:
    a, b = arguments["a"], arguments["b"]
    if name == "add":
        return str(a + b)
    if name == "multiply":
        return str(a * b)
    return f"Error: unknown tool {name}"


async def main():
    llm = make_client("ollama:qwen2.5:3b")
    task = "What is (17 + 25) * 3? Use the tools, one step at a time."
    answer, messages = await run_agent(llm, task, DEMO_TOOLS, demo_run_tool)
    for m in messages:
        detail = m.content or [f"{c.name}({c.arguments})" for c in m.tool_calls]
        print(f"{m.role:>9}: {detail}")
    print("\nfinal answer:", answer)


if __name__ == "__main__":
    asyncio.run(main())
