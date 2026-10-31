"""Step 1: send a task plus one tool to the model and look at what comes back."""

import asyncio

from agent_evals.llm import Message, ToolSpec, make_client

add_tool = ToolSpec(
    name="add",
    description="Add two numbers.",
    input_schema={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
)


async def main():
    llm = make_client("ollama:qwen2.5:3b")
    messages = [Message(role="user", content="What is 17 + 25? Use the add tool.")]
    resp = await llm.complete(messages, tools=[add_tool])
    print("text:      ", repr(resp.text))
    print("tool_calls:", resp.tool_calls)
    print("cached:    ", resp.cached, f"({resp.latency_s:.2f}s)")


asyncio.run(main())
