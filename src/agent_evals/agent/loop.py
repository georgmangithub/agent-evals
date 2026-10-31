"""The agent loop, connected to our MCP server's tools."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from agent_evals.llm import LLMClient, Message, ToolSpec, make_client

RunTool = Callable[[str, dict], Awaitable[str]]


async def run_agent(
    llm: LLMClient,
    task: str,
    tools: list[ToolSpec],
    run_tool: RunTool,
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
            result = await run_tool(call.name, call.arguments)
            messages.append(
                Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
            )
    return None, messages


# --- Connecting to the MCP server ---


async def list_tool_specs(client: Client) -> list[ToolSpec]:
    """Ask the server which tools it has and turn them into ToolSpecs for the model."""
    tools = (await client.list_tools()).tools
    return [ToolSpec(t.name, t.description or "", t.input_schema) for t in tools]


def mcp_run_tool(client: Client) -> RunTool:
    """Return a run_tool function that forwards each call to the MCP server."""

    async def run_tool(name: str, arguments: dict) -> str:
        result = await client.call_tool(name, arguments)
        if result.structured_content is not None:
            text = json.dumps(result.structured_content)
        else:
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
        return f"Error: {text}" if result.is_error else text

    return run_tool


async def main():
    task = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "Find the ReAct paper (reasoning and acting in language models) on arXiv, "
            "then save a short note about it with the tag 'agents'."
        )
    )
    llm = make_client()  # Gemini by default; set AGENT_EVALS_MODEL=ollama:qwen2.5:3b to switch
    server = StdioServerParameters(command=sys.executable, args=["-m", "agent_evals.mcp_server"])
    async with Client(server) as client:
        tools = await list_tool_specs(client)
        print("tools:", [t.name for t in tools], "\n")
        answer, messages = await run_agent(llm, task, tools, mcp_run_tool(client))
    for m in messages:
        detail = m.content or [f"{c.name}({c.arguments})" for c in m.tool_calls]
        print(f"{m.role:>9}: {str(detail)[:300]}")
    print("\nfinal answer:", answer)


if __name__ == "__main__":
    asyncio.run(main())
