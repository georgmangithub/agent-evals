"""Step 2: one full lap. The model asks for a tool, we run it, send the result back, get an answer."""

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


def run_tool(name: str, arguments: dict) -> str:
    """Our side of the deal: actually do what the model asked."""
    if name == "add":
        return str(arguments["a"] + arguments["b"])
    return f"Error: unknown tool {name}"


async def main():
    llm = make_client("ollama:qwen2.5:3b")
    messages = [Message(role="user", content="What is 17 + 25? Use the add tool.")]

    # 1. Ask the model. It replies with a tool call instead of an answer.
    resp = await llm.complete(messages, tools=[add_tool])
    messages.append(Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls))

    # 2. Run each requested tool and add its result, labelled with the request's id.
    for call in resp.tool_calls:
        result = run_tool(call.name, call.arguments)
        messages.append(
            Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
        )

    # 3. Ask again. Now the model has the result and can answer in plain text.
    resp = await llm.complete(messages, tools=[add_tool])
    messages.append(Message(role="assistant", content=resp.text))

    for m in messages:
        detail = m.content or [f"{c.name}({c.arguments})" for c in m.tool_calls]
        print(f"{m.role:>9}: {detail}")


asyncio.run(main())
