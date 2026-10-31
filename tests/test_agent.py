import pytest

from agent_evals.agent import loop
from agent_evals.agent.loop import run_agent
from agent_evals.llm import LLMError, LLMResponse, ToolCall, TransientError, Usage


class ScriptedLLM:
    """Returns (or raises) the scripted items in order, like a model with fixed replies."""

    model = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def complete(self, messages, tools=None, *, temperature=0.0):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(text="", calls=()):
    return LLMResponse(text, list(calls), Usage(), "fake", 0.0)


async def echo_tool(name, arguments):
    return f"{name} ran"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        pass

    monkeypatch.setattr(loop.asyncio, "sleep", instant)


async def test_tool_then_answer():
    llm = ScriptedLLM([reply(calls=[ToolCall("c1", "search", {})]), reply("done")])
    result = await run_agent(llm, "task", [], echo_tool)
    assert result.stop_reason == "answer" and result.answer == "done"
    tool_msg = result.messages[3]
    assert tool_msg.role == "tool" and tool_msg.tool_call_id == "c1"
    assert tool_msg.content == "search ran"


async def test_transient_error_is_retried():
    llm = ScriptedLLM([TransientError("empty reply"), reply("done")])
    result = await run_agent(llm, "task", [], echo_tool)
    assert result.answer == "done" and llm.calls == 2


async def test_persistent_error_stops_and_keeps_conversation():
    first = reply(calls=[ToolCall("c1", "search", {})])
    llm = ScriptedLLM([first, TransientError("down"), TransientError("down"), TransientError("x")])
    result = await run_agent(llm, "task", [], echo_tool, retries=2)
    assert result.stop_reason == "error" and result.answer is None
    assert "x" in result.error
    assert [m.role for m in result.messages] == ["system", "user", "assistant", "tool"]


async def test_hard_error_is_not_retried():
    llm = ScriptedLLM([LLMError("bad request")])
    result = await run_agent(llm, "task", [], echo_tool)
    assert result.stop_reason == "error" and llm.calls == 1


async def test_max_steps():
    llm = ScriptedLLM([reply(calls=[ToolCall(f"c{i}", "search", {})]) for i in range(3)])
    result = await run_agent(llm, "task", [], echo_tool, max_steps=3)
    assert result.stop_reason == "max_steps" and result.answer is None
