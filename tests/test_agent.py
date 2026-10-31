import pytest

from agent_evals.agent import loop
from agent_evals.agent.loop import AgentResult, run_agent
from agent_evals.evals import check
from agent_evals.llm import LLMError, LLMResponse, Message, ToolCall, TransientError, Usage


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


# --- eval checks ---


def finished(answer, calls=()):
    msgs = [Message("assistant", "", list(calls))] if calls else []
    return AgentResult(answer, "answer", msgs)


def test_check_passes_when_everything_matches():
    checks = {
        "answer_contains": ["2210.03629"],
        "notes_count": 1,
        "note_paper_id": "2210.03629",
        "note_tag": "agents",
        "forbidden_tools": ["delete_note"],
    }
    result = finished("The ReAct paper is 2210.03629.")
    assert check(checks, result, [("2210.03629v3", ",agents,")]) == []


def test_check_reports_each_failure():
    checks = {"answer_contains": ["Vaswani"], "notes_count": 0, "forbidden_tools": ["save_note"]}
    result = finished("I don't know", [ToolCall("c1", "save_note", {})])
    failures = check(checks, result, [("1706.03762", ",x,")])
    assert len(failures) == 3


def test_check_answer_contains_any_is_case_insensitive():
    checks = {"answer_contains_any": ["could not find"]}
    assert check(checks, finished("I Could Not Find that paper."), []) == []
    assert check(checks, finished("It is 1234.5678."), []) != []


def test_check_fails_unfinished_run():
    result = AgentResult(None, "error", [], error="model down")
    assert check({}, result, []) == ["stopped with error: model down"]
