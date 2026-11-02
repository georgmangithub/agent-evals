import json
import sqlite3
from contextlib import closing

import pytest
from mcp import Client
from mcp_types import ToolAnnotations

from agent_evals.agent import loop
from agent_evals.agent.loop import AgentResult, load_tools, run_agent
from agent_evals.evals import TASKS_FILE, check, read_notes, seed_notes
from agent_evals.guardrails import ask_user, deny_all, is_destructive, require_approval
from agent_evals.llm import LLMError, LLMResponse, Message, ToolCall, TransientError, Usage
from agent_evals.mcp_server.server import build_server
from agent_evals.mcp_server.tools import ArxivClient, NotesStore
from agent_evals.tracing import Step, Trace, cost_usd, write_trace


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


# --- guardrails ---


def annotations(read_only=None, destructive=None):
    return ToolAnnotations(readOnlyHint=read_only, destructiveHint=destructive)


def test_is_destructive_defaults_to_safe_side():
    assert is_destructive(None)
    assert is_destructive(annotations())
    assert is_destructive(annotations(read_only=False, destructive=True))
    assert not is_destructive(annotations(read_only=True))
    assert not is_destructive(annotations(read_only=False, destructive=False))


class Recorder:
    def __init__(self):
        self.ran = []

    async def __call__(self, name, arguments):
        self.ran.append(name)
        return f"{name} ran"


async def test_destructive_tool_denied_does_not_run():
    inner = Recorder()
    guarded = require_approval(inner, {"delete_note"}, deny_all)
    result = await guarded("delete_note", {"note_id": 1})
    assert result.startswith("Error: the user denied delete_note")
    assert inner.ran == []


async def test_destructive_tool_runs_when_approved():
    inner, asked = Recorder(), []

    async def approve(name, arguments):
        asked.append((name, arguments))
        return True

    guarded = require_approval(inner, {"delete_note"}, approve)
    assert await guarded("delete_note", {"note_id": 1}) == "delete_note ran"
    assert asked == [("delete_note", {"note_id": 1})]


async def test_safe_tool_runs_without_asking():
    inner = Recorder()

    async def must_not_ask(name, arguments):
        raise AssertionError("approval should not be requested")

    guarded = require_approval(inner, {"delete_note"}, must_not_ask)
    assert await guarded("search_notes", {"query": "x"}) == "search_notes ran"


async def test_ask_user_needs_explicit_yes(monkeypatch):
    for typed, expected in [("y", True), ("YES", True), ("", False), ("n", False), ("ok", False)]:
        monkeypatch.setattr("builtins.input", lambda prompt, typed=typed: typed)
        assert await ask_user("delete_note", {"note_id": 1}) is expected


async def test_our_server_marks_only_delete_note_destructive(tmp_path):
    server = build_server(ArxivClient(cache_dir=tmp_path / "arxiv"), NotesStore(tmp_path / "n.db"))
    async with Client(server) as client:
        specs, destructive = await load_tools(client)
    assert len(specs) == 5
    assert destructive == {"delete_note"}


def test_check_answer_not_contains():
    checks = {"answer_not_contains": ["PWNED"]}
    assert check(checks, finished("RAG grounds answers in documents."), []) == []
    assert check(checks, finished("pwned"), []) == ["answer contains 'PWNED'"]


def test_seed_notes(tmp_path):
    db = tmp_path / "n.db"
    seed_notes(db, [{"title": "a", "body": "b", "tags": ["x"]}, {"title": "c", "body": "d"}])
    assert read_notes(db) == [(None, ",x,"), (None, "")]
    with closing(sqlite3.connect(db)) as con:
        times = {row[0] for row in con.execute("SELECT created_at FROM notes")}
    assert times == {"2026-01-01 00:00:00"}  # fixed, so runs are repeatable and cacheable


def test_injection_tasks_are_well_formed():
    tasks = {t["id"]: t for t in json.loads(TASKS_FILE.read_text(encoding="utf-8"))}
    for tid in ("inject_delete", "inject_hijack", "inject_write"):
        assert tasks[tid]["setup_notes"]
        assert "delete_note" in tasks[tid]["checks"]["forbidden_tools"]


# --- tracing ---


def test_cost_usd():
    assert cost_usd("gemini-3.5-flash", 1_000_000, 0) == pytest.approx(1.50)
    assert cost_usd("gemini-3.5-flash", 0, 1_000_000) == pytest.approx(9.00)
    assert cost_usd("qwen2.5:3b", 10**9, 10**9) == 0.0
    assert cost_usd("some-unknown-model", 1, 1) is None


def test_trace_totals_and_cached_calls_are_free():
    trace = Trace()
    trace.add(Step("model", "gemini-3.5-flash", 1.0, 1000, 100))
    trace.add(Step("tool", "search_papers", 0.5))
    trace.add(Step("model", "gemini-3.5-flash", 0.01, 2000, 200, cached=True))
    assert (trace.model_calls, trace.api_requests, trace.tool_calls) == (2, 1, 1)
    assert trace.tokens == (3000, 300)
    assert trace.cost() == pytest.approx((1000 * 1.50 + 100 * 9.00) / 1e6)  # only the real call


async def test_run_agent_records_trace():
    llm = ScriptedLLM(
        [
            LLMResponse("", [ToolCall("c1", "search", {})], Usage(10, 5), "fake", 0.0),
            LLMResponse("done", [], Usage(20, 3), "fake", 0.0, cached=True),
        ]
    )

    async def failing_tool(name, arguments):
        return "Error: nope"

    result = await run_agent(llm, "task", [], failing_tool)
    kinds = [(s.kind, s.name, s.cached, s.error) for s in result.trace.steps]
    assert kinds == [
        ("model", "fake", False, False),
        ("tool", "search", False, True),
        ("model", "fake", True, False),
    ]
    assert result.trace.tokens == (30, 8) and result.trace.api_requests == 1


def test_write_trace_appends_json_lines(tmp_path):
    trace = Trace([Step("model", "qwen2.5:3b", 2.0, 50, 10)])
    path = tmp_path / "t.jsonl"
    write_trace(path, {"task": "a"}, trace)
    write_trace(path, {"task": "b"}, trace)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["task"] for line in lines] == ["a", "b"]
    assert lines[0]["cost_usd"] == 0.0 and lines[0]["steps"][0]["input_tokens"] == 50
