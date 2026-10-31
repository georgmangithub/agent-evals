import asyncio
import json
import logging
import socket

import httpx
import pytest
from google.genai import types

from agent_evals.llm import (
    CachedClient,
    GeminiClient,
    LLMError,
    LLMResponse,
    Message,
    OllamaClient,
    RateLimitError,
    ToolCall,
    ToolSpec,
    TransientError,
    Usage,
    make_client,
)
from agent_evals.llm.client import _build, cache_key, default_cache_dir
from agent_evals.llm.providers import (
    clean_schema,
    parse_gemini_response,
    to_gemini_contents,
    to_gemini_tools,
)


class FakeClient:
    """Scripted client: each step is an exception to raise, a delay, or a response."""

    def __init__(self, model="fake", steps=None):
        self.model = model
        self.steps = list(steps or [])
        self.calls = 0

    async def complete(self, messages, tools=None, *, temperature=0.0):
        self.calls += 1
        step = self.steps.pop(0) if self.steps else "ok"
        if isinstance(step, Exception):
            raise step
        if isinstance(step, (int, float)):
            await asyncio.sleep(step)
        return LLMResponse(f"reply from {self.model}", [], Usage(1, 2), self.model, 0.5)


@pytest.fixture
def msgs():
    return [Message("system", "be brief"), Message("user", "hi")]


@pytest.fixture
def tool():
    return ToolSpec(
        "add", "add numbers", {"type": "object", "properties": {"a": {"type": "integer"}}}
    )


@pytest.fixture
def full_message():
    return Message(
        "assistant", "calling", [ToolCall("c1", "add", {"a": 1})], tool_call_id=None, name=None
    )


def test_message_roundtrip(full_message):
    assert Message.from_dict(full_message.to_dict()) == full_message
    tool_msg = Message("tool", "3", tool_call_id="c1", name="add")
    assert Message.from_dict(tool_msg.to_dict()) == tool_msg


def test_message_defaults_not_shared():
    a, b = Message("user"), Message("user")
    a.tool_calls.append(ToolCall("x", "y", {}))
    assert b.tool_calls == []


def test_toolspec_toolcall_usage_roundtrip(tool):
    assert ToolSpec.from_dict(tool.to_dict()) == tool
    call = ToolCall("c", "n", {"k": [1, 2]})
    assert ToolCall.from_dict(call.to_dict()) == call
    assert Usage.from_dict(Usage(3, 4).to_dict()) == Usage(3, 4)


def test_response_roundtrip():
    r = LLMResponse("hi", [ToolCall("c", "n", {})], Usage(1, 2), "m", 0.1, cached=True)
    assert LLMResponse.from_dict(r.to_dict()) == r


def test_key_stable_and_sensitive(msgs, tool):
    k = cache_key("m", msgs, [tool], 0.0)
    assert k == cache_key("m", [Message.from_dict(m.to_dict()) for m in msgs], [tool], 0.0)
    other = ToolSpec("sub", "subtract", tool.input_schema)
    assert k != cache_key("m", msgs, [other], 0.0)
    assert k != cache_key("m", msgs, None, 0.0)
    assert k != cache_key("m2", msgs, [tool], 0.0)
    assert k != cache_key("m", msgs, [tool], 0.7)


def test_key_ignores_schema_dict_order(msgs):
    a = ToolSpec("t", "d", {"a": 1, "b": 2})
    b = ToolSpec("t", "d", {"b": 2, "a": 1})
    assert cache_key("m", msgs, [a], 0.0) == cache_key("m", msgs, [b], 0.0)


async def test_miss_then_hit(tmp_path, msgs):
    inner = FakeClient("fake")
    client = CachedClient(inner, tmp_path / "c")
    first = await client.complete(msgs)
    second = await client.complete(msgs)
    assert inner.calls == 1
    assert not first.cached and second.cached
    assert second.text == first.text and second.usage == first.usage
    assert len(list((tmp_path / "c").glob("*.json"))) == 1
    assert not list((tmp_path / "c").glob("*.tmp"))


async def test_different_input_misses(tmp_path, msgs, tool):
    inner = FakeClient()
    client = CachedClient(inner, tmp_path)
    await client.complete(msgs)
    await client.complete(msgs, [tool])
    assert inner.calls == 2


async def test_corrupt_entry_is_a_miss(tmp_path, msgs):
    inner = FakeClient()
    client = CachedClient(inner, tmp_path)
    await client.complete(msgs)
    (path,) = tmp_path.glob("*.json")
    path.write_text("{not json")
    resp = await client.complete(msgs)
    assert inner.calls == 2 and not resp.cached


async def test_error_is_not_cached(tmp_path, msgs):
    client = CachedClient(FakeClient("p", [TransientError("x")]), tmp_path)
    with pytest.raises(TransientError):
        await client.complete(msgs)
    assert not list(tmp_path.glob("*.json"))


async def test_write_failure_still_returns(tmp_path, msgs, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    client = CachedClient(FakeClient(), blocker / "sub")
    with caplog.at_level(logging.WARNING):
        resp = await client.complete(msgs)
    assert resp.text == "reply from fake"
    assert "cache write failed" in caplog.text


@pytest.mark.parametrize("junk", ["[]", "null", "42", '{"text": 1}'])
async def test_malformed_entry_is_a_miss(tmp_path, msgs, junk):
    inner = FakeClient()
    client = CachedClient(inner, tmp_path)
    await client.complete(msgs)
    (path,) = tmp_path.glob("*.json")
    path.write_text(junk)
    resp = await client.complete(msgs)
    assert inner.calls == 2 and not resp.cached


def test_key_normalizes_temperature(msgs):
    assert cache_key("m", msgs, None, 0) == cache_key("m", msgs, None, 0.0)


def test_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_EVALS_CACHE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    d = default_cache_dir()
    assert d.is_absolute() and d.parts[-2:] == (".cache", "llm")
    assert (d.parents[1] / "pyproject.toml").exists()
    monkeypatch.setenv("AGENT_EVALS_CACHE_DIR", str(tmp_path / "env"))
    assert CachedClient(FakeClient()).cache_dir == tmp_path / "env"
    assert CachedClient(FakeClient(), tmp_path / "x").cache_dir == tmp_path / "x"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("AGENT_EVALS_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("agent_evals.llm.client.load_dotenv", lambda: None)


def test_default_is_cached_gemini():
    c = make_client()
    assert isinstance(c, CachedClient) and isinstance(c.inner, GeminiClient)
    assert c.model == "gemini-3.5-flash"


def test_spec_with_colon_in_model_and_no_cache():
    c = make_client("ollama:qwen2.5:3b", cache=False)
    assert isinstance(c, OllamaClient) and c.model == "qwen2.5:3b"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_EVALS_MODEL", "ollama:llama3.2:1b")
    c = make_client(cache=False)
    assert c.model == "llama3.2:1b"


def test_unknown_provider():
    with pytest.raises(LLMError):
        _build("nope:model")


def test_clean_schema_recursive():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "T",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "object", "title": "X", "additionalProperties": True, "properties": {}},
            "items": {"type": "array", "items": [{"title": "n", "type": "string"}]},
            "title": {"type": "string"},
        },
    }
    out = clean_schema(schema)
    assert out == {
        "type": "object",
        "properties": {
            "x": {"type": "object", "properties": {}},
            "items": {"type": "array", "items": [{"type": "string"}]},
            "title": {"type": "string"},
        },
    }
    assert "$schema" in schema


def test_contents_conversion():
    msgs = [
        Message("system", "sys1"),
        Message("system", "sys2"),
        Message("user", "add 1 and 2, 3 and 4"),
        Message(
            "assistant",
            "",
            [ToolCall("a", "add", {"a": 1, "b": 2}), ToolCall("b", "add", {"a": 3})],
        ),
        Message("tool", "3", tool_call_id="a", name="add"),
        Message("tool", "7", tool_call_id="b", name="add"),
        Message("assistant", "done"),
    ]
    system, contents = to_gemini_contents(msgs)
    assert system == "sys1\n\nsys2"
    assert [c.role for c in contents] == ["user", "model", "user", "model"]
    assert contents[0].parts[0].text == "add 1 and 2, 3 and 4"
    calls = [p.function_call for p in contents[1].parts]
    assert (calls[0].id, calls[0].name, calls[0].args) == ("a", "add", {"a": 1, "b": 2})
    responses = [p.function_response for p in contents[2].parts]
    assert [(r.name, r.response) for r in responses] == [
        ("add", {"result": "3"}),
        ("add", {"result": "7"}),
    ]
    assert contents[3].parts[0].text == "done"


def test_no_system_gives_none():
    system, contents = to_gemini_contents([Message("user", "hi")])
    assert system is None and len(contents) == 1


def test_tools_conversion_strips_keys(tool):
    spec = ToolSpec("t", "d", {"type": "object", "additionalProperties": False, "title": "x"})
    assert to_gemini_tools(None) is None and to_gemini_tools([]) is None
    (t,) = to_gemini_tools([tool, spec])
    assert isinstance(t, types.Tool)
    assert [d.name for d in t.function_declarations] == ["add", "t"]
    assert t.function_declarations[1].parameters_json_schema == {"type": "object"}


def _resp(parts=None, finish="STOP", um=None, block=None, candidates=True):
    cands = []
    if candidates:
        content = types.Content(role="model", parts=parts) if parts is not None else None
        cands = [types.Candidate(content=content, finish_reason=finish)]
    return types.GenerateContentResponse(
        candidates=cands,
        usage_metadata=um,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block)
        if block
        else None,
    )


def test_parse_response_text_and_calls_with_synthesized_ids():
    parts = [
        types.Part(text="hello "),
        types.Part(text="hidden", thought=True),
        types.Part(function_call=types.FunctionCall(name="add", args={"a": 1})),
        types.Part(function_call=types.FunctionCall(id="given", name="sub")),
        types.Part(text="world"),
    ]
    um = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10, candidates_token_count=4, thoughts_token_count=2
    )
    r = parse_gemini_response(_resp(parts, um=um), "gemini-x", 0.3)
    assert r.text == "hello world"
    assert [(c.id, c.name, c.arguments) for c in r.tool_calls] == [
        ("call_0_add", "add", {"a": 1}),
        ("given", "sub", {}),
    ]
    assert (r.usage.input_tokens, r.usage.output_tokens) == (10, 6)
    assert r.model == "gemini-x" and r.latency_s == 0.3


@pytest.mark.parametrize(
    ("resp", "exc"),
    [
        (_resp(candidates=False), TransientError),
        (_resp([types.Part(text="x")], finish="MALFORMED_FUNCTION_CALL"), TransientError),
        (_resp([types.Part(text="x")], finish="OTHER"), TransientError),
        (_resp([types.Part(text="")]), TransientError),
        (_resp(None), TransientError),
        (_resp([types.Part(text="x")], finish="SAFETY"), LLMError),
        (_resp([types.Part(text="x")], finish="MAX_TOKENS"), LLMError),
        (_resp(candidates=False, block="SAFETY"), LLMError),
    ],
)
def test_bad_responses_raise(resp, exc):
    with pytest.raises(exc) as info:
        parse_gemini_response(resp, "m", 0.0)
    assert type(info.value) is exc


async def test_bad_response_is_never_cached(tmp_path, msgs):
    class Bad:
        model = "m"

        async def complete(self, messages, tools=None, *, temperature=0.0):
            return parse_gemini_response(_resp(candidates=False), "m", 0.0)

    with pytest.raises(TransientError):
        await CachedClient(Bad(), tmp_path).complete(msgs)
    assert not list(tmp_path.iterdir())


def test_clean_schema_keeps_defs_and_data_values():
    schema = {
        "$defs": {"A": {"type": "object", "title": "A", "additionalProperties": False}},
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["title", "additionalProperties"],
                "default": {"title": "x"},
                "const": {"$schema": 1},
            },
            "e": {"examples": [{"title": "t"}], "type": "object"},
        },
    }
    out = clean_schema(schema)
    assert out["$defs"] == {"A": {"type": "object"}}
    mode = out["properties"]["mode"]
    assert mode["enum"] == ["title", "additionalProperties"]
    assert mode["default"] == {"title": "x"} and mode["const"] == {"$schema": 1}
    assert out["properties"]["e"]["examples"] == [{"title": "t"}]


def test_clean_schema_keeps_def_named_like_keyword():
    schema = {
        "$defs": {"title": {"type": "string"}},
        "properties": {"x": {"$ref": "#/$defs/title"}},
    }
    assert clean_schema(schema)["$defs"] == {"title": {"type": "string"}}


def test_tool_name_resolved_from_earlier_call():
    msgs = [
        Message("user", "go"),
        Message("assistant", "", [ToolCall("a", "add", {})]),
        Message("tool", "3", tool_call_id="a"),
    ]
    _, contents = to_gemini_contents(msgs)
    assert contents[2].parts[0].function_response.name == "add"


def test_thought_signature_round_trip():
    part = types.Part(
        function_call=types.FunctionCall(name="add", args={"a": 1}), thought_signature=b"\x00sig"
    )
    resp = parse_gemini_response(_resp([part]), "m", 0.0)
    call = ToolCall.from_dict(resp.tool_calls[0].to_dict())  # survives the JSON cache
    _, contents = to_gemini_contents([Message("user", "go"), Message("assistant", "", [call])])
    assert contents[1].parts[0].thought_signature == b"\x00sig"


def test_unresolvable_tool_name_raises():
    with pytest.raises(LLMError):
        to_gemini_contents([Message("user", "go"), Message("tool", "3", tool_call_id="zz")])


def make(handler):
    return OllamaClient("qwen2.5:3b", transport=httpx.MockTransport(handler))


async def test_parses_text_usage_and_request_body(msgs, tool):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello"},
                "prompt_eval_count": 12,
                "eval_count": 5,
            },
        )

    r = await make(handler).complete(msgs, [tool], temperature=0.3)
    assert r.text == "hello" and r.tool_calls == [] and r.model == "qwen2.5:3b"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (12, 5)
    assert seen["url"].endswith("/api/chat")
    body = seen["body"]
    assert body["stream"] is False and body["options"] == {"temperature": 0.3, "num_ctx": 8192}
    assert body["tools"][0]["function"]["name"] == "add"
    assert body["messages"][0] == {"role": "system", "content": "be brief"}


async def test_parses_tool_calls_and_serializes_history():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}],
                }
            },
        )

    history = [
        Message("user", "go"),
        Message("assistant", "", [ToolCall("c", "add", {"a": 0})]),
        Message("tool", "0", tool_call_id="c", name="add"),
    ]
    r = await make(handler).complete(history)
    assert [(c.id, c.name, c.arguments) for c in r.tool_calls] == [
        ("call_0_add", "add", {"a": 1, "b": 2})
    ]
    assert "tools" not in seen["body"]
    sent = seen["body"]["messages"]
    assert sent[1]["tool_calls"] == [{"function": {"name": "add", "arguments": {"a": 0}}}]
    assert sent[2]["tool_name"] == "add"


@pytest.mark.parametrize(
    ("status", "exc"), [(429, RateLimitError), (503, TransientError), (400, LLMError)]
)
async def test_http_errors_mapped(msgs, status, exc):
    client = make(lambda request: httpx.Response(status, text="nope"))
    with pytest.raises(exc) as info:
        await client.complete(msgs)
    assert type(info.value) is exc


async def test_connection_error_is_transient(msgs):
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(TransientError):
        await make(handler).complete(msgs)


def _chat(**message):
    return lambda request: httpx.Response(200, json={"message": {"role": "assistant", **message}})


async def test_string_tool_arguments_are_parsed(msgs):
    call = {"function": {"name": "add", "arguments": '{"a": 1}'}}
    r = await make(_chat(content="", tool_calls=[call])).complete(msgs)
    assert r.tool_calls[0].arguments == {"a": 1}


async def test_bad_json_arguments_transient(msgs):
    call = {"function": {"name": "add", "arguments": "{oops"}}
    with pytest.raises(TransientError):
        await make(_chat(content="", tool_calls=[call])).complete(msgs)


async def test_truncated_response_is_llmerror_and_not_cached(tmp_path, msgs):
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "part"}, "done_reason": "length"})

    with pytest.raises(LLMError) as info:
        await CachedClient(make(handler), tmp_path).complete(msgs)
    assert type(info.value) is LLMError
    assert not list(tmp_path.iterdir())


async def test_empty_response_is_transient_and_not_cached(tmp_path, msgs):
    with pytest.raises(TransientError):
        await CachedClient(make(_chat(content="")), tmp_path).complete(msgs)
    assert not list(tmp_path.iterdir())


async def test_num_ctx_sent(msgs):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    await OllamaClient(transport=httpx.MockTransport(handler), num_ctx=4096).complete(msgs)
    assert seen["options"]["num_ctx"] == 4096


def _ollama_up() -> bool:
    try:
        socket.create_connection(("localhost", 11434), timeout=0.3).close()
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable")
async def test_real_ollama_call():
    r = await OllamaClient().complete([Message("user", "Say hi in one word.")])
    assert r.text and r.usage.output_tokens > 0
