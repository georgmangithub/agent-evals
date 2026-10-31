import itertools
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from agent_evals.mcp_server import tools as arxiv_mod
from agent_evals.mcp_server.server import REPO_ROOT, build_server, resolve_paths
from agent_evals.mcp_server.tools import (
    MAX_BODY,
    ArxivClient,
    ArxivError,
    NoteError,
    NotesStore,
    normalize_id,
    parse_feed,
)

FIXTURE = Path(__file__).parent / "fixtures" / "search.xml"


def test_parse_feed():
    papers = parse_feed(FIXTURE.read_text(encoding="utf-8"))
    assert [p.arxiv_id for p in papers] == ["2005.11401", "cs/0112017"]
    first = papers[0]
    assert first.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert first.authors == ["Patrick Lewis", "Ethan Perez"]
    assert first.categories == ["cs.CL", "cs.LG"]
    assert first.published == "2020-05-22T21:34:34Z"
    assert first.pdf_url == "http://arxiv.org/pdf/2005.11401v4"
    assert "\n" not in first.summary


def test_parse_garbage():
    with pytest.raises(ArxivError):
        parse_feed("not xml")


@pytest.mark.parametrize(
    ("raw", "keep", "expected"),
    [
        ("http://arxiv.org/abs/2303.08774v2", True, "2303.08774v2"),
        ("http://arxiv.org/abs/2303.08774v2", False, "2303.08774"),
        ("arXiv:2303.08774", True, "2303.08774"),
        ("https://arxiv.org/pdf/2303.08774v1.pdf", True, "2303.08774v1"),
        (" 2303.08774 ", True, "2303.08774"),
        ("cs/0112017v1", True, "cs/0112017v1"),
        ("https://arxiv.org/pdf/2303.08774v1.PDF", True, "2303.08774v1"),
    ],
)
def test_normalize_id(raw, keep, expected):
    assert normalize_id(raw, keep_version=keep) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "hello",
        "2303.1",
        "../../etc/passwd",
        "2303.08774; drop",
        "2303.08774v0",
        "٢٣٠٣.08774",
        "2303.08774\nfoo",
    ],
)
def test_normalize_id_rejects(bad):
    with pytest.raises(ArxivError):
        normalize_id(bad)


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, secs):
        self.sleeps.append(secs)
        self.now += secs


def make_client(tmp_path, calls, ft=None, cache=True):
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text=FIXTURE.read_text(encoding="utf-8"))

    ft = ft or FakeTime()
    return ArxivClient(
        cache_dir=tmp_path / "cache" if cache else None,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=ft.clock,
        sleep=ft.sleep,
    )


def test_cache_hit_skips_transport(tmp_path):
    calls: list[str] = []
    client = make_client(tmp_path, calls)
    first = client.search("rag", 2)
    second = client.search("rag", 2)
    assert first == second
    assert len(calls) == 1
    assert len(list((tmp_path / "cache").glob("*.xml"))) == 1
    assert "search_query=rag" in calls[0]


def test_cache_survives_new_client(tmp_path):
    calls: list[str] = []
    make_client(tmp_path, calls).search("rag", 2)
    make_client(tmp_path, calls).search("rag", 2)
    assert len(calls) == 1


def test_rate_limit_sleeps_between_live_requests(tmp_path):
    calls: list[str] = []
    ft = FakeTime()
    client = make_client(tmp_path, calls, ft, cache=False)
    client.search("a")
    assert ft.sleeps == []
    ft.now += 1.0
    client.search("b")
    assert ft.sleeps == [pytest.approx(2.0)]
    ft.now += 10.0
    client.search("c")
    assert len(ft.sleeps) == 1


def test_cached_requests_do_not_count_toward_rate_limit(tmp_path):
    calls: list[str] = []
    ft = FakeTime()
    client = make_client(tmp_path, calls, ft)
    client.search("a")
    client.search("a")
    assert ft.sleeps == []


def test_http_error_becomes_arxiv_error(tmp_path):
    client = ArxivClient(
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
        sleep=lambda s: None,
    )
    with pytest.raises(ArxivError):
        client.search("x")


def test_get_not_found(tmp_path):
    empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    client = ArxivClient(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text=empty))
        )
    )
    with pytest.raises(ArxivError, match="No paper"):
        client.get("2303.08774")


def make_mock_client(tmp_path, body, **kw):
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=body)

    client = ArxivClient(
        cache_dir=tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
        **kw,
    )
    return client, calls


def cached_files(tmp_path):
    cache = tmp_path / "cache"
    return list(cache.glob("*")) if cache.exists() else []


def test_request_is_https_with_user_agent(tmp_path):
    client, calls = make_mock_client(tmp_path, FIXTURE.read_bytes())
    client.search("rag")
    assert calls[0].url.scheme == "https"
    assert calls[0].headers["user-agent"].startswith("agent-evals/")


def test_html_body_rejected_and_not_cached(tmp_path):
    client, _ = make_mock_client(tmp_path, b"<html><body>Rate exceeded</body></html>")
    with pytest.raises(ArxivError):
        client.search("rag")
    assert cached_files(tmp_path) == []


def test_non_atom_xml_rejected(tmp_path):
    client, _ = make_mock_client(tmp_path, b"<root/>")
    with pytest.raises(ArxivError, match="Atom"):
        client.search("rag")
    assert cached_files(tmp_path) == []


def test_api_error_entry_raises_and_not_cached(tmp_path):
    body = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        b"<id>http://arxiv.org/api/errors#incorrect_id_format</id>"
        b"<summary>incorrect id format</summary></entry></feed>"
    )
    client, _ = make_mock_client(tmp_path, body)
    with pytest.raises(ArxivError, match="incorrect id format"):
        client.search("rag")
    assert cached_files(tmp_path) == []


def test_empty_feed_not_cached(tmp_path):
    client, calls = make_mock_client(tmp_path, b'<feed xmlns="http://www.w3.org/2005/Atom"/>')
    assert client.search("zzz") == []
    assert cached_files(tmp_path) == []
    client.search("zzz")
    assert len(calls) == 2


@pytest.mark.parametrize("junk", [b"garbage<", b"\xff\xfe\x00bad", b"<root/>"])
def test_corrupt_cache_heals(tmp_path, junk):
    client, calls = make_mock_client(tmp_path, FIXTURE.read_bytes())
    client.search("rag")
    (path,) = cached_files(tmp_path)
    path.write_bytes(junk)
    assert len(client.search("rag")) == 2
    assert len(calls) == 2
    assert path.read_text(encoding="utf-8") == FIXTURE.read_text(encoding="utf-8")


def test_cache_write_is_atomic(tmp_path, monkeypatch):
    client, _ = make_mock_client(tmp_path, FIXTURE.read_bytes())

    def failing_replace(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(arxiv_mod.os, "replace", failing_replace)
    assert client.search("rag")
    assert cached_files(tmp_path) == []


def test_response_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv_mod, "MAX_RESPONSE_BYTES", 100)
    client, _ = make_mock_client(tmp_path, FIXTURE.read_bytes())
    with pytest.raises(ArxivError, match="exceeded"):
        client.search("rag")
    assert cached_files(tmp_path) == []


def test_malformed_entry_id_skipped(caplog):
    text = FIXTURE.read_text(encoding="utf-8").replace("2005.11401v4</id>", "bogus</id>", 1)
    with caplog.at_level("WARNING"):
        papers = parse_feed(text)
    assert [p.arxiv_id for p in papers] == ["cs/0112017"]
    assert "malformed" in caplog.text


def test_get_keeps_requested_version(tmp_path):
    client, _ = make_mock_client(tmp_path, FIXTURE.read_bytes())
    assert client.get("2005.11401v4").arxiv_id == "2005.11401v4"
    assert client.get("2005.11401").arxiv_id == "2005.11401"


def run_threads(target, n, args=lambda i: ()):
    threads = [threading.Thread(target=target, args=args(i)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_requests_are_throttled():
    lock = threading.Lock()
    now = [100.0]
    starts: list[float] = []

    def clock():
        with lock:
            return now[0]

    def sleep(secs):
        with lock:
            now[0] += secs

    def handler(request):
        starts.append(clock())
        return httpx.Response(200, content=FIXTURE.read_bytes())

    client = ArxivClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)), clock=clock, sleep=sleep
    )
    run_threads(client.search, 4, lambda i: (f"q{i}",))
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert len(starts) == 4
    assert all(g >= 3.0 - 1e-9 for g in gaps)


def test_concurrent_identical_requests_make_one_call(tmp_path):
    client, calls = make_mock_client(tmp_path, FIXTURE.read_bytes())
    run_threads(client.search, 4, lambda i: ("rag",))
    assert len(calls) == 1


def test_download_deadline(tmp_path, monkeypatch):
    client, _ = make_mock_client(tmp_path, FIXTURE.read_bytes())
    ticks = itertools.count(0, 100)
    monkeypatch.setattr(arxiv_mod.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ArxivError, match="exceeded"):
        client.search("rag")
    assert cached_files(tmp_path) == []


@pytest.fixture
def store(tmp_path):
    s = NotesStore(tmp_path / "notes.db")
    yield s
    s.close()


def test_crud(store):
    nid = store.save_note("RAG basics", "Retrieval helps grounding", ["RAG", "intro"], "2005.11401")
    note = store.get_note(nid)
    assert note["title"] == "RAG basics"
    assert note["tags"] == ["intro", "rag"]
    assert note["paper_id"] == "2005.11401"
    store.delete_note(nid)
    with pytest.raises(NoteError):
        store.get_note(nid)
    with pytest.raises(NoteError):
        store.delete_note(nid)


def test_search_and_tag_filter(store):
    a = store.save_note("Transformers", "attention is all you need", ["nlp"])
    b = store.save_note("Retrieval", "attention to retrieval quality", ["rag"])
    assert {n["id"] for n in store.search_notes("attention")} == {a, b}
    assert [n["id"] for n in store.search_notes("attention", tag="rag")] == [b]
    assert store.search_notes("attention", tag="missing") == []
    assert len(store.search_notes("attention", limit=1)) == 1


def test_deleted_note_not_searchable(store):
    nid = store.save_note("Temp", "ephemeral content")
    store.delete_note(nid)
    assert store.search_notes("ephemeral") == []


@pytest.mark.parametrize(
    "query", ["'; DROP TABLE notes; --", '" OR 1=1 --', "title:* NOT body", "100%", "a_b"]
)
def test_hostile_query_is_plain_text(store, query):
    store.save_note("Safe", "ordinary body")
    store.search_notes(query)
    assert len(store.search_notes("ordinary")) == 1


def test_injection_text_stored_and_found_verbatim(store):
    store.save_note("Tricky", "contains '; DROP TABLE notes; -- inside")
    assert len(store.search_notes("DROP TABLE notes")) == 1
    assert store.get_note(1)["body"].startswith("contains '; DROP")


def test_like_fallback(tmp_path):
    s = NotesStore(tmp_path / "n.db", use_fts=False)
    assert s.fts is False
    s.save_note("Hello", "100% sure; x_y", ["t"])
    assert len(s.search_notes("100%")) == 1
    assert s.search_notes("x_z") == []
    assert len(s.search_notes("sure", tag="t")) == 1


def test_bounds_and_paper_id(store):
    with pytest.raises(NoteError, match="at most 20"):
        store.save_note("t", "b", [f"t{i}" for i in range(21)])
    with pytest.raises(NoteError, match="50"):
        store.save_note("t", "b", ["x" * 51])
    with pytest.raises(NoteError, match="Invalid arXiv id"):
        store.save_note("t", "b", paper_id="nope")
    nid = store.save_note("t", "b", paper_id="https://arxiv.org/abs/2303.08774v2")
    assert store.get_note(nid)["paper_id"] == "2303.08774v2"


@pytest.mark.parametrize("bad", [0, -1, 2**63, 2**80])
def test_id_bounds(store, bad):
    with pytest.raises(NoteError, match="between 1"):
        store.delete_note(bad)
    with pytest.raises(NoteError, match="between 1"):
        store.get_note(bad)


def test_validation(store):
    with pytest.raises(NoteError):
        store.save_note("", "body")
    with pytest.raises(NoteError):
        store.save_note("t", "x" * (MAX_BODY + 1))
    with pytest.raises(NoteError):
        store.save_note("t", "b", ["a,b"])
    with pytest.raises(NoteError):
        store.search_notes("  ")


def test_reopen_with_fts_indexes_existing_notes(tmp_path):
    s = NotesStore(tmp_path / "n.db", use_fts=False)
    s.save_note("Old", "needle in haystack", [])
    s.close()
    s = NotesStore(tmp_path / "n.db")
    assert s.fts is True
    assert len(s.search_notes("needle")) == 1
    s.close()


EXPECTED_TOOLS = {"search_papers", "get_paper", "save_note", "search_notes", "delete_note"}


def offline_arxiv(cache_dir: Path) -> ArxivClient:
    def boom(request):
        raise AssertionError("network access attempted")

    return ArxivClient(
        cache_dir=cache_dir, client=httpx.Client(transport=httpx.MockTransport(boom))
    )


def seed_cache(client: ArxivClient, query: str, max_results: int) -> None:
    path = client.cache_path(client.request_url(client.search_params(query, max_results)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")


def payload(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def exercise(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == EXPECTED_TOOLS
    assert tools["delete_note"].annotations.destructive_hint is True
    for name in ("search_papers", "get_paper", "search_notes"):
        assert tools[name].annotations.read_only_hint is True
    assert not tools["save_note"].annotations.destructive_hint
    for name in ("search_papers", "get_paper"):
        assert tools[name].annotations.open_world_hint is True
    for name in ("save_note", "search_notes", "delete_note"):
        assert tools[name].annotations.open_world_hint is False
    assert all(t.description for t in tools.values())

    saved = await client.call_tool(
        "save_note", {"title": "RAG", "body": "retrieval grounding", "tags": ["rag"]}
    )
    assert not saved.is_error
    found = await client.call_tool("search_notes", {"query": "grounding", "tag": "rag"})
    assert [n["title"] for n in payload(found)["notes"]] == ["RAG"]

    papers = await client.call_tool("search_papers", {"query": "rag", "max_results": 2})
    assert [p["arxiv_id"] for p in payload(papers)["papers"]] == ["2005.11401", "cs/0112017"]

    bad = await client.call_tool("get_paper", {"arxiv_id": "nonsense"})
    assert bad.is_error
    assert "Invalid arXiv id" in bad.content[0].text
    too_long = await client.call_tool("save_note", {"title": "t", "body": "x" * 20_001})
    assert too_long.is_error

    deleted = await client.call_tool("delete_note", {"note_id": payload(saved)["id"]})
    assert not deleted.is_error
    assert (await client.call_tool("delete_note", {"note_id": 999})).is_error


async def test_in_memory(tmp_path):
    arxiv = offline_arxiv(tmp_path / "arxiv")
    seed_cache(arxiv, "rag", 2)
    notes = NotesStore(tmp_path / "notes.db")
    async with Client(build_server(arxiv, notes)) as client:
        await exercise(client)
    notes.close()


async def test_stdio(tmp_path):
    arxiv = offline_arxiv(tmp_path / "arxiv")
    seed_cache(arxiv, "rag", 2)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_evals.mcp_server"],
        env={
            "AGENT_EVALS_NOTES_DB": str(tmp_path / "notes.db"),
            "AGENT_EVALS_ARXIV_CACHE": str(tmp_path / "arxiv"),
        },
    )
    async with Client(params) as client:
        await exercise(client)
    assert (tmp_path / "notes.db").exists()


@pytest.mark.parametrize("query", ["", "   "])
async def test_empty_query_is_tool_error(tmp_path, query):
    server = build_server(offline_arxiv(tmp_path / "a"), NotesStore(tmp_path / "n.db"))
    async with Client(server) as client:
        assert (await client.call_tool("search_papers", {"query": query})).is_error
        assert (await client.call_tool("search_notes", {"query": query})).is_error


def test_defaults_anchored_to_repo_root():
    db, cache = resolve_paths({})
    assert db == REPO_ROOT / ".cache" / "notes.db"
    assert cache == REPO_ROOT / ".cache" / "arxiv"
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_env_overrides_and_empty_is_unset(tmp_path):
    db, cache = resolve_paths(
        {"AGENT_EVALS_NOTES_DB": str(tmp_path / "x.db"), "AGENT_EVALS_ARXIV_CACHE": ""}
    )
    assert db == tmp_path / "x.db"
    assert cache == REPO_ROOT / ".cache" / "arxiv"
