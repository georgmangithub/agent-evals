"""Tool logic: arXiv API client (parsing, id normalization, rate limit, cache) and SQLite notes."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import httpx

API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "agent-evals/0.1 (research assistant demo)"
MIN_INTERVAL = 3.0
MAX_RESPONSE_BYTES = 2_000_000
MAX_DOWNLOAD_S = 60.0
_ATOM = "http://www.w3.org/2005/Atom"
_NS = {"a": _ATOM}
log = logging.getLogger(__name__)
_NEW_ID = re.compile(r"[0-9]{4}\.[0-9]{4,5}(v[1-9][0-9]*)?", re.ASCII)
_OLD_ID = re.compile(r"[a-z\-]+(\.[A-Z]{2})?/[0-9]{7}(v[1-9][0-9]*)?", re.ASCII)
_PREFIX = re.compile(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf)/|^arxiv:", re.IGNORECASE)


class ArxivError(Exception):
    """Raised for invalid input or a failed arXiv request."""


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    categories: list[str]
    pdf_url: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_id(raw: str, keep_version: bool = True) -> str:
    """Strip URL/`arXiv:` prefixes and `.pdf`; keep an explicit version unless told not to."""
    s = _PREFIX.sub("", raw.strip())
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE)
    if not (_NEW_ID.fullmatch(s) or _OLD_ID.fullmatch(s)):
        raise ArxivError(f"Invalid arXiv id {raw!r}; expected e.g. '2303.08774' or 'cs/0112017v2'")
    return s if keep_version else re.sub(r"v\d+$", "", s)


def _text(el: ET.Element, path: str) -> str:
    return " ".join(el.findtext(path, default="", namespaces=_NS).split())


def parse_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom response into papers (ids come back without version)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError(f"Could not parse arXiv response: {exc}") from exc
    if root.tag != f"{{{_ATOM}}}feed":
        raise ArxivError("Unexpected arXiv response: not an Atom feed")
    papers = []
    for entry in root.findall("a:entry", _NS):
        entry_id = _text(entry, "a:id")
        if "/api/errors" in entry_id:
            raise ArxivError(f"arXiv API error: {_text(entry, 'a:summary')}")
        try:
            arxiv_id = normalize_id(entry_id, keep_version=False)
        except ArxivError:
            log.warning("Skipping arXiv entry with malformed id %r", entry_id)
            continue
        pdf = next(
            (
                link.get("href", "")
                for link in entry.findall("a:link", _NS)
                if link.get("title") == "pdf"
            ),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_text(entry, "a:title"),
                authors=[_text(a, "a:name") for a in entry.findall("a:author", _NS)],
                summary=_text(entry, "a:summary"),
                published=_text(entry, "a:published"),
                categories=[c.get("term", "") for c in entry.findall("a:category", _NS)],
                pdf_url=pdf,
            )
        )
    return papers


class ArxivClient:
    """Rate-limited arXiv client with an optional response cache keyed by request URL."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        client: httpx.Client | None = None,
        min_interval: float = MIN_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._client = client or httpx.Client(timeout=20.0, follow_redirects=True)
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self._lock = threading.Lock()

    def request_url(self, params: dict[str, str | int]) -> str:
        return str(httpx.URL(API_URL, params=params))

    def cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.xml"

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _read_cached(self, path: Path | None) -> list[Paper] | None:
        if path is None or not path.exists():
            return None
        try:
            return parse_feed(path.read_text(encoding="utf-8"))
        except (ArxivError, UnicodeDecodeError, OSError):
            path.unlink(missing_ok=True)
            return None

    def _download(self, url: str) -> str:
        deadline = time.monotonic() + MAX_DOWNLOAD_S
        try:
            with self._client.stream("GET", url, headers={"User-Agent": USER_AGENT}) as resp:
                resp.raise_for_status()
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise ArxivError(f"arXiv response exceeded {MAX_RESPONSE_BYTES} bytes")
                    if time.monotonic() > deadline:
                        raise ArxivError(f"arXiv download exceeded {MAX_DOWNLOAD_S:.0f}s")
        except httpx.HTTPError as exc:
            raise ArxivError(f"arXiv request failed: {exc}") from exc
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArxivError("arXiv response was not valid UTF-8") from exc

    def _load(self, params: dict[str, str | int]) -> list[Paper]:
        url = self.request_url(params)
        path = self.cache_path(url)
        with self._lock:
            cached = self._read_cached(path)
            if cached is not None:
                return cached
            self._throttle()
            text = self._download(url)
            papers = parse_feed(text)
            if papers and path is not None:
                try:
                    self._write_atomic(path, text)
                except OSError as exc:
                    log.warning("arXiv cache write failed for %s: %s", path, exc)
            return papers

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def search_params(query: str, max_results: int) -> dict[str, str | int]:
        return {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }

    @staticmethod
    def get_params(norm_id: str) -> dict[str, str | int]:
        return {"id_list": norm_id, "max_results": 1}

    def search(self, query: str, max_results: int = 5) -> list[Paper]:
        return self._load(self.search_params(query, max_results))

    def get(self, arxiv_id: str) -> Paper:
        norm = normalize_id(arxiv_id)
        papers = self._load(self.get_params(norm))
        if not papers:
            raise ArxivError(f"No paper found for arXiv id {norm}")
        return replace(papers[0], arxiv_id=norm)


MAX_BODY = 20_000
MAX_TITLE = 300
MAX_TAGS = 20
MAX_TAG_LEN = 50
MAX_ID = 2**63 - 1


class NoteError(ValueError):
    """Raised for invalid note input or a missing note."""


def _check_id(note_id: int) -> None:
    if not 1 <= note_id <= MAX_ID:
        raise NoteError(f"note id must be between 1 and {MAX_ID}")


def _like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class NotesStore:
    def __init__(self, db_path: Path | str, use_fts: bool = True) -> None:
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " title TEXT NOT NULL, body TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '',"
            " paper_id TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self.fts = False
        try:
            if not use_fts:
                raise sqlite3.OperationalError("FTS disabled")
            self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5("
                "title, body, content='notes', content_rowid='id')"
            )
            self._db.executescript(
                "CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN"
                " INSERT INTO notes_fts(rowid, title, body) VALUES (new.id, new.title, new.body);"
                " END;"
                "CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN"
                " INSERT INTO notes_fts(notes_fts, rowid, title, body)"
                " VALUES ('delete', old.id, old.title, old.body); END;"
            )
            self._db.execute("INSERT INTO notes_fts(notes_fts) VALUES ('rebuild')")
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        self._db.commit()

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "tags": [t for t in row["tags"].split(",") if t],
            "paper_id": row["paper_id"],
            "created_at": row["created_at"],
        }

    def save_note(
        self, title: str, body: str, tags: list[str] | None = None, paper_id: str | None = None
    ) -> int:
        title, body = title.strip(), body.strip()
        if not title or len(title) > MAX_TITLE:
            raise NoteError(f"title must be 1-{MAX_TITLE} characters")
        if not body:
            raise NoteError("body must not be empty")
        if len(body) > MAX_BODY:
            raise NoteError(f"body too long ({len(body)} chars); the limit is {MAX_BODY}")
        clean = sorted({t.strip().lower() for t in tags or [] if t.strip()})
        if len(clean) > MAX_TAGS:
            raise NoteError(f"at most {MAX_TAGS} tags allowed")
        if any("," in t or len(t) > MAX_TAG_LEN for t in clean):
            raise NoteError(f"tags must be at most {MAX_TAG_LEN} characters and contain no commas")
        if paper_id is not None:
            try:
                paper_id = normalize_id(paper_id)
            except ArxivError as exc:
                raise NoteError(str(exc)) from exc
        cur = self._db.execute(
            "INSERT INTO notes (title, body, tags, paper_id) VALUES (?, ?, ?, ?)",
            (title, body, f",{','.join(clean)}," if clean else "", paper_id),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def get_note(self, note_id: int) -> dict:
        _check_id(note_id)
        row = self._db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            raise NoteError(f"No note with id {note_id}")
        return self._row(row)

    def search_notes(self, query: str, tag: str | None = None, limit: int = 10) -> list[dict]:
        """Search title/body as plain text (query is never interpreted as FTS or SQL syntax)."""
        query = query.strip()
        if not query:
            raise NoteError("query must not be empty")
        sql, where, args = "SELECT n.* FROM notes n", [], []
        if self.fts:
            terms = re.findall(r"\w+", query)
            if not terms:
                return []
            sql += " JOIN notes_fts f ON f.rowid = n.id"
            where.append("notes_fts MATCH ?")
            args.append(" ".join(f'"{t}"' for t in terms))
        else:
            where.append("(n.title LIKE ? ESCAPE '\\' OR n.body LIKE ? ESCAPE '\\')")
            args += [f"%{_like(query)}%"] * 2
        if tag and tag.strip():
            where.append("n.tags LIKE ? ESCAPE '\\'")
            args.append(f"%,{_like(tag.strip().lower())},%")
        sql += " WHERE " + " AND ".join(where) + " ORDER BY n.id DESC LIMIT ?"
        args.append(max(1, min(limit, 50)))
        return [self._row(r) for r in self._db.execute(sql, args)]

    def delete_note(self, note_id: int) -> None:
        _check_id(note_id)
        cur = self._db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self._db.commit()
        if cur.rowcount == 0:
            raise NoteError(f"No note with id {note_id}")

    def close(self) -> None:
        self._db.close()
