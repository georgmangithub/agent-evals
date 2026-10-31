"""MCP registration layer for the research assistant tools, plus the stdio entry point."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from agent_evals.mcp_server.tools import ArxivClient, ArxivError, NoteError, NotesStore

WEB_READ = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
)
READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
WRITE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)
DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
)


def _surface_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn domain errors into MCP tool errors the model can read and correct."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (ArxivError, NoteError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def build_server(arxiv: ArxivClient, notes: NotesStore) -> MCPServer:
    server = MCPServer("research-assistant")

    def tool(annotations: ToolAnnotations, description: str) -> Callable:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            server.add_tool(_surface_errors(fn), description=description, annotations=annotations)
            return fn

        return deco

    @tool(
        WEB_READ,
        "Search arXiv for papers matching a query. Supports arXiv query syntax, e.g. "
        "'all:retrieval augmented generation' or 'au:Hinton AND cat:cs.LG'. Returns up to "
        "max_results papers (1-20) with arxiv_id, title, authors, summary, published date, "
        "categories and pdf_url.",
    )
    def search_papers(query: str, max_results: int = 5) -> dict:
        if not query.strip():
            raise ArxivError("query must not be empty")
        papers = arxiv.search(query.strip(), max(1, min(max_results, 20)))
        return {"papers": [p.to_dict() for p in papers]}

    @tool(
        WEB_READ,
        "Fetch one arXiv paper's metadata and abstract by id (e.g. '2303.08774', optionally "
        "with a version like '2303.08774v2', or an arxiv.org/abs URL).",
    )
    def get_paper(arxiv_id: str) -> dict:
        return arxiv.get(arxiv_id).to_dict()

    @tool(
        WRITE,
        "Save a research note for later retrieval. Provide a short title, a body (max 20000 "
        "characters), optional tags for filtering and an optional arXiv paper_id the note is "
        "about. Returns the new note id.",
    )
    def save_note(
        title: str, body: str, tags: list[str] | None = None, paper_id: str | None = None
    ) -> dict:
        return {"id": notes.save_note(title, body, tags or [], paper_id or None)}

    @tool(
        READ_ONLY,
        "Full-text search over saved research notes (title and body). Optionally restrict to "
        "notes carrying an exact tag. Returns up to limit notes (1-50), newest first.",
    )
    def search_notes(query: str, tag: str | None = None, limit: int = 10) -> dict:
        return {"notes": notes.search_notes(query, tag, limit)}

    @tool(
        DESTRUCTIVE,
        "Permanently delete a saved note by id. This cannot be undone; confirm with the user "
        "before calling.",
    )
    def delete_note(note_id: int) -> dict:
        notes.delete_note(note_id)
        return {"deleted": note_id}

    return server


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_paths(env: Mapping[str, str] = os.environ) -> tuple[Path, Path]:
    """Return (notes_db, arxiv_cache); empty env values count as unset."""
    default = REPO_ROOT / ".cache"
    db = env.get("AGENT_EVALS_NOTES_DB") or default / "notes.db"
    cache = env.get("AGENT_EVALS_ARXIV_CACHE") or default / "arxiv"
    return Path(db), Path(cache)


def main() -> None:
    db_path, cache_dir = resolve_paths()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arxiv, notes = ArxivClient(cache_dir=cache_dir), NotesStore(db_path)
    try:
        build_server(arxiv, notes).run("stdio")
    finally:
        arxiv.close()
        notes.close()
