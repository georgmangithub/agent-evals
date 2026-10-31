"""Eval suite: run the agent on tasks with known answers and score the results.

Usage: python -m agent_evals.evals --model gemini:gemini-3.5-flash --model ollama:qwen2.5:3b
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

from agent_evals.agent.loop import AgentResult, run_task
from agent_evals.llm import make_client

TASKS_FILE = Path(__file__).resolve().parents[2] / "evals" / "tasks.json"


def read_notes(db: Path) -> list[tuple[str | None, str]]:
    """Return (paper_id, tags) for every note the agent saved during the run."""
    if not db.exists():
        return []
    with closing(sqlite3.connect(db)) as con:
        return con.execute("SELECT paper_id, tags FROM notes").fetchall()


def check(checks: dict, result: AgentResult, notes: list[tuple[str | None, str]]) -> list[str]:
    """Return the list of failed checks; an empty list means the task passed."""
    failures = []
    if result.stop_reason != "answer":
        failures.append(f"stopped with {result.stop_reason}: {result.error or ''}".strip())
    answer = (result.answer or "").lower()
    for text in checks.get("answer_contains", []):
        if text.lower() not in answer:
            failures.append(f"answer missing {text!r}")
    options = checks.get("answer_contains_any", [])
    if options and not any(o.lower() in answer for o in options):
        failures.append("answer does not say it found nothing")
    if "notes_count" in checks and len(notes) != checks["notes_count"]:
        failures.append(f"saved {len(notes)} notes, expected {checks['notes_count']}")
    if "note_paper_id" in checks and not any(
        (pid or "").startswith(checks["note_paper_id"]) for pid, _ in notes
    ):
        failures.append(f"no note for paper {checks['note_paper_id']}")
    if "note_tag" in checks and not any(f",{checks['note_tag']}," in tags for _, tags in notes):
        failures.append(f"no note tagged {checks['note_tag']!r}")
    called = {c.name for m in result.messages for c in m.tool_calls}
    for tool in checks.get("forbidden_tools", []):
        if tool in called:
            failures.append(f"called forbidden tool {tool}")
    return failures


async def eval_model(spec: str, tasks: list[dict]) -> int:
    """Run every task once on one model, print a row per task, return the number passed."""
    llm = make_client(spec)
    print(f"\n=== {spec}")
    passed = 0
    for t in tasks:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            notes_db = Path(tmp) / "notes.db"
            start = time.perf_counter()
            result = await run_task(llm, t["task"], notes_db)
            seconds = time.perf_counter() - start
            failures = check(t["checks"], result, read_notes(notes_db))
        tool_calls = sum(len(m.tool_calls) for m in result.messages)
        status = "PASS" if not failures else "FAIL"
        passed += not failures
        print(
            f"{status}  {t['id']:<18} {tool_calls} tool calls  {seconds:5.1f}s  {'; '.join(failures)}"
        )
    print(f"score: {passed}/{len(tasks)}")
    return passed


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", help="model spec; repeat for several")
    parser.add_argument("--task", action="append", help="only run these task ids")
    args = parser.parse_args()
    tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    if args.task:
        tasks = [t for t in tasks if t["id"] in args.task]
    for spec in args.model or [None]:
        await eval_model(spec, tasks)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
