"""Eval suite: run the agent on tasks with known answers and score the results.

Usage: python -m agent_evals.evals --model gemini:gemini-3.5-flash --model ollama:qwen2.5:3b --runs 3
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
from typing import Literal

from agent_evals.agent.loop import AgentResult, run_task
from agent_evals.llm import make_client
from agent_evals.llm.client import default_cache_dir
from agent_evals.mcp_server.tools import NotesStore
from agent_evals.tracing import Trace, write_trace

TASKS_FILE = Path(__file__).resolve().parents[2] / "evals" / "tasks.json"
TRACES_DIR = Path(__file__).resolve().parents[2] / ".cache" / "traces"


def read_notes(db: Path) -> list[tuple[str | None, str]]:
    """Return (paper_id, tags) for every note the agent saved during the run."""
    if not db.exists():
        return []
    with closing(sqlite3.connect(db)) as con:
        return con.execute("SELECT paper_id, tags FROM notes").fetchall()


def seed_notes(db: Path, notes: list[dict]) -> None:
    """Put notes in the database before the run, e.g. a poisoned note for injection tests."""
    if not notes:
        return
    store = NotesStore(db)
    try:
        for n in notes:
            store.save_note(n["title"], n["body"], n.get("tags"))
    finally:
        store.close()
    # Fixed timestamps: search_notes returns created_at, and a different time on every run
    # would change the model's input, miss the response cache, and make runs unrepeatable.
    with closing(sqlite3.connect(db)) as con:
        con.execute("UPDATE notes SET created_at = '2026-01-01 00:00:00'")
        con.commit()


def check(checks: dict, result: AgentResult, notes: list[tuple[str | None, str]]) -> list[str]:
    """Return the list of failed checks; an empty list means the task passed."""
    if result.stop_reason != "answer":  # the other checks would only repeat this failure
        return [f"stopped with {result.stop_reason}: {(result.error or '')[:120]}".strip()]
    failures = []
    answer = (result.answer or "").lower()
    for text in checks.get("answer_contains", []):
        if text.lower() not in answer:
            failures.append(f"answer missing {text!r}")
    options = checks.get("answer_contains_any", [])
    if options and not any(o.lower() in answer for o in options):
        failures.append("answer does not say it found nothing")
    for text in checks.get("answer_not_contains", []):
        if text.lower() in answer:
            failures.append(f"answer contains {text!r}")
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


def run_cache_dir(run: int) -> Path:
    """Each run number gets its own model cache, so repeated runs are real, independent samples.

    Run 0 uses the normal cache; later runs use .cache/llm/run<N>. Re-running the whole eval
    still replays every run from disk for free.
    """
    return default_cache_dir() if run == 0 else default_cache_dir() / f"run{run}"


Outcome = Literal["pass", "fail", "error"]
MARKS = {"pass": "✓", "fail": "✗", "error": "E"}


async def run_once(
    spec: str, task: dict, run: int, trace_file: Path
) -> tuple[Outcome, list[str], Trace]:
    """Run one task once with a fresh notes database and append its trace to trace_file.

    "error" means the model could not be reached (quota, outage): that says nothing about the
    model's quality, so it is kept out of the pass rate instead of counting as a failure.
    """
    llm = make_client(spec, cache_dir=run_cache_dir(run))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        notes_db = Path(tmp) / "notes.db"
        seed_notes(notes_db, task.get("setup_notes", []))
        result = await run_task(llm, task["task"], notes_db)
        failures = check(task["checks"], result, read_notes(notes_db))
    outcome: Outcome = (
        "error" if result.stop_reason == "error" else ("fail" if failures else "pass")
    )
    record = {"model": llm.model, "task": task["id"], "run": run + 1, "outcome": outcome}
    write_trace(trace_file, {**record, "failures": failures}, result.trace)
    return outcome, failures, result.trace


async def eval_model(spec: str, tasks: list[dict], runs: int, trace_file: Path) -> float | None:
    """Run every task `runs` times on one model; print a row per task, return the pass rate."""
    print(f"\n=== {spec}  ({runs} runs per task)")
    print(f"{'task':<18} {'runs':<5} pass  tokens/run  API req  s/run  failures")
    totals = {"pass": 0, "fail": 0, "error": 0}
    all_traces: list[Trace] = []
    for t in tasks:
        start = time.perf_counter()
        results = [await run_once(spec, t, run, trace_file) for run in range(runs)]
        traces = [trace for _, _, trace in results]
        all_traces += traces
        for outcome, _, _ in results:
            totals[outcome] += 1
        marks = "".join(MARKS[outcome] for outcome, _, _ in results)
        passed = sum(outcome == "pass" for outcome, _, _ in results)
        reasons = sorted({f for _, failures, _ in results for f in failures})
        tokens = sum(sum(tr.tokens) for tr in traces) / runs
        requests = sum(tr.api_requests for tr in traces)
        seconds = (time.perf_counter() - start) / runs
        print(
            f"{t['id']:<18} {marks:<5} {passed}/{runs}  {tokens:>10.0f}  {requests:>7}"
            f"  {seconds:5.1f}  {'; '.join(reasons)}"
        )
    scored = totals["pass"] + totals["fail"]
    rate = totals["pass"] / scored if scored else None
    print(f"pass rate: {totals['pass']}/{scored}" + (f" = {rate:.0%}" if rate is not None else ""))
    if totals["error"]:
        print(f"({totals['error']} runs could not reach the model and are not counted)")
    costs = [tr.cost() for tr in all_traces]
    cost = "unknown" if None in costs else f"${sum(costs):.4f}"
    tokens_in = sum(tr.tokens[0] for tr in all_traces)
    tokens_out = sum(tr.tokens[1] for tr in all_traces)
    print(
        f"API requests this eval: {sum(tr.api_requests for tr in all_traces)}"
        f" (the rest came from the cache)  tokens: {tokens_in} in / {tokens_out} out"
        f"  est. cost of new requests: {cost}"
    )
    return rate


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", help="model spec; repeat for several")
    parser.add_argument("--task", action="append", help="only run these task ids")
    parser.add_argument("--runs", type=int, default=3, help="runs per task (default 3)")
    args = parser.parse_args()
    tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    if args.task:
        tasks = [t for t in tasks if t["id"] in args.task]
    trace_file = TRACES_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    for spec in args.model or [None]:
        await eval_model(spec, tasks, args.runs, trace_file)
    print(f"\ntraces: {trace_file}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
