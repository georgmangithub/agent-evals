"""Guardrails: a human must approve destructive tool calls before they run."""

import asyncio
import json
from collections.abc import Awaitable, Callable

RunTool = Callable[[str, dict], Awaitable[str]]
Approver = Callable[[str, dict], Awaitable[bool]]


def is_destructive(annotations) -> bool:
    """Decide from a tool's MCP annotations whether it needs approval.

    Unknown means yes: a tool without annotations, or one that is not read-only and does not say
    it is safe, is treated as destructive. This follows the MCP spec, where destructiveHint
    defaults to true.
    """
    if annotations is None:
        return True
    if annotations.read_only_hint:
        return False
    return annotations.destructive_hint is not False


async def deny_all(name: str, arguments: dict) -> bool:
    """Default approver: nobody is there to approve, so the answer is no."""
    return False


async def ask_user(name: str, arguments: dict) -> bool:
    """Ask the person at the terminal. Only an explicit "y" or "yes" approves."""
    prompt = f"\nThe agent wants to run {name}({json.dumps(arguments)}). Allow? [y/N] "
    answer = await asyncio.to_thread(input, prompt)  # input() blocks, so run it off the loop
    return answer.strip().lower() in ("y", "yes")


def require_approval(run_tool: RunTool, destructive: set[str], approve: Approver) -> RunTool:
    """Wrap run_tool so destructive tools only run after approve() says yes."""

    async def guarded(name: str, arguments: dict) -> str:
        if name in destructive and not await approve(name, arguments):
            return f"Error: the user denied {name}. Do not try it again; finish without it."
        return await run_tool(name, arguments)

    return guarded
