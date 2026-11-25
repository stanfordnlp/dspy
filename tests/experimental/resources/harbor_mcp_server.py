"""Stateful MCP fixture; runs only inside the disposable Harbor task container."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

transport, label = sys.argv[1:3]
server = FastMCP("harbor-test", host="0.0.0.0", port=8123)
counts = {}


@server.tool()
async def counter(step: int, ctx: Context, payload: str = "") -> dict[str, Any]:
    """Increment this MCP session's counter and return sandbox evidence."""
    session = id(ctx.request_context.session)
    counts[session] = counts.get(session, 0) + step
    return {
        "count": counts[session],
        "label": label,
        "setting": os.environ.get("AGENT_SETTING"),
        "payload": payload,
        "sandbox": Path("/app/sandbox-marker").read_text(),
    }


@server.tool()
async def fail() -> str:
    """Return an MCP tool error."""
    raise ValueError("intentional MCP failure")


@server.tool()
async def wait() -> str:
    """Block until the caller cancels the tool call."""
    await asyncio.sleep(300)
    return "unexpected"


server.run(transport=transport)
