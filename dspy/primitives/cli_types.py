"""
Data types for CLI module execution tracing.

CLIEvent and CLITrajectory capture the full record of a CLI subprocess
execution, analogous to REPLEntry/REPLHistory for RLM.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class CLIEvent(BaseModel):
    """A single event from a CLI agent's structured output stream.

    Many CLI agents (Codex, Claude Code) emit structured JSONL events.
    This captures individual events for optimizer inspection.
    """

    type: str
    """Event type: 'thinking', 'tool_call', 'tool_result', 'agent_message', 'error', etc."""

    content: str
    """The event payload text."""

    timestamp: float | None = None
    """Optional timestamp from the event."""

    raw: dict[str, Any] | None = None
    """The original parsed JSON event, if available."""

    def format(self) -> str:
        """Human-readable format for inclusion in Predict inputs."""
        ts = f" (t={self.timestamp:.2f}s)" if self.timestamp is not None else ""
        return f"[{self.type}]{ts}: {self.content}"


class CLITrajectory(BaseModel):
    """Full record of a CLI execution for optimizer inspection.

    Analogous to REPLHistory for RLM. Captures everything needed for
    optimizers (especially GEPA) to understand what the CLI agent did.
    """

    prompt: str
    """The prompt text sent to the CLI."""

    events: list[CLIEvent]
    """Parsed structured events (empty for plain-text CLIs)."""

    raw_stdout: str
    """Full stdout from the subprocess."""

    stderr: str
    """Full stderr from the subprocess."""

    returncode: int
    """Process exit code."""

    elapsed: float
    """Wall-clock seconds for the subprocess execution."""

    def format(self, max_chars: int = 10_000) -> str:
        """Human-readable format for inclusion in Predict inputs.

        Args:
            max_chars: Maximum characters for stdout/stderr sections.
        """
        parts = [f"=== CLI Execution (exit={self.returncode}, {self.elapsed:.1f}s) ==="]
        parts.append(f"\n--- Prompt ---\n{self.prompt}")

        if self.events:
            parts.append(f"\n--- Events ({len(self.events)}) ---")
            for event in self.events:
                parts.append(event.format())

        stdout_display = self.raw_stdout
        if len(stdout_display) > max_chars:
            stdout_display = stdout_display[:max_chars] + "\n... (truncated)"
        parts.append(f"\n--- Stdout ---\n{stdout_display}")

        if self.stderr.strip():
            stderr_display = self.stderr
            if len(stderr_display) > max_chars:
                stderr_display = stderr_display[:max_chars] + "\n... (truncated)"
            parts.append(f"\n--- Stderr ---\n{stderr_display}")

        return "\n".join(parts)

    def __str__(self) -> str:
        return self.format()

    def get_agent_message(self) -> str | None:
        """Extract the final agent message from events, if any.

        Looks for the last 'agent_message' type event, which is the
        pattern used by Codex and Claude Code JSONL output.
        """
        for event in reversed(self.events):
            if event.type == "agent_message" and event.content.strip():
                return event.content.strip()
        return None


def parse_jsonl_events(stdout: str) -> list[CLIEvent]:
    """Parse JSONL-formatted stdout into CLIEvent objects.

    Handles mixed output (some lines JSON, some not) gracefully.
    Recognizes patterns from Codex and Claude Code output formats.

    Args:
        stdout: Raw stdout text, potentially containing JSONL lines.

    Returns:
        List of parsed CLIEvent objects. Empty if no valid JSONL found.
    """
    events: list[CLIEvent] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict):
            continue

        event = _parse_single_event(data)
        if event is not None:
            events.append(event)

    return events


def _parse_single_event(data: dict[str, Any]) -> CLIEvent | None:
    """Parse a single JSON object into a CLIEvent.

    Handles multiple formats:
    - Codex-style: {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
    - Generic: {"type": "...", "content": "..."} or {"type": "...", "text": "..."}
    """
    event_type = data.get("type")
    if not event_type:
        return None

    # Codex-style nested events
    if event_type == "item.completed":
        item = data.get("item")
        if isinstance(item, dict):
            item_type = item.get("type", "unknown")
            text = item.get("text") or item.get("content") or ""
            return CLIEvent(
                type=item_type,
                content=str(text),
                raw=data,
            )

    # Generic event format
    content = data.get("content") or data.get("text") or data.get("message") or ""
    timestamp = data.get("timestamp") or data.get("time")

    return CLIEvent(
        type=event_type,
        content=str(content),
        timestamp=float(timestamp) if timestamp is not None else None,
        raw=data,
    )
