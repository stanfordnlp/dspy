from __future__ import annotations

import json
import sys
from contextlib import suppress
from typing import Any, TextIO


def _green(text: str, end: str = "\n", *, use_colors: bool = True) -> str:
    if use_colors:
        return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end
    return str(text).lstrip() + end


def _red(text: str, end: str = "\n", *, use_colors: bool = True) -> str:
    if use_colors:
        return "\x1b[31m" + str(text) + "\x1b[0m" + end
    return str(text) + end


def _blue(text: str, end: str = "\n", *, use_colors: bool = True) -> str:
    if use_colors:
        return "\x1b[34m" + str(text) + "\x1b[0m" + end
    return str(text) + end


def pretty_print_history(history: list[dict[str, Any]], n: int = 1, file: TextIO | None = None) -> None:
    """Print the last n prompts and their completions.

    Args:
        history: The history list to print from.
        n: Number of recent entries to display. Defaults to 1.
        file: An optional file-like object to write output to (must have a
            `.write()` method). When provided, ANSI color codes are
            automatically disabled. Defaults to `None` (prints to stdout).
    """
    out = file or sys.stdout
    use_colors = file is None

    def print_tool_calls(tool_calls):
        if tool_calls:
            print(_red("Tool calls:", use_colors=use_colors), file=out)
        for tool_call in tool_calls or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            arguments = tool_call.get("args", tool_call.get("arguments", {})) if arguments is None else arguments
            with suppress(json.JSONDecodeError):
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            print(_green(f"{function.get('name') or tool_call.get('name', '<unknown>')}: {json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, (dict, list)) else str(arguments)}", use_colors=use_colors), file=out)

    def _render_part(part, use_colors):
        kind = part.get("type")
        if kind == "text":
            return part.get("text", "").strip()
        if kind in ("image", "audio", "video", "document", "binary"):
            if part.get("data") is not None:
                where = f"base64 ({len(part['data'])} chars)"
            else:
                where = part.get("url") or part.get("file_id") or part.get("path") or ""
            return _blue(f"<{kind} {part.get('media_type', '')}: {where}>", use_colors=use_colors)
        if kind == "tool_call":
            return _green(f"{part.get('name')}: {json.dumps(part.get('input', {}), ensure_ascii=False)}", use_colors=use_colors)
        if kind == "tool_result":
            body = " ".join(_render_part(c, use_colors) for c in part.get("content", []))
            return _green(f"tool result {part.get('id')}: {body}", use_colors=use_colors)
        return _blue(f"<{kind}>", use_colors=use_colors)

    for item in history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n", file=out)
        print(_blue(f"[{timestamp}]", use_colors=use_colors), file=out)

        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:", use_colors=use_colors), file=out)
            if isinstance(msg["content"], str):
                print(msg["content"].strip(), file=out)
            else:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
                        print(_render_part(c, use_colors), file=out)
            print_tool_calls(msg.get("tool_calls"))
            print("\n", file=out)

        if isinstance(outputs[0], dict):
            if outputs[0].get("text"):
                print(_red("Response:", use_colors=use_colors), file=out)
                print(_green(outputs[0]["text"].strip(), use_colors=use_colors), file=out)

            print_tool_calls(outputs[0].get("tool_calls"))
        else:
            print(_red("Response:", use_colors=use_colors), file=out)
            print(_green(outputs[0].strip(), use_colors=use_colors), file=out)

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end="", use_colors=use_colors), file=out)

    print("\n\n\n", file=out)
