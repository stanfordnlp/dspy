from __future__ import annotations

import sys
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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_tool_call(tool_call: Any) -> str:
    # OpenAI Chat Completions wire shape: {"function": {"name", "arguments"}}.
    function = _get(tool_call, "function")
    if function is not None:
        name = _get(function, "name", "")
        arguments = _get(function, "arguments", "")
        return f"{name}: {arguments}"
    # DSPy-native ToolCall / OpenAI Responses API shape: name+args on the item itself.
    name = _get(tool_call, "name", "")
    arguments = _get(tool_call, "args")
    if arguments is None:
        arguments = _get(tool_call, "arguments", "")
    return f"{name}: {arguments}"


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

    for item in history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n", file=out)
        print(_blue(f"[{timestamp}]", use_colors=use_colors), file=out)

        for msg in messages:
            role_label = f"{msg['role'].capitalize()} message:"
            if msg["role"] == "tool" and msg.get("tool_call_id"):
                role_label = f"Tool message: (tool_call_id={msg['tool_call_id']})"
            print(_red(role_label, use_colors=use_colors), file=out)
            if isinstance(msg.get("content"), str):
                print(msg["content"].strip(), file=out)
            elif msg.get("content") is not None:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
                        if c["type"] == "text":
                            print(c["text"].strip(), file=out)
                        elif c["type"] == "image_url":
                            image_str = ""
                            if "base64" in c["image_url"].get("url", ""):
                                len_base64 = len(c["image_url"]["url"].split("base64,")[1])
                                image_str = (
                                    f"<{c['image_url']['url'].split('base64,')[0]}base64,"
                                    f"<IMAGE BASE 64 ENCODED({len_base64!s})>"
                                )
                            else:
                                image_str = f"<image_url: {c['image_url']['url']}>"
                            print(_blue(image_str.strip(), use_colors=use_colors), file=out)
                        elif c["type"] == "input_audio":
                            audio_format = c["input_audio"]["format"]
                            len_audio = len(c["input_audio"]["data"])
                            audio_str = f"<audio format='{audio_format}' base64-encoded, length={len_audio}>"
                            print(_blue(audio_str.strip(), use_colors=use_colors), file=out)
                        elif c["type"] == "file" or c["type"] == "input_file":
                            file_info = c.get("file", c.get("input_file", {}))
                            filename = file_info.get("filename", "")
                            file_id = file_info.get("file_id", "")
                            file_data = file_info.get("file_data", "")
                            file_str = f"<file: name:{filename}, id:{file_id}, data_length:{len(file_data)}>"
                            print(_blue(file_str.strip(), use_colors=use_colors), file=out)
            if msg.get("tool_calls"):
                print(_red("Tool calls:", use_colors=use_colors), file=out)
                for tool_call in msg["tool_calls"]:
                    print(_green(_format_tool_call(tool_call), use_colors=use_colors), file=out)
            print("\n", file=out)

        if isinstance(outputs[0], dict):
            if outputs[0]["text"]:
                print(_red("Response:", use_colors=use_colors), file=out)
                print(_green(outputs[0]["text"].strip(), use_colors=use_colors), file=out)

            if outputs[0].get("tool_calls"):
                print(_red("Tool calls:", use_colors=use_colors), file=out)
                for tool_call in outputs[0]["tool_calls"]:
                    print(_green(_format_tool_call(tool_call), use_colors=use_colors), file=out)
        else:
            print(_red("Response:", use_colors=use_colors), file=out)
            print(_green(outputs[0].strip(), use_colors=use_colors), file=out)

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end="", use_colors=use_colors), file=out)

    print("\n\n\n", file=out)
