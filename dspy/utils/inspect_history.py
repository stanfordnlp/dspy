from __future__ import annotations

import sys
from typing import IO, Any


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


def pretty_print_history(history: list[dict[str, Any]], n: int = 1, file: IO[str] | None = None) -> None:
    """Print the last n prompts and their completions.

    Args:
        history: The history list to print from.
        n: Number of recent entries to display. Defaults to 1.
        file: An optional file-like object to write output to (must have a
            ``.write()`` method). When provided, ANSI color codes are
            automatically disabled. Defaults to ``None`` (prints to stdout).
    """
    out = file or sys.stdout
    colors = file is None

    for item in history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n", file=out)
        if colors:
            print("\x1b[34m" + f"[{timestamp}]" + "\x1b[0m" + "\n", file=out)
        else:
            print(f"[{timestamp}]\n", file=out)

        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:", use_colors=colors), file=out)
            if isinstance(msg["content"], str):
                print(msg["content"].strip(), file=out)
            else:
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
                            print(_blue(image_str.strip(), use_colors=colors), file=out)
                        elif c["type"] == "input_audio":
                            audio_format = c["input_audio"]["format"]
                            len_audio = len(c["input_audio"]["data"])
                            audio_str = f"<audio format='{audio_format}' base64-encoded, length={len_audio}>"
                            print(_blue(audio_str.strip(), use_colors=colors), file=out)
                        elif c["type"] == "file" or c["type"] == "input_file":
                            file_info = c.get("file", c.get("input_file", {}))
                            filename = file_info.get("filename", "")
                            file_id = file_info.get("file_id", "")
                            file_data = file_info.get("file_data", "")
                            file_str = f"<file: name:{filename}, id:{file_id}, data_length:{len(file_data)}>"
                            print(_blue(file_str.strip(), use_colors=colors), file=out)
            print("\n", file=out)

        if isinstance(outputs[0], dict):
            if outputs[0]["text"]:
                print(_red("Response:", use_colors=colors), file=out)
                print(_green(outputs[0]["text"].strip(), use_colors=colors), file=out)

            if outputs[0].get("tool_calls"):
                print(_red("Tool calls:", use_colors=colors), file=out)
                for tool_call in outputs[0]["tool_calls"]:
                    print(_green(f"{tool_call['function']['name']}: {tool_call['function']['arguments']}", use_colors=colors), file=out)
        else:
            print(_red("Response:", use_colors=colors), file=out)
            print(_green(outputs[0].strip(), use_colors=colors), file=out)

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end="", use_colors=colors), file=out)

    print("\n\n\n", file=out)
