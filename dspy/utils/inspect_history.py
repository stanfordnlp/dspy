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

    def print_compaction(provider_name: str, content: str | None, *, opaque: bool) -> None:
        print(_red(f"Provider compaction ({provider_name}):", use_colors=use_colors), file=out)
        if content is None:
            label = "<opaque provider state>" if opaque else "<empty compaction>"
            print(_blue(label, use_colors=use_colors), file=out)
        else:
            print(_green(content.strip(), use_colors=use_colors), file=out)

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
                        elif c["type"] == "compaction":
                            print_compaction(
                                c.get("provider_name", "unknown"),
                                c.get("content"),
                                opaque="encrypted_content" in c,
                            )
            print_tool_calls(msg.get("tool_calls"))
            print("\n", file=out)

        first_output = outputs[0]
        if isinstance(first_output, dict):
            if first_output.get("text"):
                print(_red("Response:", use_colors=use_colors), file=out)
                print(_green(first_output["text"].strip(), use_colors=use_colors), file=out)

            print_tool_calls(first_output.get("tool_calls"))
        elif isinstance(first_output, str):
            print(_red("Response:", use_colors=use_colors), file=out)
            print(_green(first_output.strip(), use_colors=use_colors), file=out)
        elif isinstance(first_output, list):
            text = "".join(value for value in first_output if isinstance(value, str))
            if text:
                print(_red("Response:", use_colors=use_colors), file=out)
                print(_green(text.strip(), use_colors=use_colors), file=out)

        response = getattr(item, "response", None)
        if response is not None:
            for compaction in response.compactions:
                print_compaction(
                    compaction.provider_name,
                    compaction.content,
                    opaque="encrypted_content" in compaction.provider_data,
                )

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end="", use_colors=use_colors), file=out)

    print("\n\n\n", file=out)
