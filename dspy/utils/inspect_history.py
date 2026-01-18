def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _blue(text: str, end: str = "\n"):
    return "\x1b[34m" + str(text) + "\x1b[0m" + end


def _yellow(text: str, end: str = "\n"):
    return "\x1b[33m" + str(text) + "\x1b[0m" + end


def _cyan(text: str, end: str = "\n"):
    return "\x1b[36m" + str(text) + "\x1b[0m" + end


def pretty_print_history(history, n: int = 1):
    """Prints the last n prompts and their completions."""

    for item in history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n")
        print("\x1b[34m" + f"[{timestamp}]" + "\x1b[0m" + "\n")

        for msg in messages:
            role = msg.get("role", "unknown")

            # Handle tool response messages
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                print(_yellow(f"Tool response (id: {tool_call_id}):"))
                content = msg.get("content", "")
                if content:
                    print(content.strip() if isinstance(content, str) else str(content))
                print("\n")
                continue

            print(_red(f"{role.capitalize()} message:"))

            # Handle tool_calls in assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                content = msg.get("content")
                if content:
                    print(content.strip() if isinstance(content, str) else str(content))
                print(_cyan("Tool calls:"))
                for tool_call in msg["tool_calls"]:
                    func = tool_call.get("function", {})
                    tool_id = tool_call.get("id", "unknown")
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    print(_cyan(f"  [{tool_id}] {name}({args})"))
                print("\n")
                continue

            content = msg.get("content")
            if content is None:
                print("<no content>")
            elif isinstance(content, str):
                print(content.strip())
            elif isinstance(content, list):
                for c in content:
                    if c["type"] == "text":
                        print(c["text"].strip())
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
                        print(_blue(image_str.strip()))
                    elif c["type"] == "input_audio":
                        audio_format = c["input_audio"]["format"]
                        len_audio = len(c["input_audio"]["data"])
                        audio_str = f"<audio format='{audio_format}' base64-encoded, length={len_audio}>"
                        print(_blue(audio_str.strip()))
                    elif c["type"] == "file" or c["type"] == "input_file":
                        file = c.get("file", c.get("input_file", {}))
                        filename = file.get("filename", "")
                        file_id = file.get("file_id", "")
                        file_data = file.get("file_data", "")
                        file_str = f"<file: name:{filename}, id:{file_id}, data_length:{len(file_data)}>"
                        print(_blue(file_str.strip()))
            print("\n")

        if isinstance(outputs[0], dict):
            if outputs[0]["text"]:
                print(_red("Response:"))
                print(_green(outputs[0]["text"].strip()))

            if outputs[0].get("tool_calls"):
                print(_red("Tool calls:"))
                for tool_call in outputs[0]["tool_calls"]:
                    print(_green(f"{tool_call['function']['name']}: {tool_call['function']['arguments']}"))
        else:
            print(_red("Response:"))
            print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")
