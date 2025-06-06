def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _blue(text: str, end: str = "\n"):
    return "\x1b[34m" + str(text) + "\x1b[0m" + end


def _yellow(text: str, end: str = "\n"):
    return "\x1b[33m" + str(text) + "\x1b[0m" + end


def _magenta(text: str, end: str = "\n"):
    return "\x1b[35m" + str(text) + "\x1b[0m" + end


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pretty_print_dict(data: dict):
    """Pretty prints a given dictionary, flattening nested dictionaries and
    coloring keys and values differently."""
    try:
        flattened_data = flatten_dict(data)

        for key, value in flattened_data.items():
            if value:
                print(f"{_yellow(key, end='')}: {_green(repr(value), end='')}")
    except Exception as e:
        print(f"An error occurred during pretty printing: {e}")


def pretty_print_history(history, n: int = 1, verbose: int = 0):
    """Prints the last n prompts and their completions.

    Args:
        history (list): A list of dictionaries containing the history of prompts and completions.
        n (int): The number of most recent entries to print. Defaults to 1.
        verbose (int): Verbosity level.
    """

    for index, item in enumerate(history[-n:]):
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n")
        print(_magenta("History entry: ", end="") + _green(f"{index + 1}"))
        print(_blue(f"[{timestamp}]"))

        if verbose > 0:
            usage = item["response"].get("usage")
            if usage:
                print(_blue(f"Usage"))
                pretty_print_dict(usage.to_dict())
                print("\n")
            if hasattr(item["response"].choices[0], "message"):
                response_message = item["response"].choices[0].message
                if hasattr(response_message, "reasoning_content"):
                    reasoning = response_message.reasoning_content
                    if reasoning:
                        print(_blue(f"Reasoning"))
                        print(_green(reasoning.strip()))
                        print("\n")

        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            if isinstance(msg["content"], str):
                print(msg["content"].strip())
            else:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
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

        print("-" * 50)
    print("\n\n")
