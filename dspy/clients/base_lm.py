from abc import ABC, abstractmethod

GLOBAL_HISTORY = []

class BaseLM(ABC):
    def __init__(self, model, model_type='chat', temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    @abstractmethod
    def __call__(self, prompt=None, messages=None, **kwargs):
        pass

    def inspect_history(self, n: int = 1):
        _inspect_history(self.history, n)

    def update_global_history(self, entry):
        GLOBAL_HISTORY.append(entry)


def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end

def _blue(text: str, end: str = "\n"):
    return "\x1b[34m" + str(text) + "\x1b[0m" + end


def _inspect_history(history, n: int = 1):
    """Prints the last n prompts and their completions."""

    for item in history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n")
        print("\x1b[34m" + f"[{timestamp}]" + "\x1b[0m" + "\n")

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
                                image_str = f"<{c['image_url']['url'].split('base64,')[0]}base64,<IMAGE BASE 64 ENCODED({str(len_base64)})>"
                            else:
                                image_str = f"<image_url: {c['image_url']['url']}>"
                            print(_blue(image_str.strip()))
            print("\n")

        print(_red("Response:"))
        print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs)-1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")

def inspect_history(n: int = 1):
    """The global history shared across all LMs."""
    return _inspect_history(GLOBAL_HISTORY, n)