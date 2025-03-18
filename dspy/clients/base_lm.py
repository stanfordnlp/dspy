import datetime
import uuid
from abc import ABC

from dspy.dsp.utils import settings
from dspy.utils.callback import with_callbacks

GLOBAL_HISTORY = []


class BaseLM(ABC):
    """Base class for handling LLM calls.

    Most users can directly use the `dspy.LM` class, which is a subclass of `BaseLM`. Users can also implement their
    own subclasses of `BaseLM` to support custom LLM providers and inject custom logic. To do so, simply override the
    `forward` method and make sure the return format is identical to the
    [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).

    Example:

    ```python
    from openai import OpenAI

    import dspy


    class MyLM(dspy.BaseLM):
        def forward(self, prompt, messages=None, **kwargs):
            client = OpenAI()
            return client.chat.completions.create(
                model=self.model,
                messages=messages or [{"role": "user", "content": prompt}],
                **self.kwargs,
            )


    lm = MyLM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    print(dspy.Predict("q->a")(q="Why did the chicken cross the kitchen?"))
    ```
    """

    def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        response = self.forward(prompt=prompt, messages=messages, **kwargs)

        if kwargs.get("logprobs"):
            outputs = [
                {
                    "text": c.message.content if hasattr(c, "message") else c["text"],
                    "logprobs": c.logprobs if hasattr(c, "logprobs") else c["logprobs"],
                }
                for c in response.choices
            ]
        else:
            outputs = [c.message.content if hasattr(c, "message") else c["text"] for c in response.choices]

        if settings.disable_history:
            return outputs

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": response,
            "outputs": outputs,
            "usage": dict(response.usage),
            "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "response_model": response.model,
            "model_type": self.model_type,
        }
        self.history.append(entry)
        self.update_global_history(entry)

        return outputs

    def forward(self, prompt=None, messages=None, **kwargs):
        """Forward pass for the language model.

        Subclasses must implement this method, and the response should be identical to
        [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters."""

        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                new_instance.kwargs[key] = value

        return new_instance

    def inspect_history(self, n: int = 1):
        _inspect_history(self.history, n)

    def update_global_history(self, entry):
        if settings.disable_history:
            return

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
