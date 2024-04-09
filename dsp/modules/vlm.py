from abc import abstractmethod
from typing import Any

from dsp.modules.lm import LM


class VLM(LM):
    """Abstract class for language models."""

    def __init__(self, model):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self.provider = "default"

        self.history = []

    @abstractmethod
    def basic_request(self, prompt, image, **kwargs) -> Any:
        pass

    def request(self, prompt, image, **kwargs) -> Any:
        return self.basic_request(prompt, image **kwargs)

    def print_green(self, text: str, end: str = "\n") -> None:
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)  # noqa: T201

    def print_red(self, text: str, end: str = "\n") -> None:
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)  # noqa: T201

    def inspect_history(self, n: int = 1, skip: int = 0) -> None:
        """Prints the last n prompts and their completions.

        TODO: print the valid choice that contains filled output field instead of the first.
        """
        provider: str = self.provider

        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]
            image = x["image"]
            if prompt != last_prompt:
                if provider == "clarifai" or provider == "google":
                    printed.append((prompt, image, x["response"]))
                elif provider == "anthropic":
                    blocks = [{"text": block.text} for block in x["response"].content if block.type == "text"]
                    printed.append((prompt,image, blocks))
                elif provider == "cohere":
                    printed.append((prompt,image, x["response"].generations))
                else:
                    printed.append((prompt,image, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        for idx, (prompt, image, choices) in enumerate(reversed(printed)):
            # skip the first `skip` prompts
            if (n - idx - 1) < skip:
                continue

            print("\n\n\n")  # noqa: T201
            print(prompt, image, end="")  # noqa: T201
            text = ""
            if provider == "cohere":
                text = choices[0].text
            elif provider == "openai" or provider == "ollama":
                text = " " + self._get_choice_text(choices[0]).strip()
            elif provider == "clarifai":
                text = choices
            elif provider == "google":
                text = choices[0].parts[0].text
            else:
                text = choices[0]["text"]
            self.print_green(text, end="")

            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices)-1} other completions)", end="")
            print("\n\n\n")  # noqa: T201

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass

    def copy(self, **kwargs) -> "VLM":
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(model=model, **kwargs)
