from abc import ABC, abstractmethod
from collections import deque
from typing import List, TypedDict, Optional, final

from dsp.utils import settings


class LM(ABC):
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

        self.history = deque(maxlen=settings.max_history)

    @abstractmethod
    def basic_request(self, prompt, **kwargs):
        raise NotImplementedError

    @final
    def push_record(self, prompt: str, choices: List[str], completions: Optional[List[str]] = None, **kwargs):
        self.history.append(dict(prompt=prompt, choices=choices, completions=completions, **kwargs))

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def _color_green(self, text: str, end: str = "\n"):
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)

    def _color_red(self, text: str, end: str = "\n"):
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)

    def format_history(self, n: int = 1, skip: int = 0, colored: bool = False) -> str:
        """
        Formats the last n prompts and their completions.
        :param n: max records to print
        :param skip: skip the last few records in chronological order
        :param colored: whether to differentiate some output using colors
        """
        # TODO: print the valid choice that contains filled output field instead of the first
        last_prompt = None
        printed = []

        for x in self.history[-n - skip - 1:]:
            prompt = x["prompt"]

            if prompt != last_prompt:
                text = x['choices'][0].strip()
                suffix = f' (and {len(x["choices"] - 1)} other completions)' if len(x['choices']) > 1 else ''
                printed.append(
                    prompt +
                    self._color_green(text) if colored else text +
                    self._color_red(suffix) if colored else suffix
                )

            last_prompt = prompt

        if skip > 0:
            printed = printed[:-skip]

        return '\n\n\n'.join(printed)

    def inspect_history(self, n: int = 1, skip: int = 0, colored: bool = True, file=None) -> None:
        """
        Prints the last n prompts and their completions.
        :param n: max records to print
        :param skip: skip the last few records in chronological order
        :param colored: whether to differentiate some output using colors
        :param file: a file descriptor to print to. Defaults to sys.stdout, see 'print' documentation.
        """
        print(self.format_history(n, skip, colored), file=file)

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        raise NotImplementedError

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop('model')

        return self.__class__(model, **kwargs)
