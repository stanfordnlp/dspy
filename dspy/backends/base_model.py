from abc import ABC, abstractmethod
from dataclasses import dataclass

from litellm import text_completion


@dataclass
class BaseModel(ABC):
    @abstractmethod
    def __call__(self, prompt: str) -> list[str]:
        """Generate completions for the prompt."""

    @abstractmethod
    def finetune(self, examples: list[tuple[str, str]]) -> "Model":
        """Finetune on examples and return a new model."""


@dataclass
class BaseLM(BaseModel, ABC):
    temperature: float
    n: int
    max_tokens: int


# this kwarg set works for all LiteLLM models except Anyscale, VertexAI, and Petals
# https://docs.litellm.ai/docs/completion/input#translated-openai-params to be implemented
@dataclass
class LiteLLM(BaseLM, ABC):
    top_p: float
    stream: bool = False


@dataclass
class OpenAILM(LiteLLM):
    model: str = "gpt-3.5-turbo"
    n: int
    presence_penalty: float
    frequency_penalty: float

    def __init__(
        self,
        model: str,
        temperature: float,
        n: int,
        max_tokens: int,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
    ):
        super().__init__(temperature, n, max_tokens, top_p)
        self.model = model
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

        self.kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "model": self.model,
            "stream": False,
        }

    def __call__(self, prompt: str) -> list[str]:
        # return completion(prompt, temperature=self.temperature, n=self.n, max_tokens=self.max_tokens)
        return text_completion(prompt, **self.kwargs)

    def finetune(self, examples: list[tuple[str, str]]) -> "OpenAILM":
        # Does nothing, just passing pre-commit
        examples = list(examples)

        return self
