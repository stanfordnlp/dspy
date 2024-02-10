# Below is a sketch of what the v3 API might look like.

from dspy import lm, program

"""
Idea being that people use 'program' or 'function' all the time in their
code, so it's a familiar concept, vs. 'signatures' is a bit more abstract.

Other names to consider:
- functions, Function -- explains it very clearly
- procs, Proc(edure) -- meh

Other notes:
1. Why not call metrics `loss`? it's a familiar term
2. Also, we should avoid having a .settings singleton. like openai v0.X => v1.X
"""

from dspy.contrib.programs import ChainOfThought
import openai

client = openai.Client(api_key="sk-1234", base_url="...")
openai = lm.OpenAI(client=client, temperature=0.3, n=3, max_tokens=512)
gen_answer = ChainOfThought("context, question -> answer", lm=openai)


class GenSearchQuery(program.Program):
    context = program.Input(description="may contain relevant facts")
    question = program.Input()
    answer = program.Output()


gen_answer = ChainOfThought(GenSearchQuery, model=openai)


# Below is some starter code for the v3 API. It's not complete, but it's a start.

from abc import ABC, abstractmethod
from dataclasses import dataclass
import openai


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


# exported from lm module
class OpenAI(BaseLM):
    """Passing 'complete' here lets us give end-users maximum flexibility.

    For example,

    ```
    from openai import Client
    from functools import partial

    client = Client(
        api_key="sk-1234",
        base_url="..."
    )

    model = lm.OpenAI(
        client=client,
        temperature=0.3,
        n=3,
        max_tokens=512,
    )
    ```
    """

    client: openai.Client

    def __call__(self, prompt: str) -> list[str]:
        """Generate completions for the prompt."""
        completion = self.client.chat.completions.create(
            prompt=prompt,
            temperature=self.temperature,  # May be better to just pass these as kwargs into __call__
            n=self.n,
            max_tokens=self.max_tokens,
        )
        return [choice["text"] for choice in completion.choices]


class BaseRM(BaseModel, ABC):
    """Let the user decide how to implement this however they want."""

    # Q: Why do we need a distinction between RMs and LMs?
    # Seeing as they're both programs of of the form prompt -> list[str]
