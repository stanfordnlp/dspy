import os
import pprint
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from joblib import Memory
from pydantic import BaseModel, Field

import dspy
from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.signatures.signature import Signature, ensure_signature

_cachedir = os.environ.get("DSP_CACHEDIR") or str(Path.home() / ".joblib_cache")
_cache_memory = Memory(_cachedir, verbose=0)


class BaseBackend(BaseModel, ABC):
    history: list[Completions] = Field(default_factory=list, exclude=True)
    attempts: int = Field(default=1)

    @abstractmethod
    def prepare_request(self, signature: Signature, example: Example, config: dict, **kwargs) -> dict:
        """Given a Signature, Example, and Config kwargs, provide a dictionary of arguments for the Backend."""
        ...

    @abstractmethod
    def process_response(
        self,
        signature: Signature,
        example: Example,
        response: t.Any,
        input_kwargs: dict,
        **kwargs,
    ) -> Completions:
        """Given a Signature, Example, and Generated Output, process generations and return completions."""
        ...

    @abstractmethod
    def make_request(self, **kwargs) -> t.Any:
        ...

    def generate(self, signature: Signature, demos: list[str], config: dict[str, t.Any], **kwargs) -> Completions:
        """Generates predictions (complete/partial) for the signature output."""

        # Generate Example
        example = Example(demos=demos, **kwargs)

        # Get Full Kwargs for Model
        model_kwargs = self.prepare_request(signature, example, config)

        if dspy.settings.get("cache", False):
            response = cached_request(self, **model_kwargs)
        else:
            response = self.make_request(**model_kwargs)

        return self.process_response(signature, example, response, model_kwargs)

    def __call__(
        self,
        signature: Signature,
        attempts: int = 1,
        config: t.Optional[dict[str, t.Any]] = None,
        **kwargs,
    ) -> Completions:
        """Recursively generates and checks completions for completeness, returning once complete."""
        if config is None:
            config = {}

        # Allow overriding the attempts at the Backend Initialization Step
        attempts = max(attempts, self.attempts)
        if attempts < 1:
            raise ValueError("'attempts' argument passed must be greater than 0.")

        # Recursively complete generation, until at least one complete completion is available.
        signature = ensure_signature(signature)

        i = 0
        completions = None

        while i < attempts:
            # Returns a List of Completions
            # which may or may not be complete
            completions = self.generate(signature=signature, config=config, **kwargs)

            # If 1 or more complete generations exist, simple return all complete
            if completions.has_complete_example():
                break

            # If a partial example was generated then update for all generated values
            if len(completions.examples) > 0:
                max_example = completions.get_farthest_example()

                for field in signature.fields:
                    if field in max_example:
                        kwargs[field] = max_example.get(field)

            # Setting temperature to 0.0, leads to greedy decoding
            config["temperature"] = 0.0

            # Cut the max_tokens in half each time if it has been set
            if "max_tokens" in kwargs:
                config["max_tokens"] = max(1, int(config["max_tokens"] / 2))

            i += 1

        completions.remove_incomplete()
        if len(completions) == 0:
            raise Exception(
                "Generation failed, recursively attempts to complete did not succeed.",
            )

        self.history.append(completions)

        return completions

    def inspect_history(self, element: int) -> None:
        """Index into the backend historical completions, and pretty print all details."""

        # Print Input Kwargs
        print("===INPUT KWARGS===")

        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.history[element]["input_kwargs"])

        # Print Completions
        print("\n===COMPLETIONS===")
        print(self.history[-1])


def cached_request(cls: BaseBackend, **kwargs) -> Completions:
    hashed = joblib.hash(cls.model_dump_json())

    @_cache_memory.cache(ignore=["cls"])
    def _cache_call(cls: BaseBackend, hashed: str, **kwargs):
        return cls.make_request(**kwargs)

    return _cache_call(cls=cls, hashed=hashed, **kwargs)
