import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from dspy.modeling.lm.base import LMOutput
from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.signatures.signature import Signature, ensure_signature


class BaseBackend(BaseModel, ABC):
    """A backend takes a signature, its params, and returns a list of structured predictions."""

    history: list[Completions] = Field(default_factory=list)
    attempts: int = Field(default=1)

    @abstractmethod
    def prepare_request(
        self,
        signature: Signature,
        example: Example,
        config: dict[str, t.Any],
    ) -> dict:
        """Takes params passed to call, and returns kwargs for LM."""
        ...

    @abstractmethod
    def process_response(
        self,
        signature: Signature,
        example: Example,
        output: LMOutput,
        input_kwargs: dict,
    ) -> Completions:
        """Takes output from LM, and generates Completions."""
        ...

    def __call__(
        self,
        signature: Signature,
        attempts: int = 1,
        config: t.Optional[dict[str, t.Any]] = None,
        **kwargs,
    ) -> Completions:
        # Override config provided at initialization with provided config
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

    def generate(
        self,
        signature: Signature,
        demos: t.Optional[list[str]] = None,
        config: t.Optional[dict[str, t.Any]] = None,
        **kwargs,
    ) -> Completions:
        """Generates `n` predictions (complete/partial) for the signature output."""

        if config is None:
            config = {}

        if demos is None:
            demos = []

        # TODO: Move this check to logging
        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(
                f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}.",
            )

        # Generate Example
        example = Example(demos=demos, **kwargs)

        # Get full kwargs for Model
        model_kwargs = self.prepare_request(signature, example, config)

        # Pass Through Language Model
        generations = self.lm(**model_kwargs)

        # This returns a list of Examples
        return self.process_response(signature, example, generations, model_kwargs)
