from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from dspy.signatures.signature import Signature, ensure_signature
from dspy.primitives.prediction import Completions


class BaseBackend(BaseModel, ABC):
    """A backend takes a signature, its params, and returns a list of structured predictions."""

    history: list[Completions] = Field(default_factory=list)
    attempts: int = Field(default=1)

    def __call__(
        self,
        signature: Signature,
        attempts: int = 1,
        **kwargs,
    ) -> Completions:
        # Allow overriding the attempts at the Backend Initialization Step
        attempts = max(attempts, self.attempts)

        # Recursively complete generation, until at least one complete completion is available.
        signature = ensure_signature(signature)

        i = 0
        completions = None

        while i < attempts:
            # Returns a List of Completions
            # which may or may not be complete
            completions = self.generate(signature, **kwargs)

            # If 1 or more complete generations exist, simple return all complete
            if completions.has_complete_example():
                break

            max_example = completions.get_farthest_example()

            for field in signature.fields:
                if field in max_example:
                    kwargs[field] = max_example.get(field)

            # Setting temperature to 0.0, leads to greedy decoding
            kwargs["temperature"] = 0.0

            # Cut the max_tokens in half each time if it has been set
            if "max_tokens" in kwargs:
                kwargs["max_tokens"] = max(1, int(kwargs["max_tokens"] / 2))

            i += 1

        assert completions is not None
        completions.remove_incomplete()
        if len(completions) == 0:
            raise Exception(
                "Generation failed, recursively attempts to complete did not succeed."
            )

        self.history.append(completions)

        return completions

    @abstractmethod
    def generate(
        self,
        signature: Signature,
        **kwargs,
    ) -> Completions:
        """Generates `n` predictions (complete/partial) for the signature output."""
