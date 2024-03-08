from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from dspy.signatures.signature import Signature, ensure_signature
from dspy.primitives.prediction import Completions


class BaseBackend(BaseModel, ABC):
    """A backend takes a signature, its params, and returns a list of structured predictions."""

    history: list[Completions] = Field(default=[])

    def __call__(
        self,
        signature: Signature,
        recover: bool = False,
        max_recovery_attempts: int = 5,
        **kwargs,
    ) -> Completions:
        # Recursively complete generation, until at least one complete completion is available.
        signature = ensure_signature(signature)

        i = 0
        completions = None

        output = None
        while i < max_recovery_attempts:
            # Returns a List of Completions
            # which may or may not be complete
            completions = self.generate(signature, **kwargs)

            # If 1 or more complete generations exist, simple return all complete
            if len(completions) > 0:
                break
            elif recover:
                max_example = completions[0].example

                for completion in completions.completions:
                    if len(max_example) < len(completion.example):
                        max_example = completion.example

                # Currently this is only updating the example fields
                # we will want to update kwargs aswell for max_tokens
                # temperature etc.
                for field in signature.fields:
                    if field in max_example:
                        kwargs[field] = max_example.get(field)

                # Update lm arguments
                # Setting temperature to 0.0, leads to greedy decoding
                kwargs["temperature"] = 0.0

                # Cut the max_tokens in half each time if it has been set
                if "max_tokens" in kwargs:
                    kwargs["max_tokens"] = int(kwargs["max_tokens"] / 2)

            else:
                break

            i += 1

        assert completions is not None
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
