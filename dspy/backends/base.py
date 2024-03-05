from abc import ABC, abstractmethod
import typing as t

from pydantic import BaseModel
from dspy.primitives.prediction import Completions
from dspy.primitives.example import Example

from dspy.signatures.signature import Signature


class Completion:
    def __init__(self, example: Example, complete: bool):
        self.example = example
        self.complete = complete

    def __len__(self) -> int:
        return len(self.example.keys())


def convert_to_completion(signature: Signature, example: Example) -> Completion:
    complete = True
    for field in signature.output_fields:
        if field not in example:
            complete = False

    return Completion(example=example, complete=complete)


class BaseBackend(BaseModel, ABC):
    """A backend takes a signature, its params, and returns a list of structured predictions."""

    def __call__(
        self, signature: Signature, recover: bool = False, **kwargs
    ) -> Completions:
        # Recursively complete generation, until at least one complete completion is available.

        complete_examples = None
        max_generations = 5
        i = 0

        while i < max_generations:
            # Returns a List of Completions
            # which may or may not be complete
            output = self.generate(signature, **kwargs)

            # Filter for only complete completions
            complete_examples = [
                completion.example for completion in output if completion.complete
            ]

            # If 1 or more complete generations exist, simply return all
            if len(complete_examples) > 0:
                print(
                    "BREAKING DURING RECOVERY AS AT LEAST ONE COMPLETE EXAMPLE EXISTS"
                )
                break
            # if not, recursively generation with the furthest generation as the example
            elif recover:
                max_example = output[0].example

                for completion in output:
                    if len(max_example) < len(completion.example):
                        max_example = completion.example

                # Currently this is only updating the example fields
                # we will want to update kwargs aswell for max_tokens
                # temperature etc.
                for field in signature.fields:
                    if field in max_example:
                        kwargs[field] = max_example.get(field)
            else:
                break

            i += 1

        if complete_examples is None:
            raise Exception(
                "Generation failed, recursively attempts to complete did not succeed."
            )

        return Completions(complete_examples)

    @abstractmethod
    def generate(
        self,
        signature: Signature,
        **kwargs,
    ) -> t.List[Completion]:
        """Generates `n` predictions (complete/partial) for the signature output."""
