from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
    """A module that adds step-by-step reasoning before predicting the output.

    ``ChainOfThought`` extends a given signature with a ``reasoning`` output field,
    prompting the language model to think step by step before producing its answer.
    This often improves accuracy on tasks that benefit from intermediate reasoning.

    Example::

        cot = dspy.ChainOfThought("question -> answer")
        result = cot(question="What is 2 + 2?")
        print(result.reasoning)
        print(result.answer)
    """

    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ):
        """
        A module that reasons step by step in order to predict the output of a task.

        Args:
            signature (Type[dspy.Signature]): The signature of the module.
            rationale_field (Optional[Union[dspy.OutputField, pydantic.fields.FieldInfo]]): The field that will contain the reasoning.
            rationale_field_type (Type): The type of the rationale field.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = ensure_signature(signature)
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        """Execute chain-of-thought reasoning synchronously.

        Args:
            **kwargs: Keyword arguments matching the signature's input fields.

        Returns:
            Prediction: A ``Prediction`` containing the ``reasoning`` field and all output fields.
        """
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        """Execute chain-of-thought reasoning asynchronously.

        Args:
            **kwargs: Keyword arguments matching the signature's input fields.

        Returns:
            Prediction: A ``Prediction`` containing the ``reasoning`` field and all output fields.
        """
        return await self.predict.acall(**kwargs)
