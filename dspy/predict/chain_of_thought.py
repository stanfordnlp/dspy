from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
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
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        """Execute chain-of-thought reasoning and return a prediction.

        Delegates to the internal :class:`dspy.Predict` module whose signature
        has been extended with a ``reasoning`` output field. The returned
        :class:`dspy.Prediction` therefore contains both the reasoning trace
        and the final output fields.

        Keyword Args:
            **kwargs: Input fields matching the module's signature.

        Returns:
            A :class:`dspy.Prediction` with ``reasoning`` and the signature's
            output fields.

        Example:

            .. code-block:: python

                cot = dspy.ChainOfThought("question -> answer")
                pred = cot(question="What is the capital of France?")
                print(pred.reasoning)
                print(pred.answer)
        """
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        """Asynchronous version of :meth:`forward`.

        Accepts the same keyword arguments and returns the same
        :class:`dspy.Prediction` type.
        """
        return await self.predict.acall(**kwargs)
