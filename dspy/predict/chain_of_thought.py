from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
    """A module that reasons step by step before producing the output.

    Extends a given signature with an additional ``reasoning`` output field that
    the language model fills in before answering. This chain-of-thought process
    improves accuracy on tasks that require multi-step logic or explanation.

    Args:
        signature: The DSPy signature defining the module's inputs and outputs.
        rationale_field: An optional custom ``FieldInfo`` for the reasoning field.
            If not provided, a default field with description ``${reasoning}`` is used.
        rationale_field_type: The type annotation for the reasoning field. Defaults
            to ``str``. Ignored if ``rationale_field`` is provided.
        **config: Additional keyword arguments forwarded to the underlying
            ``dspy.Predict`` module.

    Example:
        >>> import dspy
        >>> cot = dspy.ChainOfThought("question -> answer")
        >>> result = cot(question="What is the capital of France?")
        >>> print(result.reasoning)
        >>> print(result.answer)
    """

    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ):
        super().__init__()
        signature = ensure_signature(signature)
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
