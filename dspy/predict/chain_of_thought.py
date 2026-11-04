from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
    """A module that reasons before producing the output.

    Uses native reasoning when supported, otherwise requests a short plan before
    answering. The result exposes either as ``reasoning``; native reasoning may
    be ``None`` when the provider does not return a summary.

    Args:
        signature: The DSPy signature defining the module's inputs and outputs.
        rationale_field: An optional custom ``FieldInfo`` for the reasoning field.
            Its annotation and description are preserved.
        rationale_field_type: Defaults to ``dspy.Reasoning``. Pass ``str`` to
            request an explicit reasoning field instead. Ignored if
            ``rationale_field`` is provided.
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
        rationale_field_type: type | None = None,
        **config: dict[str, Any],
    ):
        super().__init__()
        signature = ensure_signature(signature)
        rationale_field_type = rationale_field_type or dspy.Reasoning
        use_default = rationale_field is None and rationale_field_type is dspy.Reasoning
        desc = "A short plan for solving the task." if use_default else "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        if use_default:
            extended_signature = extended_signature.with_updated_fields("reasoning", __dspy_reasoning_fallback="plan")
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
