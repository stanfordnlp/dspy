import logging
from typing import Any

import dspy
from dspy.primitives.module import Module
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, ensure_signature

logger = logging.getLogger(__name__)


class ChainOfThought(Module):
    def __init__(
        self,
        signature: str | type[Signature],
        **config: dict[str, Any],
    ):
        """
        A module that reasons step by step in order to predict the output of a task.

        Args:
            signature (Type[dspy.Signature]): The signature of the module.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = ensure_signature(signature)

        if "rationale_field" in config or "rationale_field_type" in config:
            logger.warning("`rationale_field` and `rationale_field_type` are deprecated, they are no-op now.")

        from dspy.adapters.types.reasoning import Reasoning

        extended_signature = signature.prepend(name="reasoning", field=OutputField(), type_=Reasoning)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
