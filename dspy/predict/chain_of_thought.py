from typing import Any, Optional, Type, Union

from pydantic.fields import FieldInfo

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.primitives.module import Module
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, ensure_signature


class ChainOfThought(Module):
    def __init__(
        self,
        signature: Union[str, Type[Signature]],
        rationale_field: Optional[Union[OutputField, FieldInfo]] = None,
        rationale_field_type: Type = str,
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
        self.predict_reasoning = dspy.Predict(signature, **config)
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def _validate_lm(self, **kwargs):
        """Helper method to validate the LM configuration."""
        lm = kwargs.get("lm", dspy.settings.lm)

        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. e.g, "
                "`dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
            )

        if isinstance(lm, str):
            # Many users mistakenly use `dspy.configure(lm="openai/gpt-4o-mini")` instead of
            # `dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))`, so we are providing a specific error message.
            raise ValueError(
                f"LM must be an instance of `dspy.BaseLM`, not a string. Instead of using a string like "
                f"'dspy.configure(lm=\"{lm}\")', please configure the LM like 'dspy.configure(lm=dspy.LM(\"{lm}\"))'"
            )
        elif not isinstance(lm, BaseLM):
            raise ValueError(f"LM must be an instance of `dspy.BaseLM`, not {type(lm)}. Received `lm={lm}`.")

        return lm

    def forward(self, **kwargs):
        lm = self._validate_lm(**kwargs)
        return self.predict(**kwargs) if not lm.reasoning_model else self.predict_reasoning(**kwargs)

    async def aforward(self, **kwargs):
        lm = self._validate_lm(**kwargs)
        return await (
            self.predict.acall(**kwargs) if not lm.reasoning_model else self.predict_reasoning.acall(**kwargs)
        )
