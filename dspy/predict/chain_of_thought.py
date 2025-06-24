from functools import cached_property
from typing import Any, Optional, Type, Union

from pydantic.fields import FieldInfo

import dspy
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
        self._signature = ensure_signature(signature)
        self._config = config
        self._rationale_field = rationale_field
        self._rationale_field_type = rationale_field_type

    @cached_property
    def predict(self):
        """Returns the appropriate predict instance based on the LM's reasoning model capability."""
        lm = dspy.settings.lm
        if lm and getattr(lm, "reasoning_model", False):
            return dspy.Predict(self._signature, **self._config)
        else:
            prefix = "Reasoning: Let's think step by step in order to"
            desc = "${reasoning}"
            rationale_field_type = (
                self._rationale_field.annotation if self._rationale_field else self._rationale_field_type
            )
            rationale_field = (
                self._rationale_field if self._rationale_field else dspy.OutputField(prefix=prefix, desc=desc)
            )
            extended_signature = self._signature.prepend(
                name="reasoning", field=rationale_field, type_=rationale_field_type
            )
            return dspy.Predict(extended_signature, **self._config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)

    def load_state(self, state):
        """Override to ensure predict parameter is created before loading state."""
        # If predict state exists but predict hasn't been accessed yet, access it first
        if "predict" in state and "predict" not in self.__dict__:
            _ = self.predict  # This creates the predict instance

        # Now call the base load_state which will load into all named_parameters
        return super().load_state(state)

    def __setstate__(self, state):
        """Custom deserialization for cloudpickle to preserve predict instance."""
        # Restore the state normally
        self.__dict__.update(state)

        # If predict was cached and serialized, we don't need to do anything special
        # since cloudpickle should have preserved it correctly

    def __getstate__(self):
        """Custom serialization for cloudpickle to ensure predict instance is preserved."""
        state = self.__dict__.copy()
        # Force evaluation of cached property if not already done
        if "predict" not in state:
            # Access the predict property to cache it before serialization
            _ = self.predict
            state = self.__dict__.copy()
        return state

    def named_parameters(self):
        """Override to ensure the predict property is cached and included in named parameters."""
        # Force evaluation of the cached_property if not already done
        # This ensures it gets stored in __dict__ and picked up by the base implementation
        if "predict" not in self.__dict__:
            try:
                _ = self.predict  # This triggers the cached_property
            except Exception:
                # If accessing predict fails for any reason, continue without it
                pass

        # Now call the base implementation which will include the cached predict
        return super().named_parameters()
