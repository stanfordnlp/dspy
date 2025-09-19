from typing import Any, Optional

import litellm
import pydantic

from dspy.adapters.types.base_type import Type


class Reasoning(Type):
    """Reasoning type in DSPy.

    This type is useful when you want the DSPy output to include the reasoning of the LM. We build this type so that
    DSPy can support the reasoning model and non-reasoning model with the same code.

    This is a str-like type, you can convert a string directly to a Reasoning object, and from DSPy adapters'
    perspective, `Reasoning` is treated as a string.
    """

    content: str

    def format(self):
        return f"{self.content}"

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        if isinstance(data, str):
            return {"content": data}

        if isinstance(data, dict):
            if "content" not in data:
                raise ValueError("`content` field is required for `dspy.Reasoning`")
            if not isinstance(data["content"], str):
                raise ValueError(f"`content` field must be a string, but received type: {type(data['content'])}")
            return {"content": data["content"]}

        raise ValueError(f"Received invalid value for `dspy.Reasoning`: {data}")

    @classmethod
    def is_natively_supported(cls, lm, lm_kwargs) -> bool:
        """Whether the Reasoning type is natively supported by the LM."""
        if not litellm.supports_reasoning(lm.model):
            return False

        if "reasoning_effort" in lm_kwargs:
            # `lm_kwargs` overrides `lm.kwargs`
            reasoning_effort = lm_kwargs["reasoning_effort"]
        elif "reasoning_effort" in lm.kwargs:
            reasoning_effort = lm.kwargs["reasoning_effort"]
        else:
            reasoning_effort = None

        return reasoning_effort is not None

    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Reasoning"]:
        """Parse the LM response into a Reasoning object."""
        if "reasoning_content" in response:
            return Reasoning(content=response["reasoning_content"])
        return None

    def __repr__(self) -> str:
        return f"{self.content!r}"

    def __str__(self) -> str:
        return self.content

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Reasoning):
            return self.content == other.content
        if isinstance(other, str):
            return self.content == other
