from typing import TYPE_CHECKING, Any, Optional

import litellm
import pydantic

from dspy.adapters.types.base_type import Type

if TYPE_CHECKING:
    from dspy.clients.lm import LM
    from dspy.signatures.signature import Signature


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
    def adapt_to_native_lm_feature(
        cls,
        signature: type["Signature"],
        field_name: str,
        lm: "LM",
        lm_kwargs: dict[str, Any],
    ) -> type["Signature"]:
        if "reasoning_effort" in lm_kwargs:
            # `lm_kwargs` overrides `lm.kwargs`.
            reasoning_effort = lm_kwargs["reasoning_effort"]
        elif "reasoning_effort" in lm.kwargs:
            reasoning_effort = lm.kwargs["reasoning_effort"]
        else:
            # Turn on the native reasoning explicitly if Reasoning field is present in the signature and no explicit
            # reasoning effort is set in `lm_kwargs` or `lm.kwargs`.
            reasoning_effort = "low"

        if reasoning_effort is None or not litellm.supports_reasoning(lm.model):
            # If users explicitly set `reasoning_effort` to None or the LM doesn't support reasoning, we don't enable
            # native reasoning.
            return signature

        if "gpt-5" in lm.model and lm.model_type == "chat":
            # There is a caveat of Litellm as 1.79.0 that when using the chat completion API on GPT-5 family models,
            # the reasoning content is not available in the response. As a workaround, we don't enable the native
            # reasoning feature for GPT-5 family models when using the chat completion API.
            # Litellm issue: https://github.com/BerriAI/litellm/issues/14748
            return signature

        lm_kwargs["reasoning_effort"] = reasoning_effort
        # Delete the reasoning field from the signature to use the native reasoning feature.
        return signature.delete(field_name)

    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Reasoning"]:
        """Parse the LM response into a Reasoning object."""
        if "reasoning_content" in response:
            return Reasoning(content=response["reasoning_content"])
        return None

    @classmethod
    def parse_stream_chunk(cls, chunk) -> str | None:
        """
        Parse a stream chunk into reasoning content if available.

        Args:
            chunk: A stream chunk from the LM.

        Returns:
            The reasoning content (str) if available, None otherwise.
        """
        try:
            if choices := getattr(chunk, "choices", None):
                return getattr(choices[0].delta, "reasoning_content", None)
        except Exception:
            return None

    @classmethod
    def is_streamable(cls) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.content!r}"

    def __str__(self) -> str:
        return self.content

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Reasoning):
            return self.content == other.content
        if isinstance(other, str):
            return self.content == other
