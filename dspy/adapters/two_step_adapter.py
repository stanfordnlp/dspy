from typing import Any

import json_repair

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import ToolCalls
from dspy.adapters.utils import get_field_description_string
from dspy.clients import LM
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature, make_signature

"""
NOTE/TODO/FIXME:

The main issue below is that the second step's signature is entirely created on the fly and is invoked with a chat
adapter explicitly constructed with no demonstrations. This means that it cannot "learn" or get optimized.
"""


class TwoStepAdapter(Adapter):
    """
    A two-stage adapter that:
        1. Uses a simpler, more natural prompt for the main LM
        2. Uses a smaller LM with chat adapter to extract structured data from the response of main LM
    This adapter uses a common __call__ logic defined in base Adapter class.
    This class is particularly useful when interacting with reasoning models as the main LM since reasoning models
    are known to struggle with structured outputs.

    Example:
    ```
    import dspy
    lm = dspy.LM(model="openai/o3-mini", max_tokens=16000, temperature = 1.0)
    adapter = dspy.TwoStepAdapter(dspy.LM("openai/gpt-4o-mini"))
    dspy.configure(lm=lm, adapter=adapter)
    program = dspy.ChainOfThought("question->answer")
    result = program("What is the capital of France?")
    print(result)
    ```
    """

    def __init__(self, extraction_model: LM, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(extraction_model, LM):
            raise ValueError("extraction_model must be an instance of LM")
        self.extraction_model = extraction_model

    def format(
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Format a prompt for the first stage with the main LM.
        This no specific structure is required for the main LM, we customize the format method
        instead of format_field_description or format_field_structure.

        Args:
            signature: The signature of the original task
            demos: A list of demo examples
            inputs: The current input

        Returns:
            A list of messages to be passed to the main LM.
        """
        messages = []

        # Create a task description for the main LM
        task_description = self.format_task_description(signature)
        messages.append({"role": "system", "content": task_description})

        messages.extend(self.format_demos(signature, demos))

        # Format the current input
        messages.append({"role": "user", "content": self.format_user_message_content(signature, inputs)})

        return messages

    def parse(self, signature: Signature, completion: str) -> dict[str, Any]:
        """
        Use a smaller LM (extraction_model) with chat adapter to extract structured data
        from the raw completion text of the main LM.

        Args:
            signature: The signature of the original task
            completion: The completion from the main LM

        Returns:
            A dictionary containing the extracted structured data.
        """
        # The signature is supposed to be "text -> {original output fields}"
        extractor_signature = self._create_extractor_signature(signature)

        try:
            # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
            parsed_result = ChatAdapter()(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": completion},
            )
            return parsed_result[0]

        except Exception as e:
            raise ValueError(f"Failed to parse response from the original completion: {completion}") from e

    async def acall(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs = self.format(signature, demos, inputs)

        outputs = await lm.acall(messages=inputs, **lm_kwargs)
        # The signature is supposed to be "text -> {original output fields}"
        extractor_signature = self._create_extractor_signature(signature)

        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
        for output in outputs:
            output_logprobs = None
            tool_calls = None
            text = output

            if isinstance(output, dict):
                text = output["text"]
                output_logprobs = output.get("logprobs")
                tool_calls = output.get("tool_calls")

            try:
                # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
                value = await ChatAdapter().acall(
                    lm=self.extraction_model,
                    lm_kwargs={},
                    signature=extractor_signature,
                    demos=[],
                    inputs={"text": text},
                )
                value = value[0]

            except Exception as e:
                raise ValueError(f"Failed to parse response from the original completion: {output}") from e

            if tool_calls and tool_call_output_field_name:
                tool_calls = [
                    {
                        "name": v["function"]["name"],
                        "args": json_repair.loads(v["function"]["arguments"]),
                    }
                    for v in tool_calls
                ]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            if output_logprobs is not None:
                value["logprobs"] = output_logprobs

            values.append(value)
        return values

    def format_task_description(self, signature: Signature) -> str:
        """Create a description of the task based on the signature"""
        parts = []

        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append("As input, you will be provided with:\n" + get_field_description_string(signature.input_fields))
        parts.append("Your outputs must contain:\n" + get_field_description_string(signature.output_fields))
        parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")

        if signature.instructions:
            parts.append(f"Specific instructions: {signature.instructions}")

        return "\n".join(parts)

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        parts = [prefix]

        for name in signature.input_fields.keys():
            if name in inputs:
                parts.append(f"{name}: {inputs.get(name, '')}")

        parts.append(suffix)
        return "\n\n".join(parts).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        parts = []

        for name in signature.output_fields.keys():
            if name in outputs:
                parts.append(f"{name}: {outputs.get(name, missing_field_message)}")

        return "\n\n".join(parts).strip()

    def _create_extractor_signature(
        self,
        original_signature: type[Signature],
    ) -> type[Signature]:
        """Create a new signature containing a new 'text' input field and all output fields.

        Args:
            original_signature: The original signature to extract output fields from

        Returns:
            A new Signature type with a text input field and all output fields
        """
        # Create new fields dict with 'text' input field and all output fields
        new_fields = {
            "text": (str, InputField()),
            **{name: (field.annotation, field) for name, field in original_signature.output_fields.items()},
        }

        outputs_str = ", ".join([f"`{field}`" for field in original_signature.output_fields])
        instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. \
            Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field."

        return make_signature(new_fields, instructions)
