from typing import Any

import json_repair

from dspy.adapters.base import Adapter, _uses_language_model_contract
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import ToolCalls
from dspy.adapters.utils import get_field_description_string
from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.types import LMOutput
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

    Examples:
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

    def __init__(self, extraction_model: BaseLM | LanguageModel, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(extraction_model, (BaseLM, LanguageModel)) and not callable(extraction_model):
            raise ValueError("extraction_model must be an instance of dspy.BaseLM, dspy.LanguageModel, or a callable LM")
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
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs = self.format(signature, demos, inputs)

        if _uses_language_model_contract(lm):
            request = self._language_model_request(lm, inputs, lm_kwargs)
            response = await lm.acall(request=request)
            return await self._acall_postprocess_language_model(signature, response.outputs)

        outputs = await lm.acall(messages=inputs, **lm_kwargs)
        return await self._acall_postprocess_legacy(signature, outputs)

    async def _acall_postprocess_language_model(
        self,
        signature: type[Signature],
        outputs: list[LMOutput],
    ) -> list[dict[str, Any]]:
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
        for output in outputs:
            extraction_signature = (
                signature.delete(tool_call_output_field_name)
                if output.tool_calls and tool_call_output_field_name
                else signature
            )
            value = await self._aextract_structured_fields(extraction_signature, output.text or "")

            if output.tool_calls and tool_call_output_field_name:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [{"name": call.name, "args": call.args} for call in output.tool_calls]
                )

            if output.logprobs is not None:
                value["logprobs"] = output.logprobs

            values.append(value)
        return values

    async def _acall_postprocess_legacy(
        self,
        signature: type[Signature],
        outputs: list[dict[str, Any] | str],
    ) -> list[dict[str, Any]]:
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

            extraction_signature = signature.delete(tool_call_output_field_name) if tool_calls and tool_call_output_field_name else signature
            value = await self._aextract_structured_fields(extraction_signature, text or "")

            if tool_calls and tool_call_output_field_name:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [
                        {
                            "name": call["function"]["name"],
                            "args": json_repair.loads(call["function"]["arguments"]),
                        }
                        for call in tool_calls
                    ]
                )

            if output_logprobs is not None:
                value["logprobs"] = output_logprobs

            values.append(value)
        return values

    async def _aextract_structured_fields(self, signature: type[Signature], text: str) -> dict[str, Any]:
        extractor_signature = self._create_extractor_signature(signature)
        try:
            value = await ChatAdapter().acall(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": text},
            )
            return value[0]
        except Exception as e:
            raise ValueError(f"Failed to parse response from the original completion: {text}") from e

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
