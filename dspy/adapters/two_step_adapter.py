from typing import Any

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import ToolCalls, Type
from dspy.adapters.utils import get_field_description_string
from dspy.clients.base_lm import BaseLM
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature, make_signature
from dspy.utils.exceptions import AdapterParseError

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

    def __init__(self, extraction_model: BaseLM, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(extraction_model, BaseLM):
            raise ValueError("extraction_model must be an instance of dspy.BaseLM")
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

    async def aparse(self, signature: Signature, completion: str) -> dict[str, Any]:
        extractor_signature = self._create_extractor_signature(signature)

        try:
            parsed_result = await ChatAdapter().acall(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": completion},
            )
            return parsed_result[0]

        except Exception as e:
            raise ValueError(f"Failed to parse response from the original completion: {completion}") from e

    async def _aparse_response(self, state, response, lm=None, request=None) -> list[dict[str, Any]]:
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(state.source_signature)

        for output in response.outputs:
            if output.metadata.get("empty_legacy_outputs"):
                continue

            value: dict[str, Any] = {}
            parsed_any = False

            if output.text and state.render_signature.output_fields:
                value.update(await self.aparse(state.render_signature, output.text))
                parsed_any = True

            if output.tool_calls and tool_call_output_field_name:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [
                        {
                            "name": tool_call.name,
                            "args": tool_call.args,
                        }
                        for tool_call in output.tool_calls
                    ]
                )
                parsed_any = True

            output_dict = output.to_output_dict()
            legacy_output = output.provider_output if output.provider_output is not None else output_dict
            for name, field_info in state.source_signature.output_fields.items():
                if (
                    isinstance(field_info.annotation, type)
                    and field_info.annotation in self.native_response_types
                    and issubclass(field_info.annotation, Type)
                ):
                    parsed_value = field_info.annotation.parse_lm_response(legacy_output)
                    if parsed_value is not None:
                        value[name] = parsed_value
                        parsed_any = True

            for field_name in state.source_signature.output_fields:
                value.setdefault(field_name, None)

            if not parsed_any:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=state.source_signature,
                    lm_response=str(output_dict),
                    message="The LM returned an empty or null response.",
                )

            if output.logprobs is not None:
                value["logprobs"] = output.logprobs

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
        main_request: bool = False,
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
