import re
import textwrap
from typing import Any, Dict, NamedTuple, Optional, Type

from litellm import ContextWindowExceededError
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    translate_field_type,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


class ChatAdapter(Adapter):
    def __init__(self, callbacks: Optional[list[BaseCallback]] = None):
        super().__init__(callbacks)

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, ContextWindowExceededError) or isinstance(self, JSONAdapter):
                # On context window exceeded error, we don't want to retry with a different adapter.
                raise e
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    def format_field_description(self, signature: Type[Signature]) -> str:
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )

    def format_field_structure(self, signature: Type[Signature]) -> str:
        """
        `ChatAdapter` requires input and output fields to be in their own sections, with section header using markers
        `[[ ## field_name ## ]]`.

        For example:
        ```
        import dspy

        class MySignature(dspy.Signature):
            text: str = dspy.InputField(description="The text to analyze")
            context: str = dspy.InputField(description="The context of the text")
            sentiment: int = dspy.OutputField(description="The sentiment of the text")

        print(dspy.ChatAdapter().format_field_structure(MySignature))
        ```

        The above code will output:
        ```
        All interactions will be structured in the following way, with the appropriate values filled in.

        [[ ## text ## ]]
        {text}

        [[ ## context ## ]]
        {context}

        [[ ## sentiment ## ]]
        {sentiment}        # note: the value you produce must be a single int value

        [[ ## completed ## ]]
        ```
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: Dict[str, FieldInfo]):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        parts.append("[[ ## completed ## ]]\n")
        return "\n\n".join(parts).strip()

    def format_task_description(self, signature: Type[Signature]) -> str:
        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def format_user_message_content(
        self,
        signature: Type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Format the user message content.

        This method formats the user message content, which can be used in formatting few-shot examples, conversation
        history, and the current input.

        Args:
            signature: The DSPy signature for which to format the user message content.
            inputs: The input arguments to the DSPy module.
            prefix: A prefix to the user message content.
            suffix: A suffix to the user message content.

        Returns:
            A string that contains the user message content.
        """
        messages = [prefix]
        for k, v in signature.input_fields.items():
            value = inputs[k]
            formatted_field_value = format_field_value(field_info=v, value=value)
            messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        output_requirements = self.user_message_output_requirements(signature)
        if output_requirements is not None:
            messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def user_message_output_requirements(self, signature: Type[Signature]) -> str:
        """
        In `ChatAdapter`, this output requirement is a simplified version of the `format_field_structure`. See below
        for an example.

        ```
        import dspy

        class MySignature(dspy.Signature):
            text: str = dspy.InputField(description="The text to analyze")
            context: str = dspy.InputField(description="The context of the text")
            sentiment: int = dspy.OutputField(description="The sentiment of the text")

        print(dspy.ChatAdapter().user_message_output_requirements(MySignature))
        ```

        The above code will output:
        ```
        Respond with the corresponding output fields, starting with the field `[[ ## sentiment ## ]]` (must be
        formatted as a valid Python int), and then ending with the marker for `[[ ## completed ## ]]`.
        ```

        The above message is part of the user message, see below for the full user message:

        ```
        import dspy

        class MySignature(dspy.Signature):
            text: str = dspy.InputField(description="The text to analyze")
            context: str = dspy.InputField(description="The context of the text")
            sentiment: int = dspy.OutputField(description="The sentiment of the text")

        print(
            dspy.ChatAdapter().format_user_message_content(
                MySignature, {"text": "Hello, world!", "context": "This is a test."}
            )
        )
        ```

        The above code will output:
        ```
        [[ ## text ## ]]
        Hello, world!

        [[ ## context ## ]]
        This is a test.

        Respond with the corresponding output fields, starting with the field `[[ ## sentiment ## ]]` (must be
        formatted as a valid Python int), and then ending with the marker for `[[ ## completed ## ]]`.
        ```
        """

        def type_info(v):
            if v.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            else:
                return ""

        message = "Respond with the corresponding output fields, starting with the field "
        message += ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message

    def format_assistant_message_content(
        self, signature: Type[Signature], outputs: dict[str, Any], missing_field_message=None
    ) -> str:
        return self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )

    def parse(self, signature: Type[Signature], completion: str) -> dict[str, Any]:
        sections = [(None, [])]

        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                # If the header pattern is found, split the rest of the line as content
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation)
                except Exception as e:
                    raise ValueError(
                        f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{v}\n```"
                    )

        if fields.keys() != signature.output_fields.keys():
            raise ValueError(f"Expected {signature.output_fields.keys()} but got {fields.keys()}")

        return fields

    def format_field_with_value(self, fields_with_values: Dict[FieldInfoWithName, Any]) -> str:
        """
        Formats the values of the specified fields according to the field's DSPy type (input or output),
        annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
        into a single string, which is is a multiline string if there are multiple fields.

        Args:
            fields_with_values: A dictionary mapping information about a field to its corresponding
                value.
        Returns:
            The joined formatted values of the fields, represented as a string
        """
        output = []
        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

        return "\n\n".join(output).strip()

    # TODO(PR): Looks ok?
    def format_finetune_data(
        self, signature: Type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        # Get system + user messages
        messages = self.format(signature, demos, inputs)

        # Add the assistant message
        role = "assistant"
        incomplete = False
        assistant_message = self.format_turn(signature, outputs, role, incomplete)
        messages.append(assistant_message)

        # Wrap the messages in a dictionary with a "messages" key
        return dict(messages=messages)
