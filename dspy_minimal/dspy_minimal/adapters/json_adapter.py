"""JSON adapter for DSPy minimal implementation."""

import json
import logging
import re
from typing import Any, Dict, List

from .base import Adapter
from .utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    serialize_for_json,
    translate_field_type,
    parse_structured_response,
    format_parse_error,
)
from ..utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


class JSONAdapter(Adapter):
    """JSON adapter for structured input/output handling."""
    
    def __init__(self, callbacks=None, use_native_function_calling: bool = True):
        super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)

    def format_field_description(self, signature) -> str:
        """Format field descriptions for JSON output."""
        input_desc = get_field_description_string(signature.input_fields)
        output_desc = get_field_description_string(signature.output_fields)
        return f"Your input fields are:\n{input_desc}\nYour output fields are:\n{output_desc}"

    def format_field_structure(self, signature) -> str:
        """Format field structure for JSON output."""
        parts = []
        parts.append("Provide your response as a valid JSON object with the following structure:")

        def format_signature_fields_for_instructions(fields: dict[str, Any], role: str):
            field_descriptions = []
            for field_name, field_info in fields.items():
                field_type = translate_field_type(field_name, field_info)
                desc = getattr(field_info, 'description', field_name)
                field_descriptions.append(f'  "{field_name}": ({field_type}) {desc}')
            return f"{role} fields:\n" + "\n".join(field_descriptions)

        parts.append(format_signature_fields_for_instructions(signature.input_fields, "Input"))
        parts.append(format_signature_fields_for_instructions(signature.output_fields, "Output"))
        return "\n\n".join(parts)

    def format_task_description(self, signature) -> str:
        """Format task description."""
        instructions = getattr(signature, 'instructions', 'Complete the task and return JSON.')
        return f"Task: {instructions}\n\nProvide your response as a valid JSON object."

    def format_user_message_content(
        self,
        signature,
        inputs: Dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format user message content for JSON."""
        parts = []
        
        if prefix:
            parts.append(prefix)
        
        # Format input fields as JSON
        input_data = {}
        for field_name, field_info in signature.input_fields.items():
            if field_name in inputs and inputs[field_name] is not None:
                input_data[field_name] = serialize_for_json(inputs[field_name])
        
        if input_data:
            parts.append("Input:")
            parts.append(json.dumps(input_data, indent=2))
        
        if main_request:
            parts.append(self.user_message_output_requirements(signature))
        
        if suffix:
            parts.append(suffix)
        
        return "\n".join(parts)

    def user_message_output_requirements(self, signature) -> str:
        """Add output requirements to user message."""
        parts = []
        parts.append("\nPlease provide your response as a JSON object with the following fields:")
        
        def type_info(v):
            if hasattr(v, 'annotation'):
                return translate_field_type('', v)
            return 'text'
        
        for field_name, field_info in signature.output_fields.items():
            field_type = type_info(field_info)
            desc = getattr(field_info, 'description', field_name)
            parts.append(f'  "{field_name}": ({field_type}) {desc}')
        
        return "\n".join(parts)

    def format_assistant_message_content(
        self,
        signature,
        outputs: Dict[str, Any],
        missing_field_message=None,
    ) -> str:
        """Format assistant message content as JSON."""
        if missing_field_message is None:
            missing_field_message = "null"
        
        # Create output data with missing fields handled
        output_data = {}
        for field_name, field_info in signature.output_fields.items():
            if field_name in outputs and outputs[field_name] is not None:
                output_data[field_name] = serialize_for_json(outputs[field_name])
            else:
                output_data[field_name] = None
        
        return json.dumps(output_data, indent=2)

    def parse(self, signature, completion: str) -> Dict[str, Any]:
        """Parse JSON completion into structured output."""
        try:
            return parse_structured_response(completion, signature)
        except Exception as e:
            error_msg = format_parse_error(e, completion, signature)
            raise AdapterParseError(error_msg)

    def format_field_with_value(self, fields_with_values: Dict[Any, Any], role: str = "user") -> str:
        """Format fields with their values for JSON."""
        data = {}
        for field_info_with_name, value in fields_with_values.items():
            data[field_info_with_name.name] = serialize_for_json(value)
        
        return f"{role} data:\n{json.dumps(data, indent=2)}"

    def format_demos(self, signature, demos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format demonstrations for JSON output."""
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            # Check if all fields are present and not None
            is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

            # Check if demo has at least one input and one output field
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                incomplete_demos.append(demo)

        messages = []

        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="null"
                    ),
                }
            )

        for demo in complete_demos:
            messages.append({"role": "user", "content": self.format_user_message_content(signature, demo)})
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(signature, demo),
                }
            )

        return messages