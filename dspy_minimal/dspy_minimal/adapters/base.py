"""Base adapter for DSPy minimal implementation."""

import logging
from typing import TYPE_CHECKING, Any, get_origin

from .types import History, Tool, ToolCalls
from ..utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..clients.lm import LM
    from ..signatures.signature import Signature


class Adapter:
    """Base adapter class for transforming inputs and outputs."""
    
    def __init__(self, callbacks=None, use_native_function_calling: bool = False):
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling

    def _call_preprocess(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type["Signature"],
        inputs: dict[str, Any],
    ) -> type["Signature"]:
        # Simplified version without tool calling support
        return signature

    def _call_postprocess(
        self,
        processed_signature: type["Signature"],
        original_signature: type["Signature"],
        outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        values = []

        for output in outputs:
            text = output
            if isinstance(output, dict):
                text = output.get("text", output)

            if text:
                value = self.parse(processed_signature, text)
                for field_name in original_signature.output_fields.keys():
                    if field_name not in value:
                        # Set missing fields to None for consistency
                        value[field_name] = None
            else:
                value = {}
                for field_name in original_signature.output_fields.keys():
                    value[field_name] = None

            values.append(value)

        return values

    def __call__(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type["Signature"],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        inputs = self.format(processed_signature, demos, inputs)

        outputs = lm(messages=inputs, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs)

    async def acall(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type["Signature"],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        inputs = self.format(processed_signature, demos, inputs)

        outputs = await lm.acall(messages=inputs, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs)

    def format(
        self,
        signature: type["Signature"],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the input messages for the LM call.

        This method converts the DSPy structured input along with few-shot examples and conversation history into
        multiturn messages as expected by the LM.

        Args:
            signature: The DSPy signature for which to format the input messages.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        inputs_copy = dict(inputs)

        # Handle conversation history if present
        history_field_name = self._get_history_field_name(signature)
        if history_field_name:
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                inputs_copy,
            )

        messages = []
        
        # Create system message
        system_message = (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )
        messages.append({"role": "system", "content": system_message})
        
        # Add demonstration examples
        messages.extend(self.format_demos(signature, demos))
        
        # Add conversation history
        if history_field_name:
            messages.extend(conversation_history)
        
        # Add the current input
        user_content = self.format_user_message_content(
            signature_without_history if history_field_name else signature, 
            inputs_copy, 
            main_request=True
        )
        messages.append({"role": "user", "content": user_content})

        return messages

    def format_field_description(self, signature: type["Signature"]) -> str:
        """Format the field description for the system message."""
        raise NotImplementedError

    def format_field_structure(self, signature: type["Signature"]) -> str:
        """Format the field structure for the system message."""
        raise NotImplementedError

    def format_task_description(self, signature: type["Signature"]) -> str:
        """Format the task description for the system message."""
        raise NotImplementedError

    def format_user_message_content(
        self,
        signature: type["Signature"],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message content."""
        raise NotImplementedError

    def format_assistant_message_content(
        self,
        signature: type["Signature"],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format the assistant message content."""
        raise NotImplementedError

    def format_demos(self, signature: type["Signature"], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format the few-shot examples."""
        messages = []

        for demo in demos:
            # Check if demo has at least one input and one output field
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if has_input and has_output:
                messages.append({
                    "role": "user", 
                    "content": self.format_user_message_content(signature, demo)
                })
                messages.append({
                    "role": "assistant", 
                    "content": self.format_assistant_message_content(signature, demo)
                })

        return messages

    def _get_history_field_name(self, signature: type["Signature"]) -> str | None:
        """Get the name of the history field if present."""
        for name, field in signature.input_fields.items():
            if hasattr(field, 'annotation') and field.annotation == History:
                return name
        return None

    def _get_tool_call_input_field_name(self, signature: type["Signature"]) -> str | None:
        """Get the name of the tool call input field if present."""
        for name, field in signature.input_fields.items():
            # Look for annotation `list[Tool]` or `Tool`
            origin = get_origin(field.annotation) if hasattr(field, 'annotation') else None
            if origin is list and hasattr(field.annotation, '__args__') and field.annotation.__args__[0] == Tool:
                return name
            if hasattr(field, 'annotation') and field.annotation == Tool:
                return name
        return None

    def _get_tool_call_output_field_name(self, signature: type["Signature"]) -> str | None:
        """Get the name of the tool call output field if present."""
        for name, field in signature.output_fields.items():
            if hasattr(field, 'annotation') and field.annotation == ToolCalls:
                return name
        return None

    def format_conversation_history(
        self,
        signature: type["Signature"],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the conversation history."""
        history_obj = inputs.get(history_field_name)
        if hasattr(history_obj, 'messages'):
            conversation_history = history_obj.messages
        else:
            conversation_history = []

        if not conversation_history:
            return []

        messages = []
        for message in conversation_history:
            messages.append({
                "role": "user",
                "content": self.format_user_message_content(signature, message),
            })
            messages.append({
                "role": "assistant",
                "content": self.format_assistant_message_content(signature, message),
            })

        # Remove the history field from the inputs
        del inputs[history_field_name]

        return messages

    def parse(self, signature: type["Signature"], completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields."""
        raise NotImplementedError


# Legacy adapters for backward compatibility
class SimpleAdapter(Adapter):
    """Simple adapter that passes data through unchanged."""
    
    def format_field_description(self, signature: type["Signature"]) -> str:
        return "Process the input and provide the output."
    
    def format_field_structure(self, signature: type["Signature"]) -> str:
        return "Provide your response in a clear format."
    
    def format_task_description(self, signature: type["Signature"]) -> str:
        return getattr(signature, 'instructions', 'Complete the task.')
    
    def format_user_message_content(self, signature: type["Signature"], inputs: dict[str, Any], **kwargs) -> str:
        return str(inputs)
    
    def format_assistant_message_content(self, signature: type["Signature"], outputs: dict[str, Any], **kwargs) -> str:
        return str(outputs)
    
    def parse(self, signature: type["Signature"], completion: str) -> dict[str, Any]:
        return {"response": completion}


class JSONAdapter(Adapter):
    """Adapter for JSON input/output handling."""
    
    def format_field_description(self, signature: type["Signature"]) -> str:
        input_desc = self._get_field_description_string(signature.input_fields)
        output_desc = self._get_field_description_string(signature.output_fields)
        return f"Your input fields are:\n{input_desc}\nYour output fields are:\n{output_desc}"
    
    def format_field_structure(self, signature: type["Signature"]) -> str:
        return "Provide your response as a valid JSON object with the required fields."
    
    def format_task_description(self, signature: type["Signature"]) -> str:
        return getattr(signature, 'instructions', 'Complete the task and return JSON.')
    
    def format_user_message_content(self, signature: type["Signature"], inputs: dict[str, Any], **kwargs) -> str:
        return str(inputs)
    
    def format_assistant_message_content(self, signature: type["Signature"], outputs: dict[str, Any], **kwargs) -> str:
        import json
        return json.dumps(outputs, indent=2)
    
    def parse(self, signature: type["Signature"], completion: str) -> str:
        import json
        try:
            return json.loads(completion)
        except json.JSONDecodeError as e:
            raise AdapterParseError(f"Failed to parse JSON output: {e}")
    
    def _get_field_description_string(self, fields: dict[str, Any]) -> str:
        descriptions = []
        for name, field in fields.items():
            desc = getattr(field, 'description', name)
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)


class JSONPropertiesAdapter(JSONAdapter):
    """Custom adapter that handles LLM responses wrapped in a 'properties' key"""

    def parse(self, signature: type["Signature"], completion: str) -> dict[str, Any]:
        """Parse the response, handling both direct JSON and properties-wrapped JSON"""
        import json
        try:
            # First try to parse as regular JSON
            data = json.loads(completion)

            # Check if the response has a 'properties' wrapper
            if isinstance(data, dict) and 'properties' in data:
                # Extract the actual data from properties
                actual_data = data['properties']
            else:
                # Use the data as-is
                actual_data = data

            # Handle special cases for specific fields
            if 'where_clause' in actual_data:
                # Convert empty list to None for where_clause
                if actual_data['where_clause'] == []:
                    actual_data['where_clause'] = None

            # Validate that all required fields are present
            required_fields = [field_name for field_name in signature.output_fields.keys() 
                             if not getattr(signature.output_fields[field_name], 'optional', False)]
            missing_fields = [field for field in required_fields if field not in actual_data]

            if missing_fields:
                raise AdapterParseError(
                    f"Missing required fields: {missing_fields}. "
                    f"Available fields: {list(actual_data.keys())}"
                )

            return actual_data

        except json.JSONDecodeError as e:
            raise AdapterParseError(f"Failed to parse JSON: {e}. Response: {completion}")
        except Exception as e:
            raise AdapterParseError(f"Unexpected error parsing response: {e}. Response: {completion}") 