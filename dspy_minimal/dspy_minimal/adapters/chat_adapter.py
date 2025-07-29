"""Chat adapter for DSPy minimal implementation."""

import re
import textwrap
from typing import Any, Dict, List, NamedTuple

from .base import Adapter
from .utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    translate_field_type,
    validate_demo,
)
from ..utils.exceptions import AdapterParseError


field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    name: str
    info: Any  # Simplified for dspy_minimal


class ChatAdapter(Adapter):
    """Chat adapter for DSPy minimal implementation."""
    
    def __call__(
        self,
        lm,
        lm_kwargs: Dict[str, Any],
        signature,
        demos: List[Dict[str, Any]],
        inputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Format the prompt and call the language model."""
        try:
            return self._call_impl(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # Fallback to JSONAdapter
            from .json_adapter import JSONAdapter
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)
    
    async def acall(
        self,
        lm,
        lm_kwargs: Dict[str, Any],
        signature,
        demos: List[Dict[str, Any]],
        inputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Async version of the call method."""
        try:
            return await self._acall_impl(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # Fallback to JSONAdapter
            from .json_adapter import JSONAdapter
            return await JSONAdapter().acall(lm, lm_kwargs, signature, demos, inputs)
    
    def _call_impl(self, lm, lm_kwargs: Dict[str, Any], signature, demos: List[Dict[str, Any]], inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implementation of the call method."""
        messages = self.format(signature, demos, inputs)
        response = lm.forward(messages=messages, **lm_kwargs)
        parsed = self.parse(signature, self._extract_content(response))
        return [parsed]
    
    async def _acall_impl(self, lm, lm_kwargs: Dict[str, Any], signature, demos: List[Dict[str, Any]], inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Implementation of the async call method."""
        messages = self.format(signature, demos, inputs)
        response = await lm.aforward(messages=messages, **lm_kwargs)
        parsed = self.parse(signature, self._extract_content(response))
        return [parsed]
    
    def _extract_content(self, response) -> str:
        """Extract content from various response formats."""
        if isinstance(response, dict) and 'choices' in response:
            return response['choices'][0]['message']['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def format_field_description(self, signature) -> str:
        """Format field descriptions."""
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )
    
    def format_field_structure(self, signature) -> str:
        """Format field structure with markers."""
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, Any]):
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

    def format_task_description(self, signature) -> str:
        """Format task description."""
        instructions = textwrap.dedent(getattr(signature, 'instructions', ''))
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def format_user_message_content(
        self,
        signature,
        inputs: Dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format user message content."""
        parts = []
        
        if prefix:
            parts.append(prefix)
        
        # Format input fields
        for field_name, field_info in signature.input_fields.items():
            if field_name in inputs and inputs[field_name] is not None:
                value = inputs[field_name]
                formatted_value = format_field_value(field_info, value)
                parts.append(f"[[ ## {field_name} ## ]]")
                parts.append(formatted_value)
        
        if main_request:
            parts.append(self.user_message_output_requirements(signature))
        
        if suffix:
            parts.append(suffix)
        
        return "\n".join(parts)

    def user_message_output_requirements(self, signature) -> str:
        """Add output requirements to user message."""
        parts = []
        parts.append("\nPlease provide your response in the following format:")
        
        def type_info(v):
            if hasattr(v, 'annotation'):
                return translate_field_type('', v)
            return 'text'
        
        for field_name, field_info in signature.output_fields.items():
            field_type = type_info(field_info)
            desc = getattr(field_info, 'description', field_name)
            parts.append(f"[[ ## {field_name} ## ]]")
            parts.append(f"({field_type}): {desc}")
        
        parts.append("[[ ## completed ## ]]")
        return "\n".join(parts)

    def format_assistant_message_content(
        self,
        signature,
        outputs: Dict[str, Any],
        missing_field_message=None,
    ) -> str:
        """Format assistant message content."""
        parts = []
        for field_name, field_info in signature.output_fields.items():
            parts.append(f"[[ ## {field_name} ## ]]")
            if field_name in outputs and outputs[field_name] is not None:
                parts.append(format_field_value(field_info, outputs[field_name]))
            else:
                parts.append(missing_field_message)
        
        parts.append("[[ ## completed ## ]]")
        return "\n".join(parts)

    def parse(self, signature, completion: str) -> Dict[str, Any]:
        """Parse the completion into structured output."""
        result = {}
        
        # Find all field markers and their positions
        matches = list(field_header_pattern.finditer(completion))
        
        for i, match in enumerate(matches):
            field_name = match.group(1)
            
            # Find the content between this field marker and the next one
            start_pos = match.end()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(completion)
            
            # Extract and clean the content
            content = completion[start_pos:end_pos].strip()
            
            # Skip empty content or placeholder messages
            if content and content.lower() not in [
                'not provided.', 
                'not supplied for this particular example.', 
                'not supplied for this conversation history message.',
                '(text):',
                '(str):',
                '(int):',
                '(float):',
                '(bool):'
            ]:
                # Clean up the content by removing type annotations and extra formatting
                content = self._clean_content(content)
                
                # Handle None values properly
                if content and content.lower() not in ['none', 'null', '']:
                    try:
                        field_info = signature.output_fields.get(field_name)
                        if field_info:
                            # For date fields, handle None values specially
                            if hasattr(field_info, 'annotation') and field_info.annotation == str:
                                # Check if this looks like a date field
                                if 'date' in field_name.lower():
                                    if content.lower() in ['none', 'null', '(text): none', '(text): null']:
                                        result[field_name] = None
                                    else:
                                        result[field_name] = content
                                else:
                                    result[field_name] = content
                            else:
                                result[field_name] = parse_value(content, field_info)
                        else:
                            result[field_name] = content
                    except ValueError:
                        result[field_name] = content
                elif content.lower() in ['none', 'null', '(text): none', '(text): null']:
                    # Explicitly handle None values
                    result[field_name] = None
        
        return result

    def _clean_content(self, content: str) -> str:
        """Clean content by removing type annotations and extra formatting."""
        # Remove type annotations like "(text): None" -> "None"
        content = re.sub(r'^\([^)]+\):\s*', '', content.strip())
        
        # Remove extra whitespace and newlines
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content

    def format_field_with_value(self, fields_with_values: Dict[FieldInfoWithName, Any]) -> str:
        """Format fields with their values."""
        parts = []
        for field_info_with_name, value in fields_with_values.items():
            parts.append(f"[[ ## {field_info_with_name.name} ## ]]")
            parts.append(f"({value})")
        return "\n".join(parts)

    def format_demos(self, signature, demos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format demonstrations with validation."""
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            validation = validate_demo(demo, signature)
            
            if validation['is_complete']:
                complete_demos.append(demo)
            elif validation['has_input'] and validation['has_output']:
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
                        signature, demo, missing_field_message="Not supplied for this particular example. "
                    ),
                }
            )

        for demo in complete_demos:
            messages.append({"role": "user", "content": self.format_user_message_content(signature, demo)})
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this conversation history message. "
                    ),
                }
            )

        return messages 