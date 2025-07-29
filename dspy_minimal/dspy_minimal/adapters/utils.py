"""Utility functions for adapters."""

import json
import re
from typing import Any, Dict, List, Union


def format_field_value(field_info: Any, value: Any) -> str:
    """Format a field value for display in prompts."""
    if value is None:
        return "Not provided"
    elif isinstance(value, list):
        return format_list_value(value)
    elif isinstance(value, dict):
        return format_dict_value(value)
    else:
        return str(value)


def format_list_value(value: List[Any]) -> str:
    """Format a list value for display."""
    if not value:
        return "[]"
    
    formatted_items = []
    for item in value:
        if isinstance(item, dict):
            formatted_items.append(format_dict_value(item))
        elif isinstance(item, list):
            formatted_items.append(format_list_value(item))
        else:
            formatted_items.append(str(item))
    
    return f"[{', '.join(formatted_items)}]"


def format_dict_value(value: Dict[str, Any]) -> str:
    """Format a dict value for display."""
    if not value:
        return "{}"
    
    formatted_items = []
    for key, val in value.items():
        if isinstance(val, dict):
            formatted_items.append(f"{key}: {format_dict_value(val)}")
        elif isinstance(val, list):
            formatted_items.append(f"{key}: {format_list_value(val)}")
        else:
            formatted_items.append(f"{key}: {str(val)}")
    
    return f"{{{', '.join(formatted_items)}}}"


def translate_field_type(field_name: str, field_info: Any) -> str:
    """Convert Python type annotations to natural language."""
    annotation = getattr(field_info, 'annotation', str)
    
    if annotation == str:
        return "text"
    elif annotation == int:
        return "integer"
    elif annotation == float:
        return "decimal number"
    elif annotation == bool:
        return "true/false"
    elif hasattr(annotation, '__origin__') and annotation.__origin__ == list:
        if hasattr(annotation, '__args__') and annotation.__args__:
            return f"list of {translate_field_type('', annotation.__args__[0])}"
        else:
            return "list"
    elif hasattr(annotation, '__origin__') and annotation.__origin__ == dict:
        return "dictionary"
    else:
        return "text"


def get_annotation_name(annotation: Any) -> str:
    """Get the name of a type annotation."""
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    elif hasattr(annotation, '__origin__'):
        origin_name = getattr(annotation.__origin__, '__name__', str(annotation.__origin__))
        if hasattr(annotation, '__args__') and annotation.__args__:
            args_names = [get_annotation_name(arg) for arg in annotation.__args__]
            return f"{origin_name}[{', '.join(args_names)}]"
        else:
            return origin_name
    else:
        return str(annotation)


def parse_value(value: str, field_info: Any) -> Any:
    """Parse a string value according to the field type."""
    annotation = getattr(field_info, 'annotation', str)
    
    if annotation == str:
        return value
    elif annotation == int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Could not parse '{value}' as integer")
    elif annotation == float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Could not parse '{value}' as float")
    elif annotation == bool:
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        else:
            raise ValueError(f"Could not parse '{value}' as boolean")
    elif hasattr(annotation, '__origin__') and annotation.__origin__ == list:
        # Try to parse as JSON list
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback: split by comma and strip whitespace
        return [item.strip() for item in value.split(',') if item.strip()]
    elif hasattr(annotation, '__origin__') and annotation.__origin__ == dict:
        # Try to parse as JSON dict
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback: try to parse key-value pairs
        result = {}
        pairs = value.split(',')
        for pair in pairs:
            if ':' in pair:
                key, val = pair.split(':', 1)
                result[key.strip()] = val.strip()
        return result
    else:
        return value


def serialize_for_json(value: Any) -> Any:
    """Serialize a value for JSON output."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, list):
        return [serialize_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): serialize_for_json(v) for k, v in value.items()}
    else:
        return str(value)


def get_field_description_string(fields: Dict[str, Any]) -> str:
    """Get a formatted string of field descriptions."""
    descriptions = []
    for name, field in fields.items():
        desc = getattr(field, 'description', name)
        field_type = translate_field_type(name, field)
        descriptions.append(f"- {name} ({field_type}): {desc}")
    return "\n".join(descriptions)


def parse_structured_response(text: str, signature: Any) -> Dict[str, Any]:
    """Parse structured responses with better error handling."""
    # Try JSON parsing first
    try:
        data = json.loads(text)
        return validate_parsed_data(data, signature)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return validate_parsed_data(data, signature)
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON from the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return validate_parsed_data(data, signature)
            except json.JSONDecodeError:
                pass
    
    # Fallback to regex-based parsing
    return parse_with_regex(text, signature)


def validate_parsed_data(data: Dict[str, Any], signature: Any) -> Dict[str, Any]:
    """Validate that parsed data has all required fields."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary, got {type(data)}")
    
    # Check for required fields
    required_fields = []
    for field_name, field_info in signature.output_fields.items():
        if not getattr(field_info, 'optional', False):
            required_fields.append(field_name)
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}. Available fields: {list(data.keys())}")
    
    return data


def parse_with_regex(text: str, signature: Any) -> Dict[str, Any]:
    """Parse response using regex patterns."""
    result = {}
    
    for field_name, field_info in signature.output_fields.items():
        # Try to find the field value using various patterns
        patterns = [
            rf'{field_name}["\s]*:["\s]*["\']?([^"\']+)["\']?',
            rf'{field_name}["\s]*:["\s]*(\d+)',
            rf'{field_name}["\s]*(true|false)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    result[field_name] = parse_value(value, field_info)
                    break
                except ValueError:
                    continue
    
    return result


def format_parse_error(error: Exception, text: str, signature: Any) -> str:
    """Format helpful error messages for parsing failures."""
    if isinstance(error, json.JSONDecodeError):
        return f"Failed to parse JSON response: {error.msg}. Response: {text[:200]}..."
    elif isinstance(error, KeyError):
        return f"Missing required field '{error}'. Available fields: {list(text.keys()) if isinstance(text, dict) else 'unknown'}"
    elif isinstance(error, ValueError):
        return f"Validation error: {error}. Response: {text[:200]}..."
    else:
        return f"Unexpected parsing error: {error}. Response: {text[:200]}..."


def split_message_content_for_custom_types(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split message content to handle custom types properly."""
    # This is a simplified version - the full implementation handles
    # complex types like images, audio, etc.
    return messages


def validate_demo(demo: Dict[str, Any], signature: Any) -> Dict[str, Any]:
    """Validate that a demo has the required structure."""
    has_input = any(field in demo for field in signature.input_fields)
    has_output = any(field in demo for field in signature.output_fields)
    is_complete = all(field in demo and demo[field] is not None 
                     for field in signature.fields)
    
    return {
        'is_complete': is_complete,
        'has_input': has_input,
        'has_output': has_output,
        'missing_fields': [field for field in signature.fields 
                          if field not in demo or demo[field] is None]
    } 