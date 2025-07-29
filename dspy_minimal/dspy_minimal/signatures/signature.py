import inspect
import re
import sys
import typing
from copy import deepcopy
from typing import Any, Union

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from .field import InputField, OutputField


def _default_instructions(cls) -> str:
    inputs_ = ", ".join([f"`{field}`" for field in cls.input_fields])
    outputs_ = ", ".join([f"`{field}`" for field in cls.output_fields])
    return f"Given the fields {inputs_}, produce the fields {outputs_}."


class SignatureMeta(type(BaseModel)):
    def __call__(cls, *args, **kwargs):
        if cls is Signature:
            # We don't create an actual Signature instance, instead, we create a new Signature class.
            custom_types = kwargs.pop("custom_types", None)

            if custom_types is None and args and isinstance(args[0], str):
                custom_types = cls._detect_custom_types_from_caller(args[0])

            return make_signature(*args, custom_types=custom_types, **kwargs)
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _detect_custom_types_from_caller(signature_str):
        """Detect custom types from the caller's frame based on the signature string."""
        # Simplified version - just return None for now
        return None

    def __new__(mcs, signature_name, bases, namespace, **kwargs):
        # At this point, the orders have been swapped already.
        field_order = [name for name, value in namespace.items() if isinstance(value, FieldInfo)]
        # Set `str` as the default type for all fields
        raw_annotations = namespace.get("__annotations__", {})
        for name, field in namespace.items():
            if not isinstance(field, FieldInfo):
                continue  # Don't add types to non-field attributes
            if not name.startswith("__") and name not in raw_annotations:
                raw_annotations[name] = str
        # Create ordered annotations dictionary that preserves field order
        ordered_annotations = {name: raw_annotations[name] for name in field_order if name in raw_annotations}
        # Add any remaining annotations that weren't in field_order
        ordered_annotations.update({k: v for k, v in raw_annotations.items() if k not in ordered_annotations})
        namespace["__annotations__"] = ordered_annotations

        # Let Pydantic do its thing
        cls = super().__new__(mcs, signature_name, bases, namespace, **kwargs)

        # If we don't have instructions, it might be because we are a derived generic type.
        # In that case, we should inherit the instructions from the base class.
        if cls.__doc__ is None:
            for base in bases:
                if isinstance(base, SignatureMeta):
                    doc = getattr(base, "__doc__", "")
                    if doc != "":
                        cls.__doc__ = doc

        # The more likely case is that the user has just not given us a type.
        # In that case, we should default to the input/output format.
        if cls.__doc__ is None:
            cls.__doc__ = _default_instructions(cls)

        # Ensure all fields are declared with InputField or OutputField
        cls._validate_fields()

        # Ensure all fields have a prefix
        for name, field in cls.model_fields.items():
            if "prefix" not in field.json_schema_extra:
                field.json_schema_extra["prefix"] = infer_prefix(name) + ":"
            if "desc" not in field.json_schema_extra:
                field.json_schema_extra["desc"] = f"${{{name}}}"

        return cls

    def _validate_fields(cls):
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra or {}
            field_type = extra.get("__dspy_field_type")
            if field_type not in ["input", "output"]:
                raise TypeError(
                    f"Field `{name}` in `{cls.__name__}` must be declared with InputField or OutputField, but "
                    f"field `{name}` has `field.json_schema_extra={field.json_schema_extra}`",
                )

    @property
    def instructions(cls) -> str:
        return inspect.cleandoc(getattr(cls, "__doc__", ""))

    @instructions.setter
    def instructions(cls, instructions: str) -> None:
        cls.__doc__ = instructions

    @property
    def input_fields(cls) -> dict[str, FieldInfo]:
        return cls._get_fields_with_type("input")

    @property
    def output_fields(cls) -> dict[str, FieldInfo]:
        return cls._get_fields_with_type("output")

    @property
    def fields(cls) -> dict[str, FieldInfo]:
        # Make sure to give input fields before output fields
        return {**cls.input_fields, **cls.output_fields}

    @property
    def signature(cls) -> str:
        inputs = ", ".join(cls.input_fields.keys())
        outputs = ", ".join(cls.output_fields.keys())
        return f"{inputs} -> {outputs}"

    def _get_fields_with_type(cls, field_type) -> dict[str, FieldInfo]:
        return {name: field for name, field in cls.model_fields.items() 
                if field.json_schema_extra and field.json_schema_extra.get("__dspy_field_type") == field_type}

    def __repr__(cls):
        return f"{cls.__name__}({cls.signature})"


class Signature(BaseModel, metaclass=SignatureMeta):
    """Base class for DSPy signatures."""
    
    @classmethod
    def dump_state(cls):
        """Dump the state of the signature."""
        state = {
            "instructions": cls.instructions,
            "fields": []
        }
        for field_name, field in cls.fields.items():
            state["fields"].append({
                "name": field_name,
                "type": "input" if field_name in cls.input_fields else "output",
                "description": field.json_schema_extra.get("desc", f"{field_name} field") if field.json_schema_extra else f"{field_name} field"
            })
        return state
    
    @classmethod
    def load_state(cls, state):
        """Load the state of the signature."""
        # For minimal version, we'll reconstruct from the signature string
        # This is a simplified implementation
        return cls


def ensure_signature(signature: Union[str, type[Signature]], instructions=None) -> type[Signature]:
    """Ensure that the signature is a Signature class."""
    if isinstance(signature, str):
        try:
            return make_signature(signature, instructions)
        except ValueError as e:
            raise ValueError(f"Failed to create signature from string '{signature}': {e}")
    elif isinstance(signature, type) and issubclass(signature, Signature):
        return signature
    else:
        raise ValueError(f"Invalid signature type: {type(signature)}, value: {signature}. Expected string or Signature class.")


def make_signature(
    signature: Union[str, dict[str, tuple[type, FieldInfo]]],
    instructions: Union[str, None] = None,
    signature_name: str = "StringSignature",
    custom_types: Union[dict[str, type], None] = None,
) -> type[Signature]:
    """Create a signature from a string or dictionary."""
    if isinstance(signature, str):
        # Parse the signature string
        parsed = _parse_signature(signature, custom_types)
        
        # Create the class
        namespace = {}
        annotations = {}
        
        for name, (type_, field) in parsed.items():
            namespace[name] = field
            annotations[name] = type_
            
        namespace["__annotations__"] = annotations
        
        # Create the class first
        signature_class = SignatureMeta(signature_name, (Signature,), namespace)
        
        # Set the instructions
        if instructions:
            signature_class.__doc__ = instructions
        else:
            # Use default instructions if none provided
            signature_class.__doc__ = _default_instructions(signature_class)
            
        return signature_class
    else:
        raise ValueError("Only string signatures are supported in minimal version")


def _parse_signature(signature: str, names=None) -> dict[str, tuple[type, Field]]:
    """Parse a signature string into a dictionary of field definitions."""
    # Simplified parsing - just split on -> and handle basic cases
    if "->" not in signature:
        raise ValueError("Signature must contain '->' to separate inputs from outputs")
        
    input_part, output_part = signature.split("->", 1)
    
    result = {}
    
    # Parse inputs
    for field_str in input_part.strip().split(","):
        field_str = field_str.strip()
        if field_str:
            name = field_str.split(":")[0].strip()
            result[name] = (str, InputField(desc=f"Input field: {name}"))
    
    # Parse outputs  
    for field_str in output_part.strip().split(","):
        field_str = field_str.strip()
        if field_str:
            name = field_str.split(":")[0].strip()
            result[name] = (str, OutputField(desc=f"Output field: {name}"))
            
    return result


def infer_prefix(attribute_name: str) -> str:
    """Infer a prefix for a field name."""
    # Simple prefix inference
    if attribute_name.startswith("input"):
        return "Input"
    elif attribute_name.startswith("output"):
        return "Output"
    else:
        return attribute_name.title() 