from copy import deepcopy
import typing
import dsp
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from typing import Type, Union, Dict, Tuple
import re

from dspy.signatures.field import InputField, OutputField, new_to_old_field


def signature_to_template(signature):
    """Convert from new to legacy format"""
    return dsp.Template(
        signature.instructions,
        **{name: new_to_old_field(field) for name, field in signature.fields.items()},
    )


def _default_instructions(cls):
    inputs_ = ", ".join([f"`{field}`" for field in cls.input_fields.keys()])
    outputs_ = ", ".join([f"`{field}`" for field in cls.output_fields.keys()])
    return f"Given the fields {inputs_}, produce the fields {outputs_}."


class SignatureMeta(type(BaseModel)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Set `str` as the default type for all fields
        raw_annotations = namespace.get("__annotations__", {})
        for name, field in namespace.items():
            if not name.startswith("__") and name not in raw_annotations:
                raw_annotations[name] = str
        namespace["__annotations__"] = raw_annotations

        # Let Pydantic do its thing
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

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
                    f"Field '{name}' in '{cls.__name__}' must be declared with InputField or OutputField."
                )

    @property
    def signature(cls) -> str:
        in_args = ", ".join(cls.input_fields.keys())
        out_args = ", ".join(cls.output_fields.keys())
        return f"{in_args} -> {out_args}"

    @property
    def instructions(cls) -> str:
        return getattr(cls, "__doc__", "")

    def with_instructions(cls, instructions: str):
        return Signature(cls.fields, instructions)

    @property
    def fields(cls):
        # Make sure to give input fields before output fields
        return {**cls.input_fields, **cls.output_fields}

    def with_updated_fields(cls, name, type_=None, **kwargs):
        """Returns a new Signature type with the field, name, updated
        with fields[name].json_schema_extra[key] = value."""
        fields_copy = deepcopy(cls.fields)
        fields_copy[name].json_schema_extra = {
            **fields_copy[name].json_schema_extra,
            **kwargs,
        }
        if type_ is not None:
            fields_copy[name].annotation = type_
        return Signature(fields_copy, cls.instructions)

    @property
    def input_fields(cls):
        return cls._get_fields_with_type("input")

    @property
    def output_fields(cls):
        return cls._get_fields_with_type("output")

    def _get_fields_with_type(cls, field_type):
        return {
            k: v
            for k, v in cls.model_fields.items()
            if v.json_schema_extra["__dspy_field_type"] == field_type
        }

    def prepend(cls, name, field, type_=None):
        return cls.insert(0, name, field, type_)

    def append(cls, name, field, type_=None):
        return cls.insert(-1, name, field, type_)

    def insert(cls, index: int, name: str, field, type_: Type = None):
        # It's posisble to set the type as annotation=type in pydantic.Field(...)
        # But this may be annoying for users, so we allow them to pass the type
        if type_ is None:
            type_ = field.annotation
        if type_ is None:
            type_ = str

        input_fields = list(cls.input_fields.items())
        output_fields = list(cls.output_fields.items())

        # Choose the list to insert into based on the field type
        lst = (
            input_fields
            if field.json_schema_extra["__dspy_field_type"] == "input"
            else output_fields
        )
        # We support negative insert indices
        if index < 0:
            index += len(lst) + 1
        if index < 0 or index > len(lst):
            raise ValueError(f"Invalid index: {index}")
        lst.insert(index, (name, (type_, field)))

        new_fields = dict(input_fields + output_fields)
        return Signature(new_fields, cls.instructions)

    def _parse_signature(cls, signature: str) -> Tuple[Type, Field]:
        pattern = r"^\s*[\w\s,]+\s*->\s*[\w\s,]+\s*$"
        if not re.match(pattern, signature):
            raise ValueError(f"Invalid signature format: '{signature}'")

        fields = {}
        inputs_str, outputs_str = map(str.strip, signature.split("->"))
        inputs = [v.strip() for v in inputs_str.split(",") if v.strip()]
        outputs = [v.strip() for v in outputs_str.split(",") if v.strip()]
        for name in inputs:
            fields[name] = (str, InputField())
        for name in outputs:
            fields[name] = (str, OutputField())

        return fields

    def __call__(
        cls,
        signature: Union[str, Dict[str, Tuple[type, FieldInfo]]],
        instructions: str = None,
    ):
        """
        Creates a new Signature type with the given fields and instructions.
        Note:
            Even though we're calling a type, we're not making an instance of the type.
            In general we don't allow instances of Signature types to be made. The call
            syntax is only for your convenience.
        Parameters:
            signature: Format: "input1, input2 -> output1, output2"
            instructions: Optional prompt for the signature.
        """

        if isinstance(signature, str):
            fields = cls._parse_signature(signature)
        else:
            fields = signature

        # Validate the fields, this is important because we sometimes forget the
        # slightly unintuitive syntax with tuples of (type, Field)
        fixed_fields = {}
        for name, type_field in fields.items():
            assert isinstance(
                name, str
            ), f"Field names must be strings, not {type(name)}"
            if isinstance(type_field, FieldInfo):
                type_ = type_field.annotation
                field = type_field
            else:
                assert isinstance(
                    type_field, tuple
                ), f"Field values must be tuples, not {type(type_field)}"
                type_, field = type_field
            # It might be better to be explicit about the type, but it currently would break
            # program of thought and teleprompters, so we just silently default to string.
            if type_ is None:
                type_ = str
            assert isinstance(type_, type) or isinstance(
                typing.get_origin(type_), type
            ), f"Field types must be types, not {type(type_)}"
            assert isinstance(
                field, FieldInfo
            ), f"Field values must be Field instances, not {type(field)}"
            fixed_fields[name] = (type_, field)

        # Fixing the fields shouldn't change the order
        assert list(fixed_fields.keys()) == list(fields.keys())

        # Default prompt when no instructions are provided
        if instructions is None:
            sig = Signature(signature, "")  # Simple way to parse input/output fields
            instructions = _default_instructions(sig)

        signature = create_model("Signature", __base__=Signature, **fixed_fields)
        signature.__doc__ = instructions
        return signature

    def equals(cls, other):
        """Compare the JSON schema of two Pydantic models."""
        if not isinstance(other, type) or not issubclass(other, BaseModel):
            return False
        if cls.instructions != other.instructions:
            return False
        for name in cls.fields.keys() | other.fields.keys():
            if name not in other.fields or name not in cls.fields:
                return False
            # TODO: Should we compare the fields?
        return True

    def __repr__(cls):
        """
        Outputs something on the form:
        Signature(question, context -> answer
            question: str = InputField(desc="..."),
            context: List[str] = InputField(desc="..."),
            answer: int = OutputField(desc="..."),
        )
        """
        field_reprs = []
        for name, field in cls.fields.items():
            field_reprs.append(f"{name} = Field({field})")
        field_repr = "\n    ".join(field_reprs)
        return (
            f"Signature({cls.signature}\n"
            f"    instructions={repr(cls.instructions)}\n"
            f"    {field_repr}\n)"
        )


class Signature(BaseModel, metaclass=SignatureMeta):
    pass


def ensure_signature(signature):
    if signature is None:
        return None
    if isinstance(signature, str):
        return Signature(signature)
    return signature


def infer_prefix(attribute_name: str) -> str:
    """Infers a prefix from an attribute name."""

    # Convert camelCase to snake_case, but handle sequences of capital letters properly
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", attribute_name)
    intermediate_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)

    # Insert underscores around numbers to ensure spaces in the final output
    with_underscores_around_numbers = re.sub(
        r"([a-zA-Z])(\d)", r"\1_\2", intermediate_name
    )
    with_underscores_around_numbers = re.sub(
        r"(\d)([a-zA-Z])", r"\1_\2", with_underscores_around_numbers
    )

    # Convert snake_case to 'Proper Title Case', but ensure acronyms are uppercased
    words = with_underscores_around_numbers.split("_")
    title_cased_words = []
    for word in words:
        if word.isupper():
            title_cased_words.append(word)
        else:
            title_cased_words.append(word.capitalize())

    return " ".join(title_cased_words)
