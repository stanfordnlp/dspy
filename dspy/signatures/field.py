import warnings

import pydantic

from dspy.utils.constants import IS_TYPE_UNDEFINED

# The following arguments can be used in DSPy InputField and OutputField in addition
# to the standard pydantic.Field arguments. We just hope pydanitc doesn't add these,
# as it would give a name clash.
DSPY_FIELD_ARG_NAMES = ["desc", "prefix", "format", "parser", "__dspy_field_type", IS_TYPE_UNDEFINED]

_DEPRECATED_FIELD_ARGS = {
    "prefix": (
        "The 'prefix' argument in InputField/OutputField is deprecated and has no effect in DSPy. "
        "It will be removed in a future version."
    ),
    "format": (
        "The 'format' argument in InputField/OutputField is deprecated and has no effect in DSPy. "
        "It will be removed in a future version."
    ),
    "parser": (
        "The 'parser' argument in InputField/OutputField is deprecated and has no effect in DSPy. "
        "It will be removed in a future version."
    ),
}

PYDANTIC_CONSTRAINT_MAP = {
    "gt": "greater than: ",
    "ge": "greater than or equal to: ",
    "lt": "less than: ",
    "le": "less than or equal to: ",
    "min_length": "minimum length: ",
    "max_length": "maximum length: ",
    "multiple_of": "a multiple of the given number: ",
    "allow_inf_nan": "allow 'inf', '-inf', 'nan' values: ",
}


def move_kwargs(**kwargs):
    # Pydantic doesn't allow arbitrary arguments to be given to fields,
    # but asks that
    # > any extra data you want to add to the JSON schema should be passed
    # > as a dictionary to the json_schema_extra keyword argument.
    # See: https://docs.pydantic.dev/2.6/migration/#changes-to-pydanticfield
    pydantic_kwargs = {}
    json_schema_extra = {}
    for k, v in kwargs.items():
        if k in DSPY_FIELD_ARG_NAMES:
            json_schema_extra[k] = v
        else:
            pydantic_kwargs[k] = v
    # Also copy over the pydantic "description" if no dspy "desc" is given.
    if "description" in kwargs and "desc" not in json_schema_extra:
        json_schema_extra["desc"] = kwargs["description"]
    constraints = _translate_pydantic_field_constraints(**kwargs)
    if constraints:
        json_schema_extra["constraints"] = constraints
    pydantic_kwargs["json_schema_extra"] = json_schema_extra
    return pydantic_kwargs


def _translate_pydantic_field_constraints(**kwargs):
    """Extracts Pydantic constraints and translates them into human-readable format."""

    constraints = []
    for key, value in kwargs.items():
        if key in PYDANTIC_CONSTRAINT_MAP:
            constraints.append(f"{PYDANTIC_CONSTRAINT_MAP[key]}{value}")

    return ", ".join(constraints)


def _warn_deprecated_field_args(**kwargs):
    for arg, message in _DEPRECATED_FIELD_ARGS.items():
        if arg in kwargs:
            warnings.warn(message, DeprecationWarning, stacklevel=3)


def InputField(**kwargs):  # noqa: N802
    """Declare an input field on a `dspy.Signature`.

    Use `InputField` for fields that should be passed into the module.
    If you omit the type annotation, DSPy treats the field as `str`.

    Args:
        **kwargs: `pydantic.Field` keyword arguments. Use `desc` for a short
            field description shown to the language model.

    Returns:
        (pydantic.fields.FieldInfo): A field marked as a DSPy input.

    Examples:
        >>> import dspy
        >>> class QA(dspy.Signature):
        ...     question: str = dspy.InputField(desc="A factual question")
        ...     answer: str = dspy.OutputField()
    """
    _warn_deprecated_field_args(**kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs):  # noqa: N802
    """Declare an output field on a `dspy.Signature`.

    Use `OutputField` for fields the module should produce. Constraints like
    `ge`, `le`, `min_length`, and `max_length` are described to the language
    model and validated on the parsed output. If you omit the type annotation,
    DSPy treats the field as `str`.

    Args:
        **kwargs: `pydantic.Field` keyword arguments. Use `desc` for a short
            field description shown to the language model.

    Returns:
        (pydantic.fields.FieldInfo): A field marked as a DSPy output.

    Examples:
        >>> import dspy
        >>> class Review(dspy.Signature):
        ...     text: str = dspy.InputField()
        ...     rating: int = dspy.OutputField(ge=0, le=5)
    """
    _warn_deprecated_field_args(**kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))


def new_to_old_field(field):
    return (OldInputField if field.json_schema_extra["__dspy_field_type"] == "input" else OldOutputField)(
        prefix=field.json_schema_extra["prefix"],
        desc=field.json_schema_extra["desc"],
        format=field.json_schema_extra.get("format"),
    )


class OldField:
    """Legacy field type. Use `InputField` or `OutputField` instead."""

    def __init__(self, *, prefix=None, desc=None, input, format=None):
        self.prefix = prefix  # This can be None initially and set later
        self.desc = desc
        self.format = format

    def finalize(self, key, inferred_prefix):
        """Set the prefix if it's not provided explicitly."""
        if self.prefix is None:
            self.prefix = inferred_prefix + ":"

        if self.desc is None:
            self.desc = f"${{{key}}}"

    def __repr__(self):
        return f"{self.__class__.__name__}(prefix={self.prefix}, desc={self.desc})"

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__


class OldInputField(OldField):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=True, format=format)


class OldOutputField(OldField):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=False, format=format)
