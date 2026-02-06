import pydantic

# The following arguments can be used in DSPy InputField and OutputField in addition
# to the standard pydantic.Field arguments. We just hope pydanitc doesn't add these,
# as it would give a name clash.
DSPY_FIELD_ARG_NAMES = ["desc", "prefix", "format", "parser", "__dspy_field_type"]

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


def InputField(**kwargs):  # noqa: N802
    """Create a DSPy input field with optional metadata.

    Wraps ``pydantic.Field`` while tagging the field as a DSPy input. Supports
    DSPy-specific arguments like ``desc``, ``prefix``, ``format``, and ``parser``
    in addition to standard Pydantic field arguments.

    Args:
        **kwargs: Keyword arguments including:
            - desc (str): Human-readable description of the field.
            - prefix (str): Prompt prefix used when rendering the field.
            - format (Callable): Function to format the field value.
            - parser (Callable): Function to parse model output for this field.
            - Any standard ``pydantic.Field`` arguments (e.g., ``default``,
              ``description``, ``gt``, ``min_length``).

    Returns:
        pydantic.fields.FieldInfo: A Pydantic FieldInfo object marked as a DSPy
            input field.

    Example:
        Define input fields in a DSPy signature:

        ```python
        import dspy

        class QASignature(dspy.Signature):
            question: str = dspy.InputField(desc="The question to answer")
            context: str = dspy.InputField(desc="Supporting context", default="")
        ```
    """
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs):  # noqa: N802
    """Create a DSPy output field with optional metadata.

    Wraps ``pydantic.Field`` while tagging the field as a DSPy output. Supports
    DSPy-specific arguments like ``desc``, ``prefix``, ``format``, and ``parser``
    in addition to standard Pydantic field arguments.

    Args:
        **kwargs: Keyword arguments including:
            - desc (str): Human-readable description of the field.
            - prefix (str): Prompt prefix used when rendering the field.
            - format (Callable): Function to format the field value.
            - parser (Callable): Function to parse model output for this field.
            - Any standard ``pydantic.Field`` arguments (e.g., ``default``,
              ``description``).

    Returns:
        pydantic.fields.FieldInfo: A Pydantic FieldInfo object marked as a DSPy
            output field.

    Example:
        Define output fields in a DSPy signature:

        ```python
        import dspy

        class QASignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(desc="A concise answer to the question")
            confidence: float = dspy.OutputField(desc="Confidence score from 0 to 1")
        ```
    """
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))


def new_to_old_field(field):
    return (OldInputField if field.json_schema_extra["__dspy_field_type"] == "input" else OldOutputField)(
        prefix=field.json_schema_extra["prefix"],
        desc=field.json_schema_extra["desc"],
        format=field.json_schema_extra.get("format"),
    )


class OldField:
    """A more ergonomic datatype that infers prefix and desc if omitted."""

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
