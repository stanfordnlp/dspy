"""InputField and OutputField factories for DSPy signatures.

Both are thin wrappers around `pydantic.Field`. They route
DSPy-specific kwargs (e.g. `desc`) into `json_schema_extra` and
forward everything else to Pydantic. Output-field constraints
(`ge`, `le`, `min_length`, …) are also translated into a
human-readable string that adapters include in prompts.

Legacy `OldField` / `OldInputField` / `OldOutputField` classes are
kept for backward compatibility but should not be used in new code.
"""

import warnings

import pydantic

# Only `desc` is used by adapters in current DSPy flows; we are deprecating
# `prefix`, `format`, and `parser`. So `desc` can be used in addition to
# the standard `pydantic.Field` arguments. We just hope Pydantic doesn't
# add these names, as that would create a name clash.
# TODO: In a future major release, remove `prefix`, `format`, and `parser`
# from `DSPY_FIELD_ARG_NAMES` and from all related code.
DSPY_FIELD_ARG_NAMES = ["desc", "prefix", "format", "parser", "__dspy_field_type"]

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
    """Translate Pydantic constraint kwargs into a human-readable string.

    Keys from `PYDANTIC_CONSTRAINT_MAP` are turned into a string in which
    the natural language version of the constraint are concatenated together
    and separated by commas. Adapters render that string into the prompt.

    Args:
        **kwargs: The full set of keyword arguments originally passed to
            `InputField` or `OutputField`.  Only keys that appear in
            `PYDANTIC_CONSTRAINT_MAP` are used; all others are ignored.

    Returns:
        (str): Comma-separated human-readable constraint descriptions, or an
        empty string if no recognized constraint kwargs are present.

    Examples:
        >>> _translate_pydantic_field_constraints(ge=5, le=10, desc="score")
        'greater than or equal to: 5, less than or equal to: 10'

        # Empty string if no constraint kwargs are present, even if other kwargs are given.
        >>> _translate_pydantic_field_constraints(desc="name")
        ''
    """
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

    Both the type hint and `desc` are optional. When provided,
    adapters pass them to the language model for additional context.

        >>> class QA(dspy.Signature):  # doctest: +SKIP
        ...     question: str = dspy.InputField(desc="a factual question")
        ...     answer: str = dspy.OutputField(desc="a short answer")

    Args:
        desc (str): Optional plain-language description of the field.
            Adapters include this when describing the field to the
            language model.
        **kwargs: Any additional `pydantic.Field` keyword argument.

    Returns:
        (pydantic.fields.FieldInfo): A field marked as a DSPy input.

    Examples:
        >>> import dspy
        >>> class Summarize(dspy.Signature):
        ...     text: str = dspy.InputField(desc="text to summarize")
        ...     summary: str = dspy.OutputField(desc="a one-sentence summary")
        >>> summarizer = dspy.Predict(Summarize)
        >>> summarizer.lm = dspy.LM("provider/model-name")  # doctest: +SKIP
        >>> summarizer(text="DSPy is a framework for...")  # doctest: +SKIP
        Prediction(...)

        You can also append an input field to an existing signature
        programmatically:

        >>> WithContext = Summarize.append(
        ...     "context", dspy.InputField(desc="background info"), type_=str
        ... )
        >>> list(WithContext.input_fields.keys())
        ['text', 'context']

        See [`dspy.LM`][dspy.LM] for supported providers.
    """
    _warn_deprecated_field_args(**kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs):  # noqa: N802
    """Declare an output field on a `dspy.Signature`.

    Both the type hint and `desc` are optional. When provided,
    adapters pass them to the language model for additional context.
    Using type hint is recommended for output fields so the language
    model knows what type to produce, defaults to str. Adapters also
    use the type hint to determine how to parse the model's outputs
    into Python objects.

        >>> class QA(dspy.Signature):  # doctest: +SKIP
        ...     question: str = dspy.InputField(desc="a factual question")
        ...     answer: str = dspy.OutputField(desc="a short answer")

    Args:
        desc (str): Optional plain-language description of the field.
            Adapters include this when describing the field to the
            language model.
        **kwargs: Any additional `pydantic.Field` keyword argument.
            Constraints like `gt`, `ge`, `lt`, `le`, `min_length`,
            `max_length`, and `multiple_of` are both described to the
            language model in the prompt and validated against its
            response. If the model's output violates a constraint,
            pydantic raises a `ValidationError`.

    Returns:
        (pydantic.fields.FieldInfo): A field marked as a DSPy output.

    Examples:
        >>> import dspy
        >>> class Review(dspy.Signature):
        ...     text: str = dspy.InputField(desc="product review text")
        ...     summary: str = dspy.OutputField(desc="a one-sentence summary")
        ...     rating: float = dspy.OutputField(desc="rating", ge=0, le=5)
        >>> reviewer = dspy.Predict(Review)
        >>> reviewer.lm = dspy.LM("provider/model-name")  # doctest: +SKIP
        >>> reviewer(text="This product is great!")  # doctest: +SKIP
        Prediction(...)

        You can also append an output field to an existing signature
        programmatically:

        >>> WithConfidence = Review.append(
        ...     "confidence", dspy.OutputField(desc="confidence score"), type_=float
        ... )
        >>> list(WithConfidence.output_fields.keys())
        ['summary', 'rating', 'confidence']

        See [`dspy.LM`][dspy.LM] for supported providers.
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
        if self.prefix is None:
            self.prefix = inferred_prefix + ":"

        if self.desc is None:
            self.desc = f"${{{key}}}"

    def __repr__(self):
        return f"{self.__class__.__name__}(prefix={self.prefix}, desc={self.desc})"

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__


class OldInputField(OldField):
    """Legacy input field. Use `InputField` instead."""

    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=True, format=format)


class OldOutputField(OldField):
    """Legacy output field. Use `OutputField` instead."""

    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=False, format=format)
