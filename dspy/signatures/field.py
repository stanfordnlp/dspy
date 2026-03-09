"""Defines DSPy InputField and OutputField.

The public `InputField` and `OutputField` factories are the primary
functions DSPy users call when declaring signature fields. Both
factories delegate to `pydantic.Field`, but they first split
caller-supplied keyword arguments into two groups:

- **DSPy-specific kwargs** (listed in `DSPY_FIELD_ARG_NAMES`) are stored
  under `json_schema_extra` so DSPy code can inspect them later.
- **Standard Pydantic kwargs** are forwarded directly to
  `pydantic.Field`.

In addition to extracting DSPy-specific kwargs, this module translates
Pydantic constraint kwargs (`ge`, `le`, `min_length`, etc.) into a
language-model-readable `"constraints"` string that adapters can render
into prompts.

This module also contains the legacy `OldField`, `OldInputField`, and
`OldOutputField` classes, along with a `new_to_old_field` bridge
function. These legacy types are still publicly exported for backward
compatibility.
"""

import pydantic

# Only `desc` is used by adapters in current DSPy flows; we are deprecating
# `prefix`, `format`, and `parser`. So `desc` can be used in addition to
# the standard `pydantic.Field` arguments. We just hope Pydantic doesn't
# add these names, as that would create a name clash.
# TODO: In a future major release, remove `prefix`, `format`, and `parser`
# from `DSPY_FIELD_ARG_NAMES` and from all related code.
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
    """Translate Pydantic constraint kwargs into a human-readable string.

    Keys from `PYDANTIC_CONSTRAINT_MAP` are turned into a string in which
    the natural language version of the constraint are concatenated together
    and separated by commas. Adapters render that string into the prompt.

    Args:
        **kwargs: The full set of keyword arguments originally passed to
            `InputField` or `OutputField`.  Only keys that appear in
            `PYDANTIC_CONSTRAINT_MAP` are used; all others are ignored.

    Returns:
        str: Comma-separated human-readable constraint descriptions, or an
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
        pydantic.fields.FieldInfo: A field marked as a DSPy input.

    Examples:
        >>> import dspy
        >>> class Summarize(dspy.Signature):
        ...     text: str = dspy.InputField(desc="text to summarize")
        ...     summary: str = dspy.OutputField(desc="a one-sentence summary")
        >>> summarizer = dspy.Predict(Summarize)
        >>> summarizer.lm = dspy.LM("provider/model-name")  # doctest: +SKIP
        >>> summarizer(text="DSPy is a framework for...")  # doctest: +SKIP
        Prediction(...)

        See [`dspy.LM`][dspy.LM] for supported providers.
    """
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs):  # noqa: N802
    """Declare an output field on a `dspy.Signature`.

    Both the type hint and `desc` are optional. When provided,
    adapters pass them to the language model for additional context.
    Using type hint is recommended for output fields so the language
    model knows what type to produce. Adapters also use the type hint
    to determine how to parse the model's outputs into Python objects.

    >>> class QA(dspy.Signature):  # doctest: +SKIP
    ...     question: str = dspy.InputField(desc="a factual question")
    ...     answer: str = dspy.OutputField(desc="a short answer")

    Args:
        desc (str): Optional plain-language description of the field.
            Adapters include this when describing the field to the
            language model. Recommended for output fields.
        **kwargs: Any additional `pydantic.Field` keyword argument.
            Constraints like `gt`, `ge`, `lt`, `le`, `min_length`,
            `max_length`, and `multiple_of` are both described to the
            language model in the prompt and validated against its
            response. If the model's output violates a constraint,
            pydantic raises a `ValidationError`.

    Returns:
        pydantic.fields.FieldInfo: A field marked as a DSPy output.

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

        See [`dspy.LM`][dspy.LM] for supported providers.
    """
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))


def new_to_old_field(field):
    return (OldInputField if field.json_schema_extra["__dspy_field_type"] == "input" else OldOutputField)(
        prefix=field.json_schema_extra["prefix"],
        desc=field.json_schema_extra["desc"],
        format=field.json_schema_extra.get("format"),
    )


class OldField:
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
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=True, format=format)


class OldOutputField(OldField):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=False, format=format)
