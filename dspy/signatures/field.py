"""Define DSPy field factories and the kwargs-splitting convention.

This module provides the public `InputField` and `OutputField` factory
functions that DSPy users call when declaring signature fields. Both
factories delegate to `pydantic.Field` but first split caller-supplied
keyword arguments into two groups:

- **DSPy-specific kwargs** (listed in `DSPY_FIELD_ARG_NAMES`) — stored
  under `json_schema_extra` so DSPy code can inspect them later.
- **Standard Pydantic kwargs** — forwarded directly to `pydantic.Field`.

The splitting is performed by the internal `move_kwargs` helper. In
addition, any Pydantic constraint kwargs (`ge`, `le`, `min_length`,
etc.) are translated into a human-readable `"constraints"` string and
stored alongside the DSPy metadata in `json_schema_extra`.

Important:
    In the current codebase, `desc` and translated `constraints` affect how
    the standard adapters describe fields to the language model. By contrast,
    field-level `prefix`, `format`, and `parser` **HAVE NO EFFECT** on
    standard prompt formatting, output parsing, or model I/O. They should be
    treated as inert metadata. Users should not expect them to do anything
    in normal DSPy prediction flows.

This module also contains the legacy `OldField` / `OldInputField` /
`OldOutputField` classes and a `new_to_old_field` bridge function.
These legacy types are still publicly exported for backward compatibility.
"""

import pydantic

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

    Scans `kwargs` for keys present in `PYDANTIC_CONSTRAINT_MAP` and
    builds a comma-separated description of all matched constraints. The
    result is stored by `move_kwargs` under
    `json_schema_extra["constraints"]`.

    In the current codebase, adapter utilities include this constraints string
    when describing fields to the language model.

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
        >>> _translate_pydantic_field_constraints(desc="name")
        ''
    """
    constraints = []
    for key, value in kwargs.items():
        if key in PYDANTIC_CONSTRAINT_MAP:
            constraints.append(f"{PYDANTIC_CONSTRAINT_MAP[key]}{value}")

    return ", ".join(constraints)


def InputField(**kwargs):  # noqa: N802
    """Create a DSPy input field backed by `pydantic.Field`.

    This factory is the standard way to declare an input on a DSPy
    `Signature`. It accepts all `pydantic.Field` keyword arguments
    **plus** the DSPy-specific kwargs listed in `DSPY_FIELD_ARG_NAMES`:

    - `desc` — description stored in metadata and used by the standard
      adapters when they describe fields to the language model. Current
      optimizer flows such as `GEPA` and `MIPROv2` do not optimize this
      per-field metadata.
      
    - `prefix` — **CURRENTLY HAS NO EFFECT.** Intended as a prompt label, but ignored by modern DSPy adapters.
    - `format` — **CURRENTLY HAS NO EFFECT.** Intended for custom formatting, but never invoked by DSPy.
    - `parser` — **CURRENTLY HAS NO EFFECT.** Intended for custom parsing, but never invoked by DSPy.

    If `description` (the standard Pydantic kwarg) is provided but `desc`
    is not, the `description` value is copied to `desc` automatically.

    Pydantic constraint kwargs (`ge`, `le`, `min_length`, `max_length`,
    etc.) are translated into a human-readable `"constraints"` string stored
    in `json_schema_extra` and used by current adapter utilities when
    describing fields.

    Note:
        Users should not expect `prefix`, `format`, or `parser` to change
        runtime behavior in ordinary DSPy prediction flows.

    Args:
        **kwargs: Any `pydantic.Field` argument and/or DSPy-specific
            metadata kwargs.

    Returns:
        pydantic.fields.FieldInfo: A Pydantic `FieldInfo` instance with
        `json_schema_extra["__dspy_field_type"]` set to `"input"`.

    Examples:
        Use `InputField` in the idiomatic context: a `dspy.Signature`
        subclass.

        >>> import dspy
        >>> class ClassifySentiment(dspy.Signature):
        ...     text: str = dspy.InputField(desc="Text to classify", min_length=1)
        ...     label: str = dspy.OutputField(desc="Sentiment label")
        >>> field = ClassifySentiment.input_fields["text"]
        >>> field.json_schema_extra["__dspy_field_type"]
        'input'
        >>> field.json_schema_extra["desc"]
        'Text to classify'
        >>> 'minimum length: 1' in field.json_schema_extra["constraints"]
        True
    """
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs):  # noqa: N802
    """Create a DSPy output field backed by `pydantic.Field`.

    This factory is the standard way to declare an output on a DSPy
    `Signature`. It accepts all `pydantic.Field` keyword arguments
    **plus** the DSPy-specific kwargs listed in `DSPY_FIELD_ARG_NAMES`:

    - `desc` — description stored in metadata and used by the standard
      adapters when they describe fields to the language model. Current
      optimizer flows such as `GEPA` and `MIPROv2` do not optimize this
      per-field metadata.

    - `prefix` — **CURRENTLY HAS NO EFFECT.** Intended as a prompt label, but ignored by modern DSPy adapters.
    - `format` — **CURRENTLY HAS NO EFFECT.** Intended for custom formatting, but never invoked by DSPy.
    - `parser` — **CURRENTLY HAS NO EFFECT.** Intended for custom parsing, but never invoked by DSPy.

    If `description` (the standard Pydantic kwarg) is provided but `desc`
    is not, the `description` value is copied to `desc` automatically.

    Pydantic constraint kwargs (`ge`, `le`, `min_length`, `max_length`,
    etc.) are translated into a human-readable `"constraints"` string stored
    in `json_schema_extra` and used by current adapter utilities when
    describing fields.

    Note:
        Users should not expect `prefix`, `format`, or `parser` to change
        runtime behavior in ordinary DSPy prediction flows.

    Args:
        **kwargs: Any `pydantic.Field` argument and/or DSPy-specific
            metadata kwargs.

    Returns:
        pydantic.fields.FieldInfo: A Pydantic `FieldInfo` instance with
        `json_schema_extra["__dspy_field_type"]` set to `"output"`.

    Examples:
        Use `OutputField` in the idiomatic context: a `dspy.Signature`
        subclass.

        >>> import dspy
        >>> class ClassifySentiment(dspy.Signature):
        ...     text: str = dspy.InputField(desc="Text to classify")
        ...     label: str = dspy.OutputField(desc="Sentiment label", max_length=20)
        >>> field = ClassifySentiment.output_fields["label"]
        >>> field.json_schema_extra["__dspy_field_type"]
        'output'
        >>> field.json_schema_extra["desc"]
        'Sentiment label'
        >>> 'maximum length: 20' in field.json_schema_extra["constraints"]
        True
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
