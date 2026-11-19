import inspect
import warnings
from collections.abc import Mapping
from copy import deepcopy

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


class _SchemaExtraCallable(Mapping):
    """Wraps a callable json_schema_extra so Pydantic executes it while DSPy can index metadata."""

    def __init__(self, extra_dict: dict, fn):
        self._extra_dict = dict(extra_dict)
        self._fn = fn
        try:
            self.__signature__ = inspect.signature(fn)
        except (ValueError, TypeError):
            pass

    def __call__(self, schema: dict, *args, **call_kwargs) -> None:
        schema.update(self._extra_dict)
        try:
            sig = getattr(self, "__signature__", None) or inspect.signature(self._fn)
        except (ValueError, TypeError):
            # Pydantic's documented callback accepts one argument. If the
            # callable cannot be inspected, use that contract rather than
            # retrying it after a callback error.
            self._fn(schema)
            return

        params = list(sig.parameters.values())
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        num_pos = sum(
            1
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if has_varargs or (num_pos > 1 and len(args) > 0):
            self._fn(schema, *args, **call_kwargs)
        else:
            self._fn(schema)

    def __getitem__(self, item):
        return self._extra_dict[item]

    def __setitem__(self, key, value):
        self._extra_dict[key] = value

    def __iter__(self):
        return iter(self._extra_dict)

    def __len__(self):
        return len(self._extra_dict)

    def __contains__(self, item):
        return item in self._extra_dict

    def get(self, key, default=None):
        return self._extra_dict.get(key, default)

    def keys(self):
        return self._extra_dict.keys()

    def values(self):
        return self._extra_dict.values()

    def items(self):
        return self._extra_dict.items()

    def copy(self):
        return _SchemaExtraCallable(self._extra_dict.copy(), self._fn)

    def __deepcopy__(self, memo):
        return _SchemaExtraCallable(deepcopy(self._extra_dict, memo), self._fn)

    def __eq__(self, other):
        if isinstance(other, _SchemaExtraCallable):
            return self._extra_dict == other._extra_dict and self._fn == other._fn
        return self._extra_dict == other

    def __repr__(self):
        return repr(self._extra_dict)


def move_kwargs(**kwargs):
    # Pydantic doesn't allow arbitrary arguments to be given to fields,
    # but asks that
    # > any extra data you want to add to the JSON schema should be passed
    # > as a dictionary to the json_schema_extra keyword argument.
    # See: https://docs.pydantic.dev/2.6/migration/#changes-to-pydanticfield
    pydantic_kwargs = {}
    user_schema_extra = kwargs.pop("json_schema_extra", None)
    if isinstance(user_schema_extra, dict):
        json_schema_extra = dict(user_schema_extra)
    else:
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

    if callable(user_schema_extra):
        pydantic_kwargs["json_schema_extra"] = _SchemaExtraCallable(json_schema_extra, user_schema_extra)
    else:
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
    _warn_deprecated_field_args(**kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs):  # noqa: N802
    _warn_deprecated_field_args(**kwargs)
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
