"""
lm15.tools — derive a FunctionTool from a Python callable.

Pure functions only. ``tool(fn)`` reads *fn*'s signature, type hints, and
docstring, and returns an ordinary, frozen :class:`lm15.FunctionTool`.
Nothing is registered, cached, wrapped, or executed. The JSON-Schema path
remains primary and canonical: this module is sugar that *produces* that
JSON. The wire format of FunctionTool is untouched (INV-033 intact).

Design notes:

* Derivation is eager: errors surface at definition time, never at request
  time — matching the ``__post_init__`` validation culture of lm15.types.
* Soft where the input is prose, hard where the input is types: docstring
  parsing is best-effort (missing descriptions are never errors), while
  type-hint mapping raises :class:`ToolDerivationError` rather than guess.
* ``Optional[X]`` / ``X | None`` is about VALUE nullability and maps to
  ``anyOf`` with ``{"type": "null"}``. Required-ness comes solely from the
  presence of a default. The two axes are orthogonal.
* No ``$ref`` / recursive schemas in v1: nested dataclasses and TypedDicts
  are inlined, and cycles raise ToolDerivationError. Adding ``$ref`` later
  is therefore an extension, not a behavior change.
* Dispatch stays in user hands — lm15 never executes tools:

      handlers = {fn.__name__: fn for fn in (get_weather,)}
      result = handlers[call.name](**call.input)

Stdlib only: inspect, typing, dataclasses, enum, types.
"""

from __future__ import annotations

import collections.abc as _collections_abc
import dataclasses
import enum
import inspect
import math
import types as _types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

from .errors import LM15Error
from .types import FunctionTool, JsonObject, JsonValue

__all__ = [
    "ToolDerivationError",
    "ToolConfig",
    "DerivedParam",
    "ToolDerivation",
    "tool",
    "derive",
]

_ESCAPE_HATCH = (
    "; override this parameter via ToolConfig(overrides=...) or pass an "
    "explicit FunctionTool with hand-written parameters"
)


class ToolDerivationError(LM15Error):
    """Raised when a callable cannot be derived into a FunctionTool.

    Always names the function and, when applicable, the offending parameter
    and the unsupported annotation, plus both escape hatches: a per-parameter
    ``ToolConfig(overrides=...)`` entry, or a hand-written FunctionTool.
    """

    default_code = "tool_derivation"


@dataclass(frozen=True, slots=True)
class ToolConfig:
    """Options for :func:`tool` / :func:`derive`.

    Explicit over magic: every knob is visible.

    docstring_style: ``"auto"`` tries Google, NumPy, then Sphinx
        (``:param x:``) markers pragmatically; or pin one style; ``"none"``
        skips parameter descriptions entirely.
    include_defaults: emit JSON-compatible default values as ``"default"``
        in the property schema. Informational only; never affects
        required-ness. Non-JSON defaults are silently skipped (documented).
    additional_properties_false: emit ``"additionalProperties": false`` on
        the top-level object. Off by default (provider-neutral); some
        providers' strict tool modes want this — see the cookbook caveat
        about strict mode also requiring all properties to be required.
    overrides: per-parameter raw JSON-Schema fragments used *in place of*
        derivation for the named parameter, e.g.
        ``(("when", {"type": "string", "format": "date-time"}),)``.
        A surgical escape hatch: one un-derivable parameter no longer forces
        a fully hand-written FunctionTool. Stored as a tuple of
        ``(param_name, schema)`` pairs so ToolConfig stays a plain frozen
        dataclass; a docstring description is still merged in when the
        override fragment has no ``"description"`` of its own.
    """

    docstring_style: Literal["auto", "google", "numpy", "sphinx", "none"] = "auto"
    include_defaults: bool = True
    additional_properties_false: bool = False
    overrides: tuple[tuple[str, JsonObject], ...] = ()

    def __post_init__(self) -> None:
        if self.docstring_style not in ("auto", "google", "numpy", "sphinx", "none"):
            raise ValueError(
                "ToolConfig.docstring_style must be one of "
                "'auto', 'google', 'numpy', 'sphinx', 'none'"
            )
        if not isinstance(self.overrides, tuple):
            raise TypeError(
                "ToolConfig.overrides must be a tuple of (name, schema) pairs"
            )
        for entry in self.overrides:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], dict)
            ):
                raise TypeError(
                    "ToolConfig.overrides entries must be (str, dict) pairs"
                )


@dataclass(frozen=True, slots=True)
class DerivedParam:
    """Explainable record of how one parameter was derived."""

    name: str
    annotation: str                 # repr of the hint, e.g. "list[int] | None"
    schema: JsonObject              # the JSON-Schema fragment produced
    required: bool
    description: str | None        # from docstring, if found
    source: Literal["hint", "hint+docstring", "override", "override+docstring"]


@dataclass(frozen=True, slots=True)
class ToolDerivation:
    """Full, typed account of a derivation — :func:`tool`'s explain() analogue.

    Python-only diagnostic surface (other lm15 languages derive from
    idiomatic sources — struct tags, macros, schema literals — and do not
    mirror this type).
    """

    tool: FunctionTool
    fn_qualname: str
    params: tuple[DerivedParam, ...]
    docstring_style_detected: Literal["google", "numpy", "sphinx", "plain", "none"]


# ─── JSON-value check (for defaults and enum/Literal members) ────────

def _is_json_value(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)  # NaN/Infinity are not JSON values
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_value(v) for k, v in value.items())
    return False


def _json_default(value: object) -> JsonValue:
    if isinstance(value, tuple):
        return [_json_default(v) for v in value]
    if isinstance(value, list):
        return [_json_default(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_default(v) for k, v in value.items()}
    return value  # type: ignore[return-value]


# ─── Hint → schema mapping ───────────────────────────────────────────

_PRIMITIVES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_NONE_TYPE = type(None)


def _err(fn_name: str, message: str) -> ToolDerivationError:
    return ToolDerivationError(
        f"cannot derive '{fn_name}': {message}{_ESCAPE_HATCH}"
    )


def _is_typed_dict(hint: object) -> bool:
    return isinstance(hint, type) and hasattr(hint, "__required_keys__") and hasattr(
        hint, "__optional_keys__"
    )


def _schema_for(
    hint: object,
    *,
    fn_name: str,
    param: str,
    seen: frozenset[object],
) -> JsonObject:
    """Map one type hint to a JSON-Schema fragment. Conservative: anything
    not obviously JSON-Schema-able raises ToolDerivationError."""

    def fail(detail: str) -> ToolDerivationError:
        return _err(fn_name, f"parameter '{param}' {detail}")

    origin = get_origin(hint)

    # Annotated[X, "text"]: first str metadata wins as description.
    if origin is typing.Annotated:
        base, *metadata = get_args(hint)
        schema = _schema_for(base, fn_name=fn_name, param=param, seen=seen)
        for meta in metadata:
            if isinstance(meta, str):
                schema = dict(schema)
                schema["description"] = meta
                break
        return schema

    # Any: explicitly-anything.
    if hint is Any:
        return {}

    # None / NoneType.
    if hint is None or hint is _NONE_TYPE:
        return {"type": "null"}

    # Unions (typing.Union and X | Y).
    if origin is Union or origin is _types.UnionType:
        members = [
            _schema_for(arg, fn_name=fn_name, param=param, seen=seen)
            for arg in get_args(hint)
        ]
        return {"anyOf": members}

    # Literal["a", "b", 3] -> enum (+ type when homogeneous).
    if origin is Literal:
        values = get_args(hint)
        for v in values:
            if not (v is None or isinstance(v, (str, bool, int, float))):
                raise fail(
                    f"has Literal value {v!r} that is not JSON-compatible"
                )
        schema = {"enum": list(values)}
        kinds = {type(v) for v in values}
        if len(kinds) == 1:
            sole = next(iter(kinds))
            if sole in _PRIMITIVES:
                schema["type"] = _PRIMITIVES[sole]
            elif sole is _NONE_TYPE:
                schema["type"] = "null"
        return schema

    # Containers.
    if origin in (list, typing.Sequence, _abc_sequence()):
        args = get_args(hint)
        if not args:  # bare typing.List / typing.Sequence
            return {"type": "array"}
        items = _schema_for(args[0], fn_name=fn_name, param=param, seen=seen)
        return {"type": "array", "items": items}
    if origin in (set, frozenset):
        args = get_args(hint)
        if not args:  # bare typing.Set / typing.FrozenSet
            return {"type": "array", "uniqueItems": True}
        items = _schema_for(args[0], fn_name=fn_name, param=param, seen=seen)
        return {"type": "array", "items": items, "uniqueItems": True}
    if origin is tuple:
        args = get_args(hint)
        if not args:  # bare typing.Tuple
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            items = _schema_for(args[0], fn_name=fn_name, param=param, seen=seen)
            return {"type": "array", "items": items}
        raise fail(
            f"has fixed-length tuple annotation {_annotation_repr(hint)}; "
            "only homogeneous tuple[X, ...] is supported"
        )
    if origin in (dict, typing.Mapping, _abc_mapping()):
        dict_args = get_args(hint)
        if not dict_args:  # bare typing.Dict / typing.Mapping
            return {"type": "object"}
        key_t, value_t = dict_args
        if key_t is not str:
            raise fail(
                f"has dict annotation with non-str keys ({_annotation_repr(hint)}); "
                "JSON object keys must be strings"
            )
        value_schema = _schema_for(value_t, fn_name=fn_name, param=param, seen=seen)
        return {"type": "object", "additionalProperties": value_schema}

    if origin is not None:
        raise fail(f"has unsupported annotation {_annotation_repr(hint)}")

    # Bare (unparameterized) types.
    if isinstance(hint, type):
        if hint in _PRIMITIVES:
            # NB: bool before int matters in isinstance checks, but here the
            # dict lookup is exact, so bool maps to "boolean" correctly.
            return {"type": _PRIMITIVES[hint]}
        if hint in (list, tuple):
            return {"type": "array"}
        if hint in (set, frozenset):
            return {"type": "array", "uniqueItems": True}
        if hint is dict:
            return {"type": "object"}

        # Enum subclass -> enum of member values (values must be JSON).
        if issubclass(hint, enum.Enum):
            values = [member.value for member in hint]
            for v in values:
                if not (v is None or isinstance(v, (str, bool, int, float))):
                    raise fail(
                        f"uses Enum {hint.__name__!s} whose member value {v!r} "
                        "is not JSON-compatible"
                    )
            return {"enum": values}

        # TypedDict -> nested object (no $ref/recursion in v1).
        if _is_typed_dict(hint):
            if hint in seen:
                raise fail(
                    f"has recursive TypedDict annotation {hint.__name__!s}; "
                    "recursive ($ref) schemas are not supported in v1"
                )
            try:
                nested_hints = get_type_hints(hint, include_extras=True)
            except Exception as exc:
                raise fail(
                    f"has TypedDict annotation {hint.__name__!s} whose hints "
                    f"could not be resolved ({exc})"
                ) from exc
            nested_seen = seen | {hint}
            properties: JsonObject = {}
            for key, key_hint in nested_hints.items():
                properties[key] = _schema_for(
                    key_hint, fn_name=fn_name, param=f"{param}.{key}", seen=nested_seen
                )
            required = [k for k in nested_hints if k in hint.__required_keys__]
            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            return schema

        # @dataclass class -> nested object from its fields (same rules).
        if dataclasses.is_dataclass(hint):
            if hint in seen:
                raise fail(
                    f"has recursive dataclass annotation {hint.__name__!s}; "
                    "recursive ($ref) schemas are not supported in v1"
                )
            try:
                nested_hints = get_type_hints(hint, include_extras=True)
            except Exception as exc:
                raise fail(
                    f"has dataclass annotation {hint.__name__!s} whose hints "
                    f"could not be resolved ({exc})"
                ) from exc
            nested_seen = seen | {hint}
            properties = {}
            required = []
            for f in dataclasses.fields(hint):
                if not f.init:
                    continue
                properties[f.name] = _schema_for(
                    nested_hints.get(f.name, f.type),
                    fn_name=fn_name,
                    param=f"{param}.{f.name}",
                    seen=nested_seen,
                )
                if (
                    f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING
                ):
                    required.append(f.name)
            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            return schema

    raise fail(
        f"has unsupported annotation {_annotation_repr(hint)}; annotate with "
        "a JSON-compatible type"
    )


def _abc_sequence() -> object:
    return _collections_abc.Sequence


def _abc_mapping() -> object:
    return _collections_abc.Mapping


def _annotation_repr(hint: object) -> str:
    if isinstance(hint, type):
        module = getattr(hint, "__module__", "")
        if module and module != "builtins":
            return f"{module}.{hint.__qualname__}"
        return hint.__qualname__
    return repr(hint)


# ─── Docstring parsing (pragmatic line-marker detection, not a parser) ───

def _docstring_summary(doc: str) -> str | None:
    lines = doc.strip().splitlines()
    summary: list[str] = []
    for line in lines:
        if not line.strip():
            break
        summary.append(line.strip())
    return " ".join(summary) if summary else None


def _parse_google(doc: str) -> dict[str, str] | None:
    lines = doc.splitlines()
    out: dict[str, str] = {}
    in_args = False
    current: str | None = None
    section_indent = 0
    for line in lines:
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            current = None
            continue
        if not in_args:
            continue
        if not stripped:
            current = None
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.endswith(":") and " " not in stripped and indent == 0:
            break  # next top-level section
        if ":" in stripped:
            name_part, _, desc = stripped.partition(":")
            name = name_part.split("(")[0].strip()
            if name.isidentifier():
                out[name] = desc.strip()
                current = name
                section_indent = indent
                continue
        if current is not None and indent > section_indent:
            out[current] = (out[current] + " " + stripped).strip()
    return out or None


def _parse_numpy(doc: str) -> dict[str, str] | None:
    lines = doc.splitlines()
    out: dict[str, str] = {}
    in_params = False
    current: str | None = None
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in ("Parameters", "Other Parameters") and i + 1 < len(lines) and set(
            lines[i + 1].strip()
        ) == {"-"}:
            in_params = True
            current = None
            i += 2
            continue
        if in_params:
            if stripped and i + 1 < len(lines) and set(lines[i + 1].strip() or ".") == {"-"}:
                break  # next underlined section
            if not stripped:
                current = None
            else:
                indent = len(lines[i]) - len(lines[i].lstrip())
                name = stripped.split(":")[0].strip()
                if indent == 0 and name.isidentifier():
                    out[name] = ""
                    current = name
                elif current is not None:
                    out[current] = (out[current] + " " + stripped).strip()
        i += 1
    return {k: v for k, v in out.items()} or None


def _parse_sphinx(doc: str) -> dict[str, str] | None:
    out: dict[str, str] = {}
    current: str | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(":param "):
            rest = stripped[len(":param "):]
            name_part, _, desc = rest.partition(":")
            name = name_part.split()[-1] if name_part.split() else ""
            if name.isidentifier():
                out[name] = desc.strip()
                current = name
            continue
        if stripped.startswith(":"):
            current = None
            continue
        if current is not None and stripped:
            out[current] = (out[current] + " " + stripped).strip()
        elif not stripped:
            current = None
    return out or None


def _parse_docstring(
    doc: str | None, style: str
) -> tuple[dict[str, str], Literal["google", "numpy", "sphinx", "plain", "none"]]:
    if not doc:
        return {}, "none"
    if style == "none":
        return {}, "plain"
    attempts: tuple[tuple[str, Callable[[str], dict[str, str] | None]], ...]
    if style == "auto":
        attempts = (
            ("google", _parse_google),
            ("numpy", _parse_numpy),
            ("sphinx", _parse_sphinx),
        )
    else:
        parser = {"google": _parse_google, "numpy": _parse_numpy, "sphinx": _parse_sphinx}[style]
        attempts = ((style, parser),)  # type: ignore[assignment]
    for detected, parser in attempts:
        result = parser(doc)
        if result:
            return result, detected  # type: ignore[return-value]
    return {}, "plain"


# ─── Public API ──────────────────────────────────────────────────────

def derive(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    config: ToolConfig = ToolConfig(),
) -> ToolDerivation:
    """Derive a FunctionTool from *fn* with a full, typed account.

    Like :func:`tool`, but returns the :class:`ToolDerivation` so users can
    inspect exactly how each parameter was mapped (explainability parity
    with the explain() ethos). ``tool(fn)`` is sugar for ``derive(fn).tool``.
    """
    if not callable(fn):
        raise ToolDerivationError(
            f"cannot derive tool from non-callable {fn!r}{_ESCAPE_HATCH}"
        )

    fn_name = name
    if fn_name is None:
        fn_name = getattr(fn, "__name__", None)
        if fn_name is None:
            raise ToolDerivationError(
                f"cannot derive tool from {fn!r}: it has no __name__; "
                f"pass name= explicitly{_ESCAPE_HATCH}"
            )
        if fn_name == "<lambda>":
            raise ToolDerivationError(
                "cannot derive tool from a lambda: '<lambda>' is not a usable "
                f"tool name; pass name= explicitly{_ESCAPE_HATCH}"
            )

    try:
        signature = inspect.signature(fn)
    except (ValueError, TypeError) as exc:
        raise _err(
            fn_name,
            f"its signature is not introspectable ({exc})",
        ) from exc

    try:
        hints = get_type_hints(fn, include_extras=True)
    except NameError as exc:
        raise _err(
            fn_name,
            f"a type annotation could not be resolved ({exc}); forward "
            "references must be importable at derivation time — define the "
            "referenced type before calling derive(), or annotate with a "
            "resolvable JSON-compatible type",
        ) from exc
    except Exception as exc:
        raise _err(
            fn_name, f"type hints could not be resolved ({exc})"
        ) from exc

    overrides = dict(config.overrides)
    doc = inspect.getdoc(fn)
    param_docs, detected = _parse_docstring(doc, config.docstring_style)

    derived: list[DerivedParam] = []
    properties: JsonObject = {}
    required: list[str] = []

    parameters = list(signature.parameters.values())
    # Bound methods and classmethods already have self/cls stripped by
    # inspect.signature; an explicit leading self/cls only remains on plain
    # unbound functions, where it is treated as an ordinary parameter.

    for p in parameters:
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            raise _err(
                fn_name,
                f"parameter '*{p.name}' is variadic; tool signatures must be "
                "fully nameable — wrap the function",
            )
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            raise _err(
                fn_name,
                f"parameter '**{p.name}' is variadic; tool signatures must "
                "be fully nameable — wrap the function",
            )
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise _err(
                fn_name,
                f"parameter '{p.name}' is positional-only; tool calls "
                "dispatch by name — make it keyword-able",
            )

        doc_desc = param_docs.get(p.name)

        if p.name in overrides:
            schema = dict(overrides[p.name])
            if doc_desc and "description" not in schema:
                schema["description"] = doc_desc
            source: Literal[
                "hint", "hint+docstring", "override", "override+docstring"
            ] = "override+docstring" if doc_desc else "override"
            annotation_repr = (
                _annotation_repr(hints[p.name]) if p.name in hints else "<override>"
            )
        else:
            if p.name not in hints:
                raise _err(
                    fn_name,
                    f"parameter '{p.name}' has no type annotation",
                )
            hint = hints[p.name]
            schema = _schema_for(
                hint, fn_name=fn_name, param=p.name, seen=frozenset()
            )
            schema = dict(schema)
            if doc_desc and "description" not in schema:
                schema["description"] = doc_desc
            source = "hint+docstring" if doc_desc else "hint"
            annotation_repr = _annotation_repr(hint)

        is_required = p.default is inspect.Parameter.empty
        if is_required:
            required.append(p.name)
        elif (
            config.include_defaults
            and "default" not in schema
            and _is_json_value(p.default)
        ):
            schema["default"] = _json_default(p.default)

        properties[p.name] = schema
        derived.append(
            DerivedParam(
                name=p.name,
                annotation=annotation_repr,
                schema=schema,
                required=is_required,
                description=schema.get("description"),  # type: ignore[arg-type]
                source=source,
            )
        )

    unknown_overrides = set(overrides) - set(signature.parameters)
    if unknown_overrides:
        raise _err(
            fn_name,
            "ToolConfig.overrides names parameters not in the signature: "
            + ", ".join(sorted(unknown_overrides)),
        )

    schema_top: JsonObject = {"type": "object", "properties": properties}
    if required:
        schema_top["required"] = required
    if config.additional_properties_false:
        schema_top["additionalProperties"] = False

    tool_description = description
    if tool_description is None and doc:
        tool_description = _docstring_summary(doc)

    derived_tool = FunctionTool(
        name=fn_name,
        description=tool_description,
        parameters=schema_top,
    )

    return ToolDerivation(
        tool=derived_tool,
        fn_qualname=getattr(fn, "__qualname__", fn_name),
        params=tuple(derived),
        docstring_style_detected=detected,
    )


def tool(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    config: ToolConfig = ToolConfig(),
) -> FunctionTool:
    """Derive a FunctionTool from *fn*. Pure; returns a plain FunctionTool.

    ``name``/``description`` override derivation (name falls back to
    ``fn.__name__``, description to the docstring summary paragraph, else
    None). A parameter is required iff it has no default; ``Optional[X]``
    affects only value nullability (``anyOf`` with null), never
    required-ness. See the module docstring for the full mapping table and
    :class:`ToolConfig` for the knobs.

    Deliberately not a decorator: a decorator would replace or wrap the
    function — magic. Call ``tool(fn)`` and keep *fn* yourself; dispatch with
    ``{f.__name__: f for f in (...)}``.
    """
    return derive(fn, name=name, description=description, config=config).tool
