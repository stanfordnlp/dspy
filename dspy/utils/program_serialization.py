"""Safe, pickle-free declarative serialization for DSPy programs.

Serializes supported DSPy modules and allowlisted custom ``dspy.Module``
subclasses to JSON without invoking ``cloudpickle``, ``eval``, or dynamic
module imports during deserialization.
"""

from __future__ import annotations

import ast
import collections.abc
import dataclasses
import datetime
import inspect
import keyword
import math
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard, cast, get_args, get_origin

import orjson
import pydantic
from pydantic_core import PydanticUndefined

from dspy.adapters.types.audio import Audio
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.bounding_box import BoundingBox
from dspy.adapters.types.code import Code
from dspy.adapters.types.file import File
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.predict.parameter import Parameter
from dspy.predict.predict import _sanitize_lm_state
from dspy.primitives.base_module import BaseModule
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLHistory, REPLVariable
from dspy.signatures.signature import Signature, make_signature
from dspy.utils.constants import IS_TYPE_UNDEFINED

if TYPE_CHECKING:
    from dspy.primitives.module import Module

PROGRAM_SCHEMA_VERSION = 1

_MAX_TYPE_EXPR_LENGTH = 4096
_MAX_TYPE_AST_DEPTH = 32
_BASE_MODULE_RUNTIME_ATTRS = frozenset({"_compiled", "callbacks", "history"})

_PREDICT_STATE_KEYS = frozenset({"traces", "train", "demos", "signature", "fields", "lm"})
_FLEX_STATE_KEYS = frozenset({"module_src", "lm"})
_SAFE_EXTRA_METADATA_KEYS = frozenset(
    {
        "__dspy_field_type",
        "__dspy_reasoning_fallback",
        "constraints",
        "desc",
        "format",
        "prefix",
        IS_TYPE_UNDEFINED,
    }
)

# ``pydantic.Field`` constraints, which pydantic stores in ``FieldInfo.metadata``, and their JSON value types.
_FIELD_CONSTRAINTS: dict[str, tuple[type, ...]] = {
    "gt": (int, float),
    "ge": (int, float),
    "lt": (int, float),
    "le": (int, float),
    "multiple_of": (int, float),
    "min_length": (int,),
    "max_length": (int,),
    "max_digits": (int,),
    "decimal_places": (int,),
    "pattern": (str,),
    "union_mode": (str,),
    "strict": (bool,),
    "allow_inf_nan": (bool,),
    "coerce_numbers_to_str": (bool,),
    "fail_fast": (bool,),
}

_SAFE_TYPE_NAMESPACE: dict[str, Any] = {
    "None": type(None),
    "NoneType": type(None),
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "complex": complex,
    "dict": dict,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "set": set,
    "str": str,
    "tuple": tuple,
    "Any": Any,
    "Optional": typing.Optional,
    "Union": typing.Union,
    "Literal": typing.Literal,
    "List": list,
    "Dict": dict,
    "Tuple": tuple,
    "Set": set,
    "FrozenSet": frozenset,
    "Sequence": collections.abc.Sequence,
    "Mapping": collections.abc.Mapping,
    "MutableSequence": collections.abc.MutableSequence,
    "MutableMapping": collections.abc.MutableMapping,
    "Iterable": collections.abc.Iterable,
    "AbstractSet": collections.abc.Set,
    "Image": Image,
    "dspy.Image": Image,
    "Audio": Audio,
    "dspy.Audio": Audio,
    "File": File,
    "dspy.File": File,
    "History": History,
    "dspy.History": History,
    "Reasoning": Reasoning,
    "dspy.Reasoning": Reasoning,
    "Tool": Tool,
    "dspy.Tool": Tool,
    "ToolCalls": ToolCalls,
    "dspy.ToolCalls": ToolCalls,
    "ToolCallResults": ToolCallResults,
    "dspy.ToolCallResults": ToolCallResults,
    "Code": Code,
    "dspy.Code": Code,
    "BoundingBox": BoundingBox,
    "dspy.BoundingBox": BoundingBox,
    "Type": Type,
    "dspy.Type": Type,
    "REPLHistory": REPLHistory,
    "dspy.REPLHistory": REPLHistory,
    "REPLVariable": REPLVariable,
    "dspy.REPLVariable": REPLVariable,
    "datetime.date": datetime.date,
    "datetime.datetime": datetime.datetime,
    "datetime.time": datetime.time,
}

_CANONICAL_TYPE_NAMES: dict[Any, str] = {
    type(None): "None",
    bool: "bool",
    bytearray: "bytearray",
    bytes: "bytes",
    complex: "complex",
    dict: "dict",
    float: "float",
    frozenset: "frozenset",
    int: "int",
    list: "list",
    set: "set",
    str: "str",
    tuple: "tuple",
    Any: "Any",
    collections.abc.Sequence: "Sequence",
    collections.abc.Mapping: "Mapping",
    collections.abc.MutableSequence: "MutableSequence",
    collections.abc.MutableMapping: "MutableMapping",
    collections.abc.Iterable: "Iterable",
    collections.abc.Set: "AbstractSet",
    Image: "dspy.Image",
    Audio: "dspy.Audio",
    File: "dspy.File",
    History: "dspy.History",
    Reasoning: "dspy.Reasoning",
    Tool: "dspy.Tool",
    ToolCalls: "dspy.ToolCalls",
    ToolCallResults: "dspy.ToolCallResults",
    Code: "dspy.Code",
    BoundingBox: "dspy.BoundingBox",
    Type: "dspy.Type",
    REPLHistory: "dspy.REPLHistory",
    REPLVariable: "dspy.REPLVariable",
    datetime.date: "datetime.date",
    datetime.datetime: "datetime.datetime",
    datetime.time: "datetime.time",
}

_UNARY_GENERIC_ORIGINS = frozenset(
    {
        list,
        set,
        frozenset,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
        collections.abc.Iterable,
        collections.abc.Set,
    }
)
_BINARY_GENERIC_ORIGINS = frozenset(
    {
        dict,
        collections.abc.Mapping,
        collections.abc.MutableMapping,
    }
)


def _is_strict_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_constraint(key: Any, value: Any, *, context: str) -> None:
    allowed = _FIELD_CONSTRAINTS.get(key)
    if allowed is None:
        raise ValueError(f"{context} uses unsupported constraint {key!r} for JSON program serialization.")
    if isinstance(value, bool) and bool not in allowed:
        raise ValueError(f"{context} constraint {key!r} must be {allowed[0].__name__}, got bool.")
    if not isinstance(value, allowed) or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"{context} constraint {key!r}={value!r} cannot be serialized to JSON.")


def _metadata_constraints(metadata: Any, *, context: str) -> dict[str, Any]:
    """Return the ``pydantic.Field`` constraints behind a ``FieldInfo.metadata`` entry, rejecting anything else."""
    owner = type(metadata).__module__.partition(".")[0]
    if owner not in ("annotated_types", "pydantic"):
        raise ValueError(f"{context} uses unsupported metadata {metadata!r} for JSON program serialization.")
    if dataclasses.is_dataclass(metadata):
        items = {f.name: getattr(metadata, f.name) for f in dataclasses.fields(metadata)}
    else:
        items = dict(vars(metadata))
    for key, value in items.items():
        _check_constraint(key, value, context=context)
    return items


def _require_signature(signature: type[Signature] | None, *, context: str) -> type[Signature]:
    if signature is None:
        raise ValueError(f"{context} does not have a signature.")
    return signature


def _validate_json_value(value: Any, *, context: str) -> None:
    """Ensure ``value`` contains only plain JSON-serializable data."""
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{context} contains non-finite float value {value!r}, which cannot be serialized to JSON."
            )
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, context=context)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains non-string dict key {key!r}.")
            _validate_json_value(item, context=context)
        return
    raise ValueError(f"{context} contains unsupported non-JSON value of type {type(value).__name__}.")


def _ensure_json_dict(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a dict, got {type(value).__name__}.")
    _validate_json_value(value, context=context)
    return value


def serialize_type_annotation(annotation: Any) -> str:
    """Serialize a Python type annotation into a safe canonical type expression."""
    if annotation is None or annotation is type(None):
        return "None"

    origin = get_origin(annotation)
    if origin is None:
        if annotation in _CANONICAL_TYPE_NAMES:
            return _CANONICAL_TYPE_NAMES[annotation]
        if isinstance(annotation, type) and issubclass(annotation, Code) and annotation is not Code:
            language = getattr(annotation, "language", None)
            if isinstance(language, str):
                return f"dspy.Code[{language!r}]"
        raise ValueError(
            f"Unsupported type annotation for JSON program serialization: {annotation!r}. "
            "Only built-in types, standard typing generics, Literal/Optional/Union, datetime types, "
            "and built-in DSPy types are supported."
        )

    args = get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        if len(args) == 2 and type(None) in args:
            other = args[1] if args[0] is type(None) else args[0]
            return f"Optional[{serialize_type_annotation(other)}]"
        rendered_args = ", ".join(serialize_type_annotation(arg) for arg in args)
        return f"Union[{rendered_args}]"

    if origin is typing.Literal:
        if not args:
            raise ValueError("Literal annotation must have at least one argument.")
        for arg in args:
            if not (arg is None or isinstance(arg, (str, int, bool))):
                raise ValueError(
                    f"Unsupported Literal argument {arg!r} in {annotation!r} for JSON program serialization."
                )
        rendered_args = ", ".join(repr(arg) for arg in args)
        return f"Literal[{rendered_args}]"

    if origin is tuple:
        if not args:
            return "tuple"
        if len(args) == 1 and args[0] == ():
            return "tuple[()]"
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{serialize_type_annotation(args[0])}, ...]"
        rendered_args = ", ".join(serialize_type_annotation(arg) for arg in args)
        return f"tuple[{rendered_args}]"

    if origin in _UNARY_GENERIC_ORIGINS or origin in _BINARY_GENERIC_ORIGINS:
        origin_name = _CANONICAL_TYPE_NAMES[origin]
        if not args:
            return origin_name
        rendered_args = ", ".join(serialize_type_annotation(arg) for arg in args)
        return f"{origin_name}[{rendered_args}]"

    raise ValueError(f"Unsupported generic type annotation for JSON program serialization: {annotation!r}.")


def _extract_dotted_name(node: ast.AST) -> str:
    """Flatten ``Name`` / ``Attribute`` AST chains without dynamic attribute access."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        if current.attr.startswith("__"):
            raise ValueError(f"Disallowed dunder attribute in type annotation: {current.attr!r}.")
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise ValueError(f"Disallowed syntax in type name: {ast.dump(node)}.")
    if current.id.startswith("__"):
        raise ValueError(f"Disallowed dunder identifier in type annotation: {current.id!r}.")
    parts.append(current.id)
    parts.reverse()
    return ".".join(parts)


def _parse_literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (str, int, bool)):
            return node.value
        raise ValueError(f"Disallowed Literal constant type: {type(node.value).__name__}.")
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and _is_strict_int(node.operand.value)
    ):
        val = node.operand.value
        return val if isinstance(node.op, ast.UAdd) else -val
    raise ValueError(f"Disallowed Literal argument syntax: {ast.dump(node)}.")


def _resolve_type_ast(node: ast.AST, *, depth: int) -> Any:
    if depth > _MAX_TYPE_AST_DEPTH:
        raise ValueError("Type annotation exceeds maximum nesting depth.")

    if isinstance(node, ast.Constant):
        if node.value is None:
            return type(None)
        raise ValueError(f"Unexpected constant {node.value!r} in type annotation.")

    if isinstance(node, (ast.Name, ast.Attribute)):
        dotted = _extract_dotted_name(node)
        if dotted not in _SAFE_TYPE_NAMESPACE:
            raise ValueError(f"Type '{dotted}' is not in the safe type allowlist.")
        return _SAFE_TYPE_NAMESPACE[dotted]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _resolve_type_ast(node.left, depth=depth + 1)
        right = _resolve_type_ast(node.right, depth=depth + 1)
        union_form: Any = typing.Union
        return union_form[left, right]

    if isinstance(node, ast.Subscript):
        base_name = _extract_dotted_name(node.value)
        if base_name not in _SAFE_TYPE_NAMESPACE:
            raise ValueError(f"Generic base type '{base_name}' is not in the safe type allowlist.")
        base = _SAFE_TYPE_NAMESPACE[base_name]
        slice_node = node.slice

        if base is typing.Literal:
            elts = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
            if not elts:
                raise ValueError("Literal requires at least one argument.")
            values = tuple(_parse_literal_value(elt) for elt in elts)
            return typing.Literal[values]

        if base is Code:
            if not isinstance(slice_node, ast.Constant) or not isinstance(slice_node.value, str):
                raise ValueError("dspy.Code[...] requires a string language literal.")
            return Code[slice_node.value]

        if base is typing.Optional:
            if isinstance(slice_node, ast.Tuple):
                raise ValueError("Optional requires exactly one type argument.")
            return base[_resolve_type_ast(slice_node, depth=depth + 1)]

        if base is typing.Union:
            elts = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
            if not elts:
                raise ValueError("Union requires at least one type argument.")
            resolved = tuple(_resolve_type_ast(elt, depth=depth + 1) for elt in elts)
            return base[resolved]

        if base is tuple:
            if isinstance(slice_node, ast.Tuple):
                if len(slice_node.elts) == 0:
                    return types.GenericAlias(tuple, ((),))
                if (
                    len(slice_node.elts) == 2
                    and isinstance(slice_node.elts[1], ast.Constant)
                    and slice_node.elts[1].value is Ellipsis
                ):
                    return types.GenericAlias(tuple, (_resolve_type_ast(slice_node.elts[0], depth=depth + 1), ...))
                resolved = tuple(_resolve_type_ast(elt, depth=depth + 1) for elt in slice_node.elts)
                return types.GenericAlias(tuple, resolved)
            return types.GenericAlias(tuple, (_resolve_type_ast(slice_node, depth=depth + 1),))

        if base in _UNARY_GENERIC_ORIGINS:
            if isinstance(slice_node, ast.Tuple):
                raise ValueError(f"Generic type '{base_name}' requires exactly one type argument.")
            arg = _resolve_type_ast(slice_node, depth=depth + 1)
            return base[arg]

        if base in _BINARY_GENERIC_ORIGINS:
            if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
                raise ValueError(f"Generic type '{base_name}' requires exactly two type arguments.")
            key_type = _resolve_type_ast(slice_node.elts[0], depth=depth + 1)
            val_type = _resolve_type_ast(slice_node.elts[1], depth=depth + 1)
            return base[key_type, val_type]

        raise ValueError(f"Type '{base_name}' cannot be subscripted in safe type annotations.")

    raise ValueError(f"Disallowed AST node in type annotation: {type(node).__name__}.")


def parse_safe_type_annotation(expr: str) -> Any:
    """Parse a type annotation string using strict AST validation and an explicit allowlist."""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"Type annotation must be a non-empty string, got {expr!r}.")
    if len(expr) > _MAX_TYPE_EXPR_LENGTH:
        raise ValueError(f"Type annotation exceeds maximum length of {_MAX_TYPE_EXPR_LENGTH} characters.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid syntax in type annotation {expr!r}: {exc}") from exc
    return _resolve_type_ast(tree.body, depth=0)


def serialize_signature(signature: type[Signature]) -> dict[str, Any]:
    """Serialize a ``dspy.Signature`` class into a JSON-safe declarative spec."""
    fields_list: list[dict[str, Any]] = []
    for name, field_info in signature.fields.items():
        if field_info.default_factory is not None:
            raise ValueError(
                f"Signature field '{name}' uses `default_factory`, which cannot be serialized in JSON format."
            )
        raw_extra = field_info.json_schema_extra
        if raw_extra is None:
            extra: dict[str, Any] = {}
        elif isinstance(raw_extra, dict):
            extra = dict(raw_extra)
        else:
            raise ValueError(
                f"Signature field '{name}' uses a non-dict `json_schema_extra`, which cannot be serialized to JSON."
            )
        field_type = extra.get("__dspy_field_type")
        if field_type not in ("input", "output"):
            raise ValueError(f"Signature field '{name}' has unsupported field type: {field_type!r}.")

        for extra_key, extra_val in extra.items():
            if extra_key not in _SAFE_EXTRA_METADATA_KEYS:
                raise ValueError(
                    f"Signature field '{name}' contains unsupported metadata key '{extra_key}' "
                    "for JSON program serialization."
                )
            _validate_json_value(extra_val, context=f"Signature field '{name}' metadata '{extra_key}'")

        constraints: dict[str, Any] = {}
        for meta in field_info.metadata:
            constraints.update(_metadata_constraints(meta, context=f"Signature field '{name}'"))

        field_spec: dict[str, Any] = {
            "name": name,
            "field_type": field_type,
            "annotation": serialize_type_annotation(field_info.annotation),
            "desc": extra.get("desc", f"${{{name}}}"),
            "prefix": extra.get("prefix"),
        }
        if IS_TYPE_UNDEFINED in extra:
            field_spec["is_type_undefined"] = bool(extra[IS_TYPE_UNDEFINED])
        if "__dspy_reasoning_fallback" in extra:
            field_spec["reasoning_fallback"] = extra["__dspy_reasoning_fallback"]
        if "format" in extra and extra["format"] is not None:
            field_spec["format"] = extra["format"]
        if "constraints" in extra:
            field_spec["constraints_text"] = extra["constraints"]
        if constraints:
            field_spec["constraints"] = constraints
        if field_info.default is not PydanticUndefined:
            _validate_json_value(field_info.default, context=f"Signature field '{name}' default")
            field_spec["has_default"] = True
            field_spec["default"] = field_info.default

        fields_list.append(field_spec)

    return {
        "name": signature.__name__,
        "instructions": signature.instructions,
        "fields": fields_list,
    }


def deserialize_signature(spec: Any) -> type[Signature]:
    """Reconstruct a ``dspy.Signature`` class from a declarative JSON spec."""
    if not isinstance(spec, dict):
        raise ValueError(f"Signature spec must be a dict, got {type(spec).__name__}.")

    sig_name = spec.get("name", "StringSignature")
    if not isinstance(sig_name, str) or not sig_name.isidentifier() or keyword.iskeyword(sig_name):
        raise ValueError(f"Invalid signature name: {sig_name!r}.")

    instructions = spec.get("instructions")
    if not isinstance(instructions, str):
        raise ValueError("Signature 'instructions' must be a string.")

    raw_fields = spec.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError("Signature 'fields' must be a list.")

    parsed_fields: dict[str, tuple[Any, Any]] = {}
    for entry in raw_fields:
        if not isinstance(entry, dict):
            raise ValueError("Each signature field entry must be a dict.")

        field_name = entry.get("name")
        if (
            not isinstance(field_name, str)
            or not field_name.isidentifier()
            or keyword.iskeyword(field_name)
            or field_name.startswith("_")
        ):
            raise ValueError(f"Invalid signature field name: {field_name!r}.")
        if field_name in parsed_fields:
            raise ValueError(f"Duplicate signature field name: '{field_name}'.")

        field_type = entry.get("field_type")
        if field_type not in ("input", "output"):
            raise ValueError(f"Invalid field_type for '{field_name}': {field_type!r}.")

        annotation = parse_safe_type_annotation(entry.get("annotation"))

        desc = entry.get("desc", f"${{{field_name}}}")
        if not isinstance(desc, str):
            raise ValueError(f"Field '{field_name}' desc must be a string.")

        prefix = entry.get("prefix")
        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(f"Field '{field_name}' prefix must be a string or None.")

        extra: dict[str, Any] = {
            "__dspy_field_type": field_type,
            "desc": desc,
        }
        if prefix is not None:
            extra["prefix"] = prefix
        if "is_type_undefined" in entry:
            is_type_undefined = entry["is_type_undefined"]
            if not isinstance(is_type_undefined, bool):
                raise ValueError(f"Field '{field_name}' is_type_undefined must be a bool.")
            extra[IS_TYPE_UNDEFINED] = is_type_undefined
        if "reasoning_fallback" in entry:
            fallback = entry["reasoning_fallback"]
            if not isinstance(fallback, str):
                raise ValueError(f"Field '{field_name}' reasoning_fallback must be a string.")
            extra["__dspy_reasoning_fallback"] = fallback
        if "format" in entry:
            fmt = entry["format"]
            if not isinstance(fmt, str):
                raise ValueError(f"Field '{field_name}' format must be a string.")
            extra["format"] = fmt
        if "constraints_text" in entry:
            ctext = entry["constraints_text"]
            if not isinstance(ctext, str):
                raise ValueError(f"Field '{field_name}' constraints_text must be a string.")
            extra["constraints"] = ctext

        pydantic_kwargs: dict[str, Any] = {"json_schema_extra": extra}
        if "constraints" in entry:
            raw_constraints = entry["constraints"]
            if not isinstance(raw_constraints, dict):
                raise ValueError(f"Field '{field_name}' constraints must be a dict.")
            for ckey, cval in raw_constraints.items():
                _check_constraint(ckey, cval, context=f"Field '{field_name}'")
                pydantic_kwargs[ckey] = cval

        if entry.get("has_default", False):
            if "default" not in entry:
                raise ValueError(f"Field '{field_name}' declares has_default=True without 'default'.")
            pydantic_kwargs["default"] = entry["default"]

        parsed_fields[field_name] = (annotation, pydantic.Field(**pydantic_kwargs))

    return make_signature(parsed_fields, instructions=instructions, signature_name=sig_name)


def _normalize_caller_tools(
    tools: Mapping[str, Callable[..., Any] | Tool] | Sequence[Callable[..., Any] | Tool] | None,
) -> dict[str, Tool]:
    """Normalize caller-provided tools into a name -> Tool mapping."""
    if tools is None:
        return {}

    normalized: dict[str, Tool] = {}
    if isinstance(tools, Mapping):
        for name, item in tools.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"Tool mapping keys must be non-empty strings, got {name!r}.")
            if isinstance(item, Tool):
                if item.name != name:
                    raise ValueError(f"Tool mapping key '{name}' does not match Tool.name '{item.name}'.")
                normalized[name] = item
            elif callable(item):
                normalized[name] = Tool(item, name=name)
            else:
                raise TypeError(f"Tool '{name}' must be callable or a dspy.Tool, got {type(item).__name__}.")
        return normalized

    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        for item in tools:
            if isinstance(item, Tool):
                tool = item
            elif callable(item):
                tool = Tool(item)
            else:
                raise TypeError(f"Each tool must be callable or a dspy.Tool, got {type(item).__name__}.")
            if not isinstance(tool.name, str) or not tool.name:
                raise ValueError(f"Tool {tool!r} must have a non-empty string name.")
            if tool.name in normalized:
                raise ValueError(f"Duplicate tool name '{tool.name}' in `tools`.")
            normalized[tool.name] = tool
        return normalized

    raise TypeError("`tools` must be a mapping of tool names to callables/Tools or a sequence of callables/Tools.")


def _bind_required_tools(
    required_names: Any,
    available_tools: dict[str, Tool],
    *,
    module_label: str,
    disallowed_names: frozenset[str] = frozenset(),
) -> list[Tool]:
    if not isinstance(required_names, list) or not all(isinstance(n, str) and n for n in required_names):
        raise ValueError(f"{module_label} 'tool_names' must be a list of non-empty strings.")
    if len(set(required_names)) != len(required_names):
        raise ValueError(f"{module_label} 'tool_names' contains duplicate tool names.")
    disallowed = sorted(disallowed_names.intersection(required_names))
    if disallowed:
        raise ValueError(f"{module_label} 'tool_names' contains reserved tool name(s): {', '.join(disallowed)}.")

    missing = [name for name in required_names if name not in available_tools]
    if missing:
        raise ValueError(
            f"Missing required tool(s) for {module_label}: {', '.join(missing)}. "
            "Provide them via `dspy.load(path, tools=...)`."
        )
    return [available_tools[name] for name in required_names]


def _validate_predict_state(state: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError(f"{context} state must be a dict, got {type(state).__name__}.")
    unexpected = sorted(set(state) - _PREDICT_STATE_KEYS)
    if unexpected:
        raise ValueError(f"{context} state contains unexpected key(s): {', '.join(unexpected)}.")
    if "signature" not in state or "lm" not in state:
        raise ValueError(f"{context} state is missing required keys ('signature', 'lm').")
    if state["lm"] is not None and not isinstance(state["lm"], dict):
        raise ValueError(f"{context} state 'lm' must be a dict or None.")
    return state


def _dump_predictor_config(config: Any, *, context: str) -> dict[str, Any]:
    """Return a predictor's config without its API key, as ``BaseLM.dump_state`` does."""
    return {key: value for key, value in _ensure_json_dict(config, context=context).items() if key != "api_key"}


def _load_predictor_config(
    config: Any,
    *,
    allow_unsafe_lm_state: bool,
    context: str,
    forbidden_keys: frozenset[str] = frozenset({"signature", "callbacks"}),
) -> dict[str, Any]:
    """Validate a saved predictor config and sanitize it like LM state, since it overrides the LM's endpoint."""
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"{context} must be a dict, got {type(config).__name__}.")
    _validate_json_value(config, context=context)
    collisions = sorted(forbidden_keys.intersection(config))
    if collisions:
        raise ValueError(f"{context} contains disallowed key(s): {', '.join(collisions)}.")
    return dict(_sanitize_lm_state(config, allow_unsafe_lm_state))


def _builtin_registry() -> dict[tuple[str, str], type[Module]]:
    from dspy.predict.chain_of_thought import ChainOfThought
    from dspy.predict.flex.flex import Flex
    from dspy.predict.multi_chain_comparison import MultiChainComparison
    from dspy.predict.predict import Predict
    from dspy.predict.react import ReAct
    from dspy.predict.rlm import RLM

    classes: tuple[type[Module], ...] = (
        Predict,
        ChainOfThought,
        MultiChainComparison,
        ReAct,
        RLM,
        Flex,
    )
    return {(cls.__module__, cls.__qualname__): cls for cls in classes}


def _serialize_builtin_module(module: BaseModule) -> dict[str, Any] | None:
    from dspy.predict.chain_of_thought import ChainOfThought
    from dspy.predict.flex.flex import Flex
    from dspy.predict.multi_chain_comparison import MultiChainComparison
    from dspy.predict.predict import Predict
    from dspy.predict.react import ReAct
    from dspy.predict.rlm import RLM

    cls = type(module)

    if cls is Predict:
        assert isinstance(module, Predict)
        sig = _require_signature(module.signature, context="Predict")
        lm_config = _dump_predictor_config(module.config, context="Predict.config")
        state = _ensure_json_dict(module.dump_state(json_mode=True), context="Predict state")
        return {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "config": {
                "signature": serialize_signature(sig),
                "lm_config": lm_config,
            },
            "state": state,
            "compiled": bool(getattr(module, "_compiled", False)),
        }

    if cls is ChainOfThought:
        assert isinstance(module, ChainOfThought)
        cot_sig = _require_signature(module.predict.signature, context="ChainOfThought.predict")
        lm_config = _dump_predictor_config(module.predict.config, context="ChainOfThought.predict.config")
        state = _ensure_json_dict(module.dump_state(json_mode=True), context="ChainOfThought state")
        return {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "config": {
                "signature": serialize_signature(cot_sig),
                "lm_config": lm_config,
            },
            "state": state,
            "compiled": bool(getattr(module, "_compiled", False)),
        }

    if cls is MultiChainComparison:
        assert isinstance(module, MultiChainComparison)
        if not _is_strict_int(module.M) or module.M < 1:
            raise ValueError(f"MultiChainComparison.M must be a positive int, got {module.M!r}.")
        predict_sig = _require_signature(module.predict.signature, context="MultiChainComparison.predict")
        base_sig = predict_sig
        for idx in range(module.M):
            base_sig = base_sig.delete(f"reasoning_attempt_{idx + 1}")
        base_sig = base_sig.delete("rationale")
        lm_config = _dump_predictor_config(module.predict.config, context="MultiChainComparison.predict.config")
        state = _ensure_json_dict(module.dump_state(json_mode=True), context="MultiChainComparison state")
        return {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "config": {
                "signature": serialize_signature(base_sig),
                "predict_signature": serialize_signature(predict_sig),
                "M": module.M,
                "last_key": module.last_key,
                "lm_config": lm_config,
            },
            "state": state,
            "compiled": bool(getattr(module, "_compiled", False)),
        }

    if cls is ReAct:
        assert isinstance(module, ReAct)
        if not _is_strict_int(module.max_iters) or module.max_iters < 1:
            raise ValueError(f"ReAct.max_iters must be a positive int, got {module.max_iters!r}.")
        react_sig = _require_signature(module.signature, context="ReAct")
        tool_names = [name for name in module.tools if name != "submit"]
        react_config = _dump_predictor_config(module.react.config, context="ReAct.react.config")
        state = _ensure_json_dict(module.dump_state(json_mode=True), context="ReAct state")
        return {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "config": {
                "signature": serialize_signature(react_sig),
                "max_iters": module.max_iters,
                "tool_names": tool_names,
                "react_config": react_config,
            },
            "state": state,
            "compiled": bool(getattr(module, "_compiled", False)),
        }

    if cls is RLM:
        assert isinstance(module, RLM)
        if module._interpreter_factory is not PythonInterpreter:
            raise ValueError(
                "RLM with a custom `interpreter_factory` cannot be serialized in JSON format. "
                "Use the default `PythonInterpreter` and configure custom interpreters at runtime."
            )
        if module.history_processor is not None:
            raise ValueError(
                "RLM with a custom `history_processor` callable cannot be serialized in JSON format. "
                "Pass `history_processor` at call time instead."
            )
        sub_lm_state: dict[str, Any] | None = None
        if module.sub_lm is not None:
            if not isinstance(module.sub_lm, BaseLM):
                raise ValueError(
                    "RLM `sub_lm` must be a `dspy.BaseLM` instance or `None` for JSON program serialization."
                )
            sub_lm_state = _ensure_json_dict(module.sub_lm.dump_state(), context="RLM.sub_lm state")

        rlm_sig = _require_signature(module.signature, context="RLM")
        generate_action_config = _dump_predictor_config(
            module.generate_action.config, context="RLM.generate_action.config"
        )
        extract_config = _dump_predictor_config(module.extract.config, context="RLM.extract.config")
        state = _ensure_json_dict(module.dump_state(json_mode=True), context="RLM state")
        return {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "config": {
                "signature": serialize_signature(rlm_sig),
                "max_iters": module.max_iters,
                "max_llm_calls": module.max_llm_calls,
                "max_output_chars": module.max_output_chars,
                "verbose": bool(module.verbose),
                "tool_names": list(module._user_tools.keys()),
                "sub_lm": sub_lm_state,
                "generate_action_config": generate_action_config,
                "extract_config": extract_config,
            },
            "state": state,
            "compiled": bool(getattr(module, "_compiled", False)),
        }

    if cls is Flex:
        assert isinstance(module, Flex)
        if module._interpreter_factory is not PythonInterpreter:
            raise ValueError(
                "Flex with a custom `interpreter_factory` cannot be serialized in JSON format. "
                "Use the default `PythonInterpreter` and configure custom interpreters at runtime."
            )
        tool_names = list(module._flex_ctx.context_names().keys())
        state = _ensure_json_dict(module.dump_state(json_mode=True), context="Flex state")
        return {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "config": {
                "signature": serialize_signature(module.signature),
                "tool_names": tool_names,
                "max_predictor_calls": module._max_predictor_calls,
            },
            "state": state,
            "compiled": bool(getattr(module, "_compiled", False)),
        }

    return None


def _is_module_container(value: Any) -> bool:
    if isinstance(value, (BaseModule, Parameter)):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(isinstance(item, (BaseModule, Parameter)) for item in value)
    if isinstance(value, dict):
        return len(value) > 0 and all(
            isinstance(k, str) and isinstance(v, (BaseModule, Parameter)) for k, v in value.items()
        )
    return False


def _supports_zero_arg_init(cls: type[BaseModule]) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
        sig.bind(object())
        return True
    except (TypeError, ValueError):
        return False


def _all_named_parameters(module: BaseModule) -> list[tuple[str, Parameter]]:
    """Like ``named_parameters()``, but includes the parameters of compiled sub-modules."""
    from dspy.primitives.module import Module

    found: list[tuple[str, Parameter]] = []
    seen: set[int] = set()

    def add(name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if id(value) not in seen:
                seen.add(id(value))
                found.append((name, value))
        elif isinstance(value, Module):
            walk(f"{name}.", value)

    def walk(prefix: str, owner: BaseModule) -> None:
        for attr, value in owner.__dict__.items():
            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    add(f"{prefix}{attr}[{idx}]", item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    add(f"{prefix}{attr}['{key}']", item)
            else:
                add(f"{prefix}{attr}", value)

    if isinstance(module, Parameter):
        add("self", module)
    walk("", module)
    return found


def _serialize_parameter(name: str, param: Parameter) -> dict[str, Any]:
    from dspy.predict.predict import Predict

    entry: dict[str, Any] = {"state": _ensure_json_dict(param.dump_state(json_mode=True), context=f"'{name}' state")}
    if isinstance(param, Predict):
        signature = _require_signature(param.signature, context=f"Predictor '{name}'")
        entry["signature"] = serialize_signature(signature)
        entry["config"] = _dump_predictor_config(param.config, context=f"Predictor '{name}' config")
    return entry


def _overrides_state_methods(cls: type[BaseModule]) -> bool:
    return cls.dump_state is not BaseModule.dump_state or cls.load_state is not BaseModule.load_state


def _compiled_sub_modules(module: BaseModule) -> list[str]:
    return [name for name, sub in module.named_sub_modules() if sub is not module and getattr(sub, "_compiled", False)]


def _serialize_custom_module(module: BaseModule) -> dict[str, Any]:
    from dspy.primitives.module import Module

    cls = type(module)
    if not isinstance(module, Module) or cls is Module:
        raise ValueError(
            f"Unsupported module type '{cls.__module__}.{cls.__qualname__}' for JSON program serialization."
        )

    has_to_config = callable(getattr(cls, "to_program_config", None))
    has_from_config = callable(getattr(cls, "from_program_config", None))
    if has_to_config != has_from_config:
        raise ValueError(
            f"Custom module '{cls.__qualname__}' must implement both `to_program_config()` and "
            "`from_program_config(config)` for JSON program serialization."
        )

    if has_to_config and has_from_config:
        raw_config = module.to_program_config()  # type: ignore[attr-defined]
        config = _ensure_json_dict(raw_config, context=f"{cls.__qualname__}.to_program_config()")
        return _custom_module_payload(module, construction_mode="hooks", config=config)

    if not _supports_zero_arg_init(cls):
        raise ValueError(
            f"Module '{cls.__module__}.{cls.__qualname__}' requires constructor arguments and does not implement "
            "`to_program_config()` / `from_program_config(config)` for JSON program serialization."
        )

    extra_attrs = sorted(
        attr
        for attr, value in module.__dict__.items()
        if attr not in _BASE_MODULE_RUNTIME_ATTRS and not _is_module_container(value)
    )
    if extra_attrs:
        raise ValueError(
            f"Module '{cls.__module__}.{cls.__qualname__}' has non-module attribute(s) {extra_attrs} "
            "that cannot be preserved by zero-argument reconstruction. Implement `to_program_config()` "
            "and `from_program_config(config)` to serialize this module declaratively in JSON format."
        )

    return _custom_module_payload(module, construction_mode="zero_arg", config={})


def _custom_module_payload(module: BaseModule, *, construction_mode: str, config: dict[str, Any]) -> dict[str, Any]:
    cls = type(module)
    payload = {
        "module": cls.__module__,
        "qualname": cls.__qualname__,
        "construction_mode": construction_mode,
        "config": config,
        "parameters": {name: _serialize_parameter(name, param) for name, param in _all_named_parameters(module)},
        "compiled_sub_modules": _compiled_sub_modules(module),
        "compiled": bool(getattr(module, "_compiled", False)),
    }
    if _overrides_state_methods(cls):
        payload["state"] = _ensure_json_dict(module.dump_state(json_mode=True), context=f"{cls.__qualname__} state")
    return payload


def serialize_program(module: BaseModule) -> dict[str, Any]:
    """Serialize a DSPy program to a JSON-compatible declarative dictionary."""
    module_payload = _serialize_builtin_module(module)
    if module_payload is None:
        module_payload = _serialize_custom_module(module)

    payload = {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "module": module_payload,
    }
    # Verify final payload round-trips cleanly through orjson before touching disk.
    orjson.dumps(payload)
    return payload


def _call_from_program_config(
    cls: type[Module],
    config: dict[str, Any],
    available_tools: dict[str, Tool],
) -> Module:
    from_config = getattr(cls, "from_program_config", None)
    if not callable(from_config):
        raise ValueError(f"Custom module '{cls.__module__}.{cls.__qualname__}' does not define `from_program_config`.")

    sig = inspect.signature(from_config)
    if "tools" in sig.parameters:
        result = from_config(config, tools=available_tools)
    else:
        result = from_config(config)

    if not isinstance(result, cls):
        raise TypeError(
            f"`{cls.__qualname__}.from_program_config` must return an instance of {cls.__qualname__}, "
            f"got {type(result).__name__}."
        )
    return result


def _deserialize_builtin_module(
    cls: type[Module],
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    allow_unsafe_lm_state: bool,
    available_tools: dict[str, Tool],
) -> Module:
    from dspy.predict.chain_of_thought import ChainOfThought
    from dspy.predict.flex.flex import Flex
    from dspy.predict.multi_chain_comparison import MultiChainComparison
    from dspy.predict.predict import Predict
    from dspy.predict.react import ReAct
    from dspy.predict.rlm import RLM

    if cls is Predict:
        signature = deserialize_signature(config.get("signature"))
        lm_config = _load_predictor_config(
            config.get("lm_config", {}),
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            context="Predict lm_config",
        )
        predict_state = _validate_predict_state(state, context="Predict")
        instance = Predict(signature, **lm_config)
        instance.load_state(predict_state, allow_unsafe_lm_state=allow_unsafe_lm_state)
        return instance

    if cls is ChainOfThought:
        extended_sig = deserialize_signature(config.get("signature"))
        lm_config = _load_predictor_config(
            config.get("lm_config", {}),
            forbidden_keys=frozenset({"signature", "rationale_field", "rationale_field_type", "callbacks"}),
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            context="ChainOfThought lm_config",
        )
        if set(state) != {"predict"}:
            raise ValueError("ChainOfThought state must contain exactly the 'predict' key.")
        _validate_predict_state(state["predict"], context="ChainOfThought.predict")
        base_sig = extended_sig.delete("reasoning") if "reasoning" in extended_sig.output_fields else extended_sig
        cot = ChainOfThought(base_sig, **lm_config)
        cot.predict.signature = extended_sig
        cot.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
        return cot

    if cls is MultiChainComparison:
        base_sig = deserialize_signature(config.get("signature"))
        m_val = config.get("M")
        if not _is_strict_int(m_val) or m_val < 1:
            raise ValueError(f"MultiChainComparison 'M' must be a positive int, got {m_val!r}.")
        last_key = config.get("last_key")
        if not isinstance(last_key, str) or last_key not in base_sig.output_fields:
            raise ValueError(f"MultiChainComparison 'last_key' must name an output field, got {last_key!r}.")
        lm_config = _load_predictor_config(
            config.get("lm_config", {}),
            forbidden_keys=frozenset({"signature", "M", "callbacks"}),
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            context="MultiChainComparison lm_config",
        )
        temperature = lm_config.pop("temperature", 0.7)
        if set(state) != {"predict"}:
            raise ValueError("MultiChainComparison state must contain exactly the 'predict' key.")
        _validate_predict_state(state["predict"], context="MultiChainComparison.predict")
        mcc = MultiChainComparison(base_sig, M=m_val, temperature=temperature, **lm_config)
        mcc.last_key = last_key
        if "predict_signature" in config:
            mcc.predict.signature = deserialize_signature(config["predict_signature"])
        mcc.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
        return mcc

    if cls is ReAct:
        signature = deserialize_signature(config.get("signature"))
        max_iters = config.get("max_iters", 20)
        if not _is_strict_int(max_iters) or max_iters < 1:
            raise ValueError(f"ReAct 'max_iters' must be a positive int, got {max_iters!r}.")
        bound_tools = _bind_required_tools(
            config.get("tool_names"),
            available_tools,
            module_label="ReAct",
            disallowed_names=frozenset({"submit"}),
        )
        react_config = _load_predictor_config(
            config.get("react_config", {}),
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            context="ReAct react_config",
        )
        if set(state) != {"react"}:
            raise ValueError("ReAct state must contain exactly the 'react' key.")
        _validate_predict_state(state["react"], context="ReAct.react")
        react_tools: list[Callable[..., Any] | Tool] = list(bound_tools)
        react = ReAct(signature, tools=react_tools, max_iters=max_iters)
        if react_config:
            react.react.config = react_config
        react.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
        return react

    if cls is RLM:
        signature = deserialize_signature(config.get("signature"))
        max_iters = config.get("max_iters", 20)
        max_llm_calls = config.get("max_llm_calls", 50)
        max_output_chars = config.get("max_output_chars", 10_000)
        verbose = config.get("verbose", False)
        if not _is_strict_int(max_iters) or max_iters < 1:
            raise ValueError(f"RLM 'max_iters' must be a positive int, got {max_iters!r}.")
        if not _is_strict_int(max_llm_calls) or max_llm_calls < 1:
            raise ValueError(f"RLM 'max_llm_calls' must be a positive int, got {max_llm_calls!r}.")
        if not _is_strict_int(max_output_chars) or max_output_chars < 1:
            raise ValueError(f"RLM 'max_output_chars' must be a positive int, got {max_output_chars!r}.")
        if not isinstance(verbose, bool):
            raise ValueError(f"RLM 'verbose' must be a bool, got {verbose!r}.")

        bound_tools = _bind_required_tools(
            config.get("tool_names", []),
            available_tools,
            module_label="RLM",
        )
        raw_sub_lm = config.get("sub_lm")
        sub_lm = None
        if raw_sub_lm is not None:
            if not isinstance(raw_sub_lm, dict):
                raise ValueError("RLM 'sub_lm' must be a dict or None.")
            sanitized_sub_lm = _sanitize_lm_state(raw_sub_lm, allow_unsafe_lm_state)
            sub_lm = BaseLM.load_state(sanitized_sub_lm, allow_custom_lm_class=allow_unsafe_lm_state)

        generate_action_config = _load_predictor_config(
            config.get("generate_action_config", {}),
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            context="RLM generate_action_config",
        )
        extract_config = _load_predictor_config(
            config.get("extract_config", {}),
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            context="RLM extract_config",
        )
        if set(state) != {"generate_action", "extract"}:
            raise ValueError("RLM state must contain exactly 'generate_action' and 'extract' keys.")
        _validate_predict_state(state["generate_action"], context="RLM.generate_action")
        _validate_predict_state(state["extract"], context="RLM.extract")

        rlm = RLM(
            signature,
            max_iters=max_iters,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=cast("list[Callable[..., Any]] | None", bound_tools or None),
            sub_lm=cast("Any", sub_lm),
        )
        if generate_action_config:
            rlm.generate_action.config = generate_action_config
        if extract_config:
            rlm.extract.config = extract_config
        rlm.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
        return rlm

    if cls is Flex:
        signature = deserialize_signature(config.get("signature"))
        max_predictor_calls = config.get("max_predictor_calls", 100)
        if max_predictor_calls is not None and (not _is_strict_int(max_predictor_calls) or max_predictor_calls < 0):
            raise ValueError(
                f"Flex 'max_predictor_calls' must be a non-negative int or None, got {max_predictor_calls!r}."
            )
        bound_tools = _bind_required_tools(
            config.get("tool_names", []),
            available_tools,
            module_label="Flex",
        )
        unexpected_flex_keys = sorted(set(state) - _FLEX_STATE_KEYS)
        if unexpected_flex_keys:
            raise ValueError(f"Flex state contains unexpected key(s): {', '.join(unexpected_flex_keys)}.")
        module_src = state.get("module_src")
        if module_src is not None and not isinstance(module_src, str):
            raise ValueError("Flex state 'module_src' must be a string or None.")
        if state.get("lm") is not None and not isinstance(state.get("lm"), dict):
            raise ValueError("Flex state 'lm' must be a dict or None.")

        flex = Flex(
            signature,
            tools=bound_tools or None,
            max_predictor_calls=max_predictor_calls,
        )
        flex.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
        return flex

    raise ValueError(f"Unsupported built-in module class: '{cls.__module__}.{cls.__qualname__}'.")


def deserialize_program(
    data: Any,
    *,
    allow_unsafe_lm_state: bool = False,
    safe_modules: Sequence[type[Module]] | None = None,
    tools: Mapping[str, Callable[..., Any] | Tool] | Sequence[Callable[..., Any] | Tool] | None = None,
) -> Module:
    """Reconstruct a DSPy program safely from parsed ``program.json`` data."""
    from dspy.primitives.module import Module

    if not isinstance(data, dict):
        raise ValueError(f"Invalid program.json: expected a JSON object, got {type(data).__name__}.")

    schema_version = data.get("schema_version")
    if not _is_strict_int(schema_version) or schema_version != PROGRAM_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported program.json schema_version: {schema_version!r}. Expected {PROGRAM_SCHEMA_VERSION}."
        )

    module_data = data.get("module")
    if not isinstance(module_data, dict):
        raise ValueError("Invalid program.json: 'module' must be a JSON object.")

    module_name = module_data.get("module")
    qualname = module_data.get("qualname")
    if not isinstance(module_name, str) or not module_name or not isinstance(qualname, str) or not qualname:
        raise ValueError("Invalid program.json: 'module' and 'qualname' must be non-empty strings.")

    config = module_data.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Invalid program.json: 'config' must be a JSON object.")

    compiled = module_data.get("compiled", False)
    if not isinstance(compiled, bool):
        raise ValueError("Invalid program.json: 'compiled' must be a bool.")

    available_tools = _normalize_caller_tools(tools)
    builtins = _builtin_registry()
    class_key = (module_name, qualname)

    if class_key in builtins:
        state = module_data.get("state")
        if not isinstance(state, dict):
            raise ValueError("Invalid program.json: 'state' must be a JSON object.")
        instance = _deserialize_builtin_module(
            builtins[class_key],
            config,
            state,
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            available_tools=available_tools,
        )
        instance._compiled = compiled
        return instance

    custom_allowlist: dict[tuple[str, str], type[Module]] = {}
    if safe_modules is not None:
        for mod_cls in safe_modules:
            if not isinstance(mod_cls, type) or not issubclass(mod_cls, Module):
                raise TypeError(f"Entries in `safe_modules` must be `dspy.Module` subclasses, got {mod_cls!r}.")
            custom_allowlist[(mod_cls.__module__, mod_cls.__qualname__)] = mod_cls

    if class_key not in custom_allowlist:
        raise ValueError(
            f"Module class '{module_name}.{qualname}' is not in the built-in registry or `safe_modules` allowlist. "
            f"Pass `safe_modules=[{qualname}]` to `dspy.load()` only if you trust this file."
        )

    custom_cls = custom_allowlist[class_key]
    mode = module_data.get("construction_mode", "zero_arg")
    if mode == "hooks":
        instance = _call_from_program_config(custom_cls, config, available_tools)
    elif mode == "zero_arg":
        instance = custom_cls()
    else:
        raise ValueError(f"Unsupported custom module construction_mode: {mode!r}.")

    _restore_custom_module(instance, module_data, allow_unsafe_lm_state=allow_unsafe_lm_state)
    instance._compiled = compiled
    return instance


def _restore_custom_module(instance: Module, module_data: dict[str, Any], *, allow_unsafe_lm_state: bool) -> None:
    from dspy.predict.predict import Predict

    saved = module_data.get("parameters")
    if not isinstance(saved, dict):
        raise ValueError("Invalid program.json: custom module 'parameters' must be a JSON object.")
    params = dict(_all_named_parameters(instance))
    if set(saved) != set(params):
        missing, unexpected = sorted(set(params) - set(saved)), sorted(set(saved) - set(params))
        raise ValueError(
            f"Saved parameters do not match the structure built by {type(instance).__qualname__}: "
            f"missing {missing}, unexpected {unexpected}. The class may have changed since the program was saved."
        )

    compiled_names = module_data.get("compiled_sub_modules", [])
    if not isinstance(compiled_names, list) or not all(isinstance(name, str) for name in compiled_names):
        raise ValueError("Invalid program.json: 'compiled_sub_modules' must be a list of strings.")
    sub_modules = dict(instance.named_sub_modules())
    unknown = sorted(set(compiled_names) - set(sub_modules))
    if unknown:
        raise ValueError(f"Invalid program.json: unknown compiled sub-module(s): {', '.join(unknown)}.")

    for name, param in params.items():
        entry = saved[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("state"), dict):
            raise ValueError(f"Invalid program.json: parameter '{name}' must be an object with a 'state' object.")
        if isinstance(param, Predict):
            _validate_predict_state(entry["state"], context=f"Predictor '{name}'")
            param.signature = deserialize_signature(entry.get("signature"))
            param.config = _load_predictor_config(
                entry.get("config"), allow_unsafe_lm_state=allow_unsafe_lm_state, context=f"Predictor '{name}' config"
            )
        param.load_state(entry["state"], allow_unsafe_lm_state=allow_unsafe_lm_state)
    for name in compiled_names:
        sub_modules[name]._compiled = True

    if "state" in module_data:
        state = module_data["state"]
        if not _overrides_state_methods(type(instance)):
            raise ValueError(
                "Invalid program.json: 'state' is only saved for modules that override their state methods."
            )
        if not isinstance(state, dict):
            raise ValueError("Invalid program.json: 'state' must be a JSON object.")
        for name, param in params.items():
            if isinstance(param, Predict) and name in state:
                _validate_predict_state(state[name], context=f"Predictor '{name}'")
        instance.load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
