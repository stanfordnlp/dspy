"""Sandbox side of the dspy bridge: wires the real dspy inside a sandbox to its host.

``dspy.Flex`` modules and ``dspy.RLM`` REPL code import the real dspy in their interpreter. Once they do, the host
(``dspy.primitives.facade``) has this module ``install`` itself there. It routes to the host only what the sandbox
must not own, as JSON over the interpreter's tool-call protocol:

- LM calls. ``HostLM`` sends each request to the host through one host tool, ``LM_TOOL``. The host builds the
  request from its own LM, admits only the options sandboxed code may set, meters the call and runs it, so the
  model, credentials, cache, retries and call budget stay on the host.
- Predictor calls. ``HostAdapter`` has the host's adapter format each one and parse its answer, on the host, so the
  prompt and the parsing come from the real types, as they would without a sandbox.
- Provided tools. Each is bound as a ``HostTool``: a ``dspy.Tool`` with the host tool's name, description and
  argument schemas, called through ``CALL_TOOL``.
- The user's custom types. The host owns them: a value of one is plain data here, the JSON its type writes, and the
  host reads it back as the type. Code here can still name the type, as in a signature (``HostType``).

Values cross both ways in one JSON form that carries their types (``Wire``). Code-executing modules built in the
sandbox, like ``dspy.RLM``, each get a ``HostInterpreter``: a fresh interpreter the host creates from its factory,
outside this sandbox.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import dataclasses
import datetime
import decimal
import enum
import functools
import importlib
import inspect
import json
import math
import secrets
import sys
import uuid
from typing import Any, ClassVar, Literal, Union

import pydantic
import pydantic_core
from pydantic_core import core_schema

import dspy
from dspy._vendor.lm15 import serde
from dspy.adapters.base import Prompt
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value
from dspy.clients.base_lm import BaseLM
from dspy.lm15 import BuiltinTool, CacheConfig, FunctionTool, Request
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.primitives.sandbox_serializable import SandboxSerializable
from dspy.utils import exceptions

# The host tool every LM call goes through, and the key of an LM error it reports.
LM_TOOL = "__dspy_lm__"
LM_ERROR_KEY = "__dspy_lm_error__"
# The host tool each predictor call recorded in the sandbox's trace is reported to.
TRACE_TOOL = "__dspy_trace__"
# The host tool provided tools are called through.
CALL_TOOL = "__dspy_tool_call__"


# -- wire format, shared with the host ----------------------------------------------------------------------------

# The key that marks a JSON object as a tagged value rather than a dict.
TAG = "__dspy__"
# The name a SandboxSerializable's assignment writes its value to (``Wire.encode_codec``).
_VALUE = "__dspy_codec_value"
# The tag of a value a host function RLM gives the interpreter (SUBMIT) gets in the session's wire form (``bind``).
WIRE = "wire"
_COLLECTIONS = {"tuple": tuple, "set": set, "frozenset": frozenset}


class Wire:
    """The JSON form values cross the sandbox boundary in, read back the same way on both sides.

    A value carries its own types, so neither side needs the other's annotations to read it. JSON's own values cross
    as they are. A tuple, set, bytes, date or time, ``Decimal``, ``UUID`` or non-finite float is an object tagged
    with what it is, and so is a dict whose keys aren't all strings. An object of a class both sides have (dspy's,
    the standard library's, pydantic's) crosses as that class, by where it is imported from, and its fields, and an
    enum member as its enum and value. A ``dspy.Tool`` crosses as what describes it, without its function. Anything
    else crosses as pydantic writes it in JSON, or as its text: that is how a value of the user's own types reaches
    the sandbox (``facade.HostValues``).

    An object the host sends carries a reference to it, which goes back with it, so the host can use its own object
    when it comes back unchanged.
    """

    def encode(self, value: Any) -> Any:
        if isinstance(value, enum.Enum):  # before int and str, which some enums are
            return self.encode_enum(value)
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {TAG: "float", "value": repr(value)}
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, dict):
            if TAG not in value and all(type(key) is str for key in value):
                return {key: self.encode(item) for key, item in value.items()}
            return {TAG: "dict", "items": [[self.encode(key), self.encode(item)] for key, item in value.items()]}
        if isinstance(value, Tool):
            return self.encode_tool(value)
        if isinstance(value, SandboxSerializable):  # its own codec, before any other form its class has
            return self.encode_codec(value)
        if (fields := _fields_of(value)) is not None:
            return self.encode_object(value, fields)
        if isinstance(value, (tuple, set, frozenset)):
            kind = next(name for name, kind in _COLLECTIONS.items() if isinstance(value, kind))
            return {TAG: kind, "items": [self.encode(item) for item in value]}
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {TAG: "bytes", "base64": base64.b64encode(bytes(value)).decode()}
        for kind in (datetime.datetime, datetime.date, datetime.time):  # a datetime is a date too
            if isinstance(value, kind):
                return {TAG: kind.__name__, "iso": value.isoformat()}
        if isinstance(value, datetime.timedelta):
            return {TAG: "timedelta", "days": value.days, "seconds": value.seconds, "microseconds": value.microseconds}
        if isinstance(value, (decimal.Decimal, uuid.UUID)):
            return {TAG: "decimal" if isinstance(value, decimal.Decimal) else "uuid", "value": str(value)}
        return self.encode(pydantic_core.to_jsonable_python(value, by_alias=False, round_trip=True, fallback=str))

    def decode(self, data: Any) -> Any:
        if isinstance(data, list):
            return [self.decode(item) for item in data]
        if not isinstance(data, dict):
            return data
        tag = data.get(TAG)
        if tag is None:
            return {key: self.decode(item) for key, item in data.items()}
        if tag in _COLLECTIONS:
            items = [self.decode(item) for item in data["items"]]
            with contextlib.suppress(TypeError):  # a set of values that aren't hashable here stays a list
                return _COLLECTIONS[tag](items)
            return items
        if tag == "dict":
            out = {}
            for key, item in data["items"]:
                value = self.decode(item)
                try:
                    out[self.decode(key)] = value
                except TypeError:  # a key that isn't hashable here, as its JSON
                    out[json.dumps(key)] = value
            return out
        if tag == "bytes":
            return base64.b64decode(data["base64"])
        if tag in ("datetime", "date", "time"):
            return getattr(datetime, tag).fromisoformat(data["iso"])
        if tag == "timedelta":
            return datetime.timedelta(days=data["days"], seconds=data["seconds"], microseconds=data["microseconds"])
        if tag in ("decimal", "uuid", "float"):
            return {"decimal": decimal.Decimal, "uuid": uuid.UUID, "float": float}[tag](data["value"])
        if tag == "enum":
            value, owner = self.decode(data["value"]), self.find_class(data)
            if isinstance(owner, type) and issubclass(owner, enum.Enum):
                with contextlib.suppress(ValueError):
                    return owner(value)
            return value
        if tag == "object":
            return self.decode_object(data)
        if tag == "tool":
            return self.decode_tool(data)
        if tag == "ref":
            return self.decode_ref(data)
        if tag == "codec":
            return self.decode_codec(data)
        raise ValueError(f"Unknown value tag {tag!r} on the wire.")

    # How each side names the classes it sends and finds those it receives.

    def name_class(self, cls: type) -> dict[str, Any]:
        return {"import": f"{cls.__module__}.{cls.__qualname__}"}

    def find_class(self, node: dict[str, Any]) -> Any:
        with contextlib.suppress(Exception):
            return _import(node["import"])
        return None

    def encode_enum(self, member: enum.Enum) -> Any:
        return {TAG: "enum", **self.name_class(type(member)), "value": self.encode(member.value)}

    def encode_object(self, value: Any, fields: dict[str, Any]) -> dict[str, Any]:
        return {TAG: "object", **self.name_class(type(value)),
                "fields": {name: self.encode(item) for name, item in fields.items()}}

    def decode_object(self, node: dict[str, Any]) -> Any:
        """An object of a class this side has, read by field name, or its fields if not."""
        owner, fields = self.find_class(node), self.decode(node.get("fields", {}))
        if not isinstance(owner, type):
            return fields
        try:
            return pydantic.TypeAdapter(owner).validate_python(fields, by_name=True)
        except Exception:
            return fields

    def encode_tool(self, tool: Tool) -> dict[str, Any]:
        return {TAG: "tool", "name": tool.name, "desc": tool.desc, "args": self.encode(tool.args),
                "arg_desc": self.encode(tool.arg_desc), "has_kwargs": tool.has_kwargs}

    def decode_tool(self, node: dict[str, Any]) -> Tool:
        """A tool from the other side, which describes it: its function stays there."""
        name = node["name"]

        def elsewhere(**kwargs: Any) -> Any:
            raise RuntimeError(f"The tool {name!r} runs on the other side of the sandbox boundary.")

        args = self.decode(node.get("args")) or {}
        tool = Tool(elsewhere, name=name, desc=node.get("desc"), args=args, arg_types=dict.fromkeys(args, Any),
                    arg_desc=self.decode(node.get("arg_desc")))
        tool.has_kwargs = bool(node.get("has_kwargs"))
        return tool

    def decode_ref(self, node: dict[str, Any]) -> Any:
        raise ValueError("Only the host reads references to its objects.")

    def encode_codec(self, value: SandboxSerializable) -> dict[str, Any]:
        """A ``SandboxSerializable`` as its own hooks write it: its payload (text if it is UTF-8, else base64), the
        setup and the assignment that rebuild it in the sandbox (``decode_codec``). The payload gets a name of its
        own, which stays bound for code the assignment defines to read later (say, a lambda)."""
        name = f"__dspy_codec_data_{secrets.token_hex(8)}"
        payload = value.to_sandbox()
        if isinstance(payload, bytes):
            try:
                data = {"text": payload.decode("utf-8")}
            except UnicodeDecodeError:
                data = {"base64": base64.b64encode(payload).decode("ascii")}
        else:
            data = {"text": str(payload)}
        assign = value.sandbox_assignment(_VALUE, name)
        return {TAG: "codec", "setup": value.sandbox_setup(), "assign": assign, "name": name, **data}

    def decode_codec(self, node: dict[str, Any]) -> Any:
        raise ValueError("Only the sandbox rebuilds a SandboxSerializable value.")


def read_wire(value: Any, session: str, wire: Wire) -> Any:
    """A value a host function got from the interpreter: read back from the wire form if ``bind`` sent it so for
    ``session``, else the interpreter's plain JSON, as is."""
    if isinstance(value, dict) and value.keys() == {TAG, "session", "value"} and value[TAG] == WIRE:
        if value["session"] == session:
            return wire.decode(value["value"])
    return value


def _rebuild(node: dict[str, Any], namespace: dict[str, Any]) -> Any:
    """Run a ``SandboxSerializable``'s setup and assignment (``Wire.encode_codec``) in ``namespace``; its value."""
    if node.get("setup"):
        exec(node["setup"], namespace)
    namespace[node["name"]] = node["text"] if "text" in node else base64.b64decode(node["base64"])
    exec(node["assign"], namespace)
    return namespace.pop(_VALUE)


def _fields_of(value: Any) -> dict[str, Any] | None:
    """The fields ``value`` crosses as, if it is an object of a class: a pydantic model's or dataclass's but those marked
    ``exclude=True``, which stay where they are, or a ``NamedTuple``'s."""
    if isinstance(value, type):
        return None
    if isinstance(value, pydantic.BaseModel):
        fields = {name: getattr(value, name) for name, info in type(value).model_fields.items() if info.exclude is not True}
        return {**fields, **(value.__pydantic_extra__ or {})}
    if dataclasses.is_dataclass(value):
        infos = getattr(type(value), "__pydantic_fields__", None) or {}
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)
                if hasattr(value, f.name) and getattr(infos.get(f.name), "exclude", None) is not True}
    if isinstance(value, tuple) and hasattr(type(value), "_fields"):
        return dict(zip(type(value)._fields, value, strict=False))
    return None


def _import(path: str) -> Any:
    """What ``path`` names: an attribute, maybe nested, of the longest importable module it starts with."""
    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            found = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for name in parts[split:]:
            found = getattr(found, name)
        return found
    raise ImportError(f"Nothing to import at {path!r}")


def _jsonable(value: Any) -> Any:
    return pydantic_core.to_jsonable_python(value, by_alias=False, round_trip=True, fallback=str)


def encode_prompt(prompt: Any) -> Any:
    if isinstance(prompt, Request):
        return {"request": serde.request_to_dict(prompt)}
    if isinstance(prompt, Prompt):
        return {"prompt": serde.request_to_dict(Request(model="-", system=prompt.system, messages=prompt.messages))}
    return prompt


def decode_prompt(payload: Any) -> Any:
    if isinstance(payload, dict) and "request" in payload:
        return serde.request_from_dict(payload["request"])
    if isinstance(payload, dict) and "prompt" in payload:
        request = serde.request_from_dict(payload["prompt"])
        return Prompt(messages=request.messages, system=request.system)
    return payload


def encode_options(options: dict[str, Any]) -> dict[str, Any]:
    from dspy.clients.requests import close_object_schemas

    out = {}
    for key, value in options.items():
        if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
            # How build_request sends a pydantic response_format.
            value = {"type": "json_schema", "json_schema": {
                "name": value.__name__, "schema": close_object_schemas(value.model_json_schema()), "strict": True,
            }}
        elif isinstance(value, CacheConfig):
            value = {"__lm15_cache__": serde.cache_config_to_dict(value)}
        elif key == "tools" and isinstance(value, (list, tuple)):
            value = [{"__lm15_tool__": serde.tool_to_dict(t)} if isinstance(t, (FunctionTool, BuiltinTool)) else t for t in value]
        out[key] = value
    return out


def decode_options(options: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in options.items():
        if isinstance(value, dict) and "__lm15_cache__" in value:
            value = serde.cache_config_from_dict(value["__lm15_cache__"])
        elif key == "tools" and isinstance(value, list):
            value = [serde.tool_from_dict(t["__lm15_tool__"]) if isinstance(t, dict) and "__lm15_tool__" in t else t
                     for t in value]
        out[key] = value
    return out


# -- the user's custom types ----------------------------------------------------------------------------------------

# Typing names a signature's annotations may use; sandboxed code is told not to import at module scope.
TYPING_NAMES = {"Any": Any, "Literal": Literal, "Optional": __import__("typing").Optional, "Union": Union}


class HostType:
    """A custom type of the user's, as sandboxed code can name it, as in a signature. The type and its code stay on the
    host, which owns it: its values are plain data here, the JSON the type writes, which the host reads back as the
    type. Pydantic takes any value for it here, and describes it with the type's JSON schema, which the host sends."""

    __dspy_schema__: ClassVar[dict[str, Any]] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            f"{cls.__name__} is the host's type: in the sandbox its values are plain data, like a dict of its fields."
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> dict[str, Any]:
        return copy.deepcopy(cls.__dspy_schema__)


def _host_type(name: str, schema: dict[str, Any]) -> type:
    return type(name, (HostType,), {"__dspy_schema__": schema, "__module__": "__dspy_types"})


class GuestValues(Wire):
    """The sandbox's side of the wire. The reference each object from the host comes with is kept here, for when the
    object goes back (see ``Wire``). A ``SandboxSerializable`` value is rebuilt in ``namespace``, where its setup's
    names stay in scope for the code that uses it."""

    def __init__(self, namespace: dict[str, Any] | None = None) -> None:
        self._refs: dict[int, tuple[Any, int]] = {}  # objects from the host, by id: the object and its reference
        self._namespace = namespace if namespace is not None else {}

    def decode_codec(self, node: dict[str, Any]) -> Any:
        return _rebuild(node, self._namespace)

    def decode_object(self, node: dict[str, Any]) -> Any:
        value = super().decode_object(node)
        if isinstance(node.get("ref"), int) and _fields_of(value) is not None:
            self._refs[id(value)] = (value, node["ref"])
        return value

    def encode_object(self, value: Any, fields: dict[str, Any]) -> dict[str, Any]:
        node = super().encode_object(value, fields)
        kept = self._refs.get(id(value))
        return {**node, "ref": kept[1]} if kept is not None and kept[0] is value else node

    def encode_tool(self, tool: Tool) -> dict[str, Any]:
        # A provided tool goes back by name, as the host's own tool.
        return {TAG: "tool", "host": tool.name} if isinstance(tool, HostTool) else super().encode_tool(tool)


class InMemory(Wire):
    """The wire for code in the host's own memory, like an in-process interpreter's: nothing crosses, so values stay
    as they are, except that a ``SandboxSerializable`` from the host is rebuilt in ``namespace``, as in a sandbox."""

    def __init__(self, namespace: dict[str, Any] | None = None) -> None:
        self._namespace = namespace if namespace is not None else {}

    def encode(self, value: Any) -> Any:
        if isinstance(value, SandboxSerializable):
            return _rebuild(self.encode_codec(value), self._namespace)
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, dict):
            return {key: self.encode(item) for key, item in value.items()}
        return value

    def decode(self, data: Any) -> Any:
        return data


# The sandbox's side of the wire, set by ``install``.
_wire: Wire = Wire()


# -- the host LM ----------------------------------------------------------------------------------------------------


class HostLM(BaseLM):
    """The host's LM as sandboxed code sees it. Requests are built and run on the host."""

    def __init__(self, host: Any, spec: dict[str, Any]) -> None:
        super().__init__(model=spec["model"], model_type=spec["model_type"], cache=False, num_retries=0,
                         **spec["kwargs"])
        self._host = host
        self._defaults = dict(self.kwargs)
        # The cache setting a copy asked for. This LM's own `cache` stays off: the host caches, not the sandbox.
        self._cache: bool | None = None
        self._capabilities = spec["capabilities"]

    supports_function_calling = property(lambda self: self._capabilities["supports_function_calling"])
    supports_reasoning = property(lambda self: self._capabilities["supports_reasoning"])
    supports_response_schema = property(lambda self: self._capabilities["supports_response_schema"])
    supported_params = property(lambda self: set(self._capabilities["supported_params"]))
    supported_tool_result_media = property(lambda self: set(self._capabilities.get("supported_tool_result_media", ())))

    def _send(self, op: str, prompt: Any, options: dict[str, Any]) -> Any:
        return _raise_lm_error(self._host(op=op, prompt=encode_prompt(prompt), options=encode_options(options)))

    def _changed(self) -> dict[str, Any]:
        """Options set on this LM (``lm.copy(...)``), which override the host's defaults like call options do."""
        return {key: value for key, value in self.kwargs.items() if self._defaults.get(key) != value}

    def _build_adapter_request(self, prompt: Any, options: dict[str, Any] | None) -> Request:
        return serde.request_from_dict(self._send("request", prompt, {**self._changed(), **(options or {})}))

    def copy(self, **kwargs: Any) -> HostLM:
        new = super().copy(**{key: value for key, value in kwargs.items() if key != "cache"})
        new._cache = kwargs.get("cache", self._cache)
        return new

    def __call__(self, prompt: Any = None, **options: Any) -> Any:
        if prompt is None and "request" in options:
            prompt = options.pop("request")
        changed = self._changed()
        if isinstance(prompt, Request):  # it holds its generation settings, but not these, which belong to the call
            changed = {key: value for key, value in changed.items() if key == "rollout_id"}
        if self._cache is not None:
            changed["cache"] = self._cache
        out = self._send("call", prompt, {**changed, **options})
        return serde.response_from_dict(out) if isinstance(prompt, Request) else _wire.decode(out)

    async def acall(self, prompt: Any = None, **options: Any) -> Any:
        # The bridge is synchronous: the host's loop, if it has one, waits under this call. So the host runs its
        # LM's sync path, as for any call from the sandbox, rather than an async engine on a loop of its own.
        return self(prompt, **options)


def _raise_lm_error(out: Any) -> Any:
    if isinstance(out, dict) and LM_ERROR_KEY in out:
        error = out[LM_ERROR_KEY]
        raise getattr(exceptions, error["type"], exceptions.LMError)(error["message"])
    return out


class HostAdapter(ChatAdapter):
    """The host's adapter as sandboxed predictors see it: each call runs on the host, with the host's adapter, as
    it would without a sandbox. It formats the request with the real types, calls the host's LM, metered, and parses
    the answer there. Only the signature, the demos and the inputs cross, and the parsed outputs back. A predictor
    given an LM of the sandbox's own runs this adapter here instead."""

    _host: Any = None  # the LM_TOOL host tool, set by install()

    def __call__(self, lm: Any, lm_kwargs: dict[str, Any], signature: Any, demos: list[Any], inputs: dict[str, Any]):
        # The host's LM, maybe behind proxies that meter and check each call, like those of an RLM built here.
        chain = [lm]
        while not isinstance(chain[-1], HostLM) and getattr(chain[-1], "_lm", None) is not None:
            chain.append(chain[-1]._lm)
        if not isinstance(chain[-1], HostLM) or HostAdapter._host is None:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        for proxy in chain[:-1]:
            if callable(admit := getattr(proxy, "_admit", None)):
                admit(dict(lm_kwargs or {}))
        lm = chain[-1]
        options = {**lm._changed(), **(lm_kwargs or {})}
        if lm._cache is not None:
            options.setdefault("cache", lm._cache)
        call = {"signature": signature_spec(signature), "inputs": _wire.encode(dict(inputs)),
                "demos": [_wire.encode(dict(demo.items())) for demo in demos]}
        outputs = _wire.decode(_raise_lm_error(HostAdapter._host(op="adapter", prompt=call, options=encode_options(options))))
        return [{name: _read(value, signature.output_fields[name].annotation) if name in signature.output_fields
                 else value for name, value in completion.items()} for completion in outputs]

    async def acall(self, lm: Any, lm_kwargs: dict[str, Any], signature: Any, demos: list[Any], inputs: dict[str, Any]):
        return self(lm, lm_kwargs, signature, demos, inputs)  # on the host's sync path, as HostLM.acall


def _read(value: Any, annotation: Any) -> Any:
    """``value`` from the host, as its declared type here. The host formats a type it doesn't know, like one
    sandboxed code defined, as text, which is parsed here as an adapter would."""
    with contextlib.suppress(Exception):
        return pydantic.TypeAdapter(annotation).validate_python(value)
    if isinstance(value, str):
        with contextlib.suppress(Exception):
            return parse_value(value, annotation)
    return value


# -- host tools -----------------------------------------------------------------------------------------------------


class HostTool(Tool):
    """A provided tool as sandboxed code sees it: a ``dspy.Tool`` that calls the host tool by name, through
    ``call`` (the ``CALL_TOOL`` host tool), with values crossing on ``wire``."""

    def __init__(self, spec: dict[str, Any], call: Any, wire: Wire) -> None:
        name = spec["name"]

        def func(**kwargs: Any) -> Any:
            return wire.decode(call(tool=name, arguments=wire.encode(kwargs)))

        func.__name__ = func.__qualname__ = name
        func.__dspy_host_tool__ = name  # kept by wrappers, so a nested interpreter calls the host tool directly
        super().__init__(func, name=name, desc=spec["desc"], args=spec["args"], arg_types=dict.fromkeys(spec["args"], Any))
        self.has_kwargs = spec["has_kwargs"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Generated code may call a tool positionally, in the order the host declares its arguments.
        return super().__call__(**dict(zip(self.args, args, strict=False)), **kwargs)


# -- interpreters for code-executing modules built in the sandbox ----------------------------------------------------

# The host tool that gives each code-executing module built in the sandbox an interpreter of its own.
INTERPRETER_TOOL = "__dspy_interpreter__"
_SIMPLE_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}


def _parameters(fn: Any) -> list[dict[str, Any]]:
    """``fn``'s parameters as JSON, so the host can give its stand-in for ``fn`` the same signature."""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return [{"name": "args", "kind": "VAR_POSITIONAL"}, {"name": "kwargs", "kind": "VAR_KEYWORD"}]
    out = []
    for p in parameters:
        entry: dict[str, Any] = {"name": p.name, "kind": p.kind.name}
        annotation = getattr(p.annotation, "__name__", p.annotation)
        if annotation in _SIMPLE_TYPES:
            entry["type"] = annotation
        if p.default is not inspect.Parameter.empty:
            entry["default"] = _jsonable(p.default)
        out.append(entry)
    return out


class HostInterpreter:
    """A fresh interpreter from the host's factory, for a code-executing module built in the sandbox,
    like ``dspy.RLM``.

    The host creates it outside this sandbox, so nothing its code does reaches this one. Its code can still call
    functions that live here, such as ``llm_query`` or a helper the sandboxed code defined: the host pauses the
    execution at each such call and hands it back to ``execute``, which runs it here and resumes. Provided tools
    run on the host directly.
    """

    execution_instructions = ""
    _host: Any = None  # the INTERPRETER_TOOL host tool, set by install()

    def __init__(self, tools: dict[str, Any] | None = None, output_fields: list[dict] | None = None) -> None:
        self.tools = dict(tools or {})
        self.output_fields = output_fields
        self._id: str | None = None

    def start(self) -> None:
        if self._id is None:
            self._id = HostInterpreter._host(op="create")

    def provide_dspy(self) -> None:
        self.start()
        HostInterpreter._host(op="provide", id=self._id)

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self.start()
        tools = {}
        for name, fn in self.tools.items():
            host_tool = fn.name if isinstance(fn, HostTool) else getattr(fn, "__dspy_host_tool__", None)
            tools[name] = {"host": host_tool} if host_tool else {"parameters": _parameters(fn)}
        out = HostInterpreter._host(
            op="execute", id=self._id, code=code, variables=_wire.encode(dict(variables or {})), tools=tools,
            output_fields=self.output_fields,
        )
        while "call" in out:
            call = out["call"]
            try:
                result = self.tools[call["name"]](*_wire.decode(call["args"]), **_wire.decode(call["kwargs"]))
                reply = {"value": _wire.encode(result)}
            except Exception as e:
                reply = {"error": f"{type(e).__name__}: {e}"}
            out = HostInterpreter._host(op="resume", id=self._id, reply=reply)
        if "error" in out:
            error = out["error"]
            kind = {"SyntaxError": SyntaxError, "CodeExecutionError": CodeExecutionError}.get(
                error["type"], CodeInterpreterError
            )
            raise kind(error["message"])
        return FinalOutput(_wire.decode(out["final"])) if "final" in out else _wire.decode(out.get("output"))

    def shutdown(self) -> None:
        if self._id is not None:
            HostInterpreter._host(op="shutdown", id=self._id)
            self._id = None

    def __enter__(self) -> HostInterpreter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)


# -- the trace -------------------------------------------------------------------------------------------------------


class HostTrace(list):
    """The sandbox's trace. Each predictor call recorded here is also added to the host's trace as it happens, so
    trace-reading metrics and optimizers on the host see the calls made in the sandbox."""

    def __init__(self, host: Any) -> None:
        super().__init__()
        self._host = host

    def append(self, entry: Any) -> None:
        super().append(entry)
        # The host's trace is for observing calls, never a reason for one to fail.
        with contextlib.suppress(Exception):
            self._host(entry=_trace_entry(*entry))

    def extend(self, entries: Any) -> None:
        # BestOfN and Refine add the calls of their chosen attempt this way.
        for entry in entries:
            self.append(entry)

    def __iadd__(self, entries: Any) -> HostTrace:
        self.extend(entries)
        return self


# -- install and the Flex driver --------------------------------------------------------------------------------------


def install(namespace: dict[str, Any], spec: dict[str, Any]) -> None:
    """Wire this sandbox's dspy to the host; called once, right after sandboxed code first imports dspy."""
    global _wire
    _wire = GuestValues(namespace)
    settings: dict[str, Any] = {}
    if INTERPRETER_TOOL in namespace:
        HostInterpreter._host = namespace[INTERPRETER_TOOL]
        HostInterpreter.execution_instructions = spec.get("execution_instructions", "")
        settings["interpreter_factory"] = HostInterpreter
    if LM_TOOL in namespace:
        HostAdapter._host = namespace[LM_TOOL]
        settings["adapter"] = HostAdapter()
        if spec.get("lm") is not None:
            settings["lm"] = HostLM(namespace[LM_TOOL], spec["lm"])
    settings["trace"] = HostTrace(namespace[TRACE_TOOL]) if spec.get("trace") and TRACE_TOOL in namespace else None
    dspy.configure(**settings)
    namespace["__dspy_values"] = _wire
    namespace["__dspy_types"] = {
        name: _import(entry["import"]) if "import" in entry else _host_type(name, entry.get("schema") or {})
        for name, entry in spec["types"].items()
    }
    namespace["__dspy_guest_spec"] = spec
    bind(namespace)


def bind(namespace: dict[str, Any]) -> None:
    """Bind the provided tools and custom types in ``namespace``.

    Interpreters may rebind their tools before each execution, so the Flex driver binds again before it runs.
    """
    spec = namespace["__dspy_guest_spec"]
    wire = namespace["__dspy_values"]
    call = namespace.get(CALL_TOOL)
    if call is not None:
        for name, tool in spec["tools"].items():
            current = namespace.get(name)
            # The interpreter's own binding, or none: code may have reused the name for a value of its own.
            if current is None or (callable(current) and not isinstance(current, HostTool)):
                namespace[name] = HostTool(tool, call, wire)
    submit = namespace.get("SUBMIT")
    if callable(submit) and not getattr(submit, "__dspy_wire__", False):
        # The interpreter sends SUBMIT's values as JSON: each goes in the wire form, marked with this session so the
        # host reads back only what was encoded (``read_wire``), never plain JSON that looks like a tagged value.
        def send(value: Any) -> Any:
            return {TAG: WIRE, "session": spec["session"], "value": wire.encode(value)}

        @functools.wraps(submit)
        def wire_submit(*args: Any, **kwargs: Any) -> Any:
            return submit(*map(send, args), **{key: send(value) for key, value in kwargs.items()})

        wire_submit.__dspy_wire__ = True
        namespace["SUBMIT"] = wire_submit
    for name, value in TYPING_NAMES.items():
        namespace.setdefault(name, value)
    namespace.update(namespace["__dspy_types"])


def run_forward(namespace: dict[str, Any], class_name: str, inputs: dict[str, Any]) -> Any:
    """Run one Flex forward in the sandbox; return its output fields as JSON, or as they are in the host's memory."""
    bind(namespace)
    wire = namespace["__dspy_values"]
    prediction = namespace[class_name]().forward(**wire.decode(inputs))
    fields = {"fields": wire.encode(dict(getattr(prediction, "_store", prediction)))}
    return fields if isinstance(wire, InMemory) else json.dumps(fields)


def signature_spec(signature: Any) -> dict[str, Any]:
    """``signature`` as the host rebuilds it (``SandboxSession._signature``): the user's types by their name on the
    host, dspy's and the standard library's (dspy.Reasoning, Decimal, ...) by where they are imported from, and each
    field's settings that are data, like its description, prefix, or the name an adapter writes a reasoning field as."""
    from dspy.predict.flex.ctx import FlexContext

    ctx = FlexContext(signature_cls=signature)
    types = {}
    for name, t in ctx.custom_types().items():
        if isinstance(t, type) and issubclass(t, HostType):
            types[name] = {"host": t.__name__}
        elif t.__module__.split(".")[0] == "dspy" or t.__module__.split(".")[0] in sys.stdlib_module_names:
            types[name] = {"import": f"{t.__module__}.{t.__qualname__}"}
    extra = {name: {key: value for key, value in (field.json_schema_extra or {}).items() if _is_json(value)}
             for name, field in signature.fields.items()}
    return {"signature": ctx.render_signature_string(), "instructions": signature.instructions,
            "fields": ctx.field_kwargs(), "extra": extra, "types": types,
            "inputs": list(signature.input_fields), "outputs": list(signature.output_fields)}


def _is_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _trace_entry(predictor: Any, inputs: dict[str, Any], prediction: Any) -> dict[str, Any]:
    signature = getattr(predictor, "signature", None)
    outputs = getattr(prediction, "_store", None)
    return {"signature": signature_spec(signature) if signature is not None else None,
            "inputs": _wire.encode(dict(inputs)),
            "outputs": _wire.encode(dict(outputs)) if outputs is not None else None}
