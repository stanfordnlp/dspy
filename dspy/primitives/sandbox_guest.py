"""Sandbox side of the dspy bridge: wires the real dspy inside a sandbox to its host.

``dspy.Flex`` modules and ``dspy.RLM`` REPL code import the real dspy in their interpreter. Once they do, the host
(``dspy.primitives.facade``) has this module ``install`` itself there. It routes to the host only what the sandbox
must not own, as JSON over the interpreter's tool-call protocol:

- LM calls. ``HostLM`` sends each request to the host through one host tool, ``LM_TOOL``. The host builds the
  request from its own LM, admits only the options sandboxed code may set, meters the call and runs it, so the
  model, credentials, cache, retries and call budget stay on the host.
- Provided tools. Each is bound as a ``HostTool``: a ``dspy.Tool`` with the host tool's name, description and
  argument schemas, so modules built in the sandbox present it as the host would.
- Custom types named in the host signature, built here as plain classes of their shape: their fields or members,
  never their code (see ``rebuild_types``). Their methods and validators stay on the host.
- An adapter class of the user's, which runs on the host as a whole (``HostAdapter``).

Code-executing modules built in the sandbox, like ``dspy.RLM``, each get a ``HostInterpreter``: a fresh interpreter
the host creates from its factory, outside this sandbox.
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import enum
import importlib
import inspect
import io
import json
import pickle
import sys
import types
from typing import Annotated, Any, ForwardRef, Literal, NamedTuple, Union

import pydantic
import pydantic.dataclasses
import pydantic_core
from typing_extensions import NotRequired, TypedDict

import dspy
from dspy._vendor.lm15 import serde
from dspy.adapters.base import Prompt
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value
from dspy.clients.base_lm import BaseLM
from dspy.lm15 import BuiltinTool, CacheConfig, FunctionTool, Request
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.utils import exceptions

# The host tool every LM call goes through, and the key of an LM error it reports.
LM_TOOL = "__dspy_lm__"
LM_ERROR_KEY = "__dspy_lm_error__"
# The host tool each predictor call recorded in the sandbox's trace is reported to.
TRACE_TOOL = "__dspy_trace__"


# -- wire format, shared with the host ----------------------------------------------------------------------------


def encode_value(value: Any) -> Any:
    """The JSON form of ``value`` that its type reads back on the other side (by field name, see ``from_wire``).

    That is pydantic's own, which keeps a model's exclusions, serializers and settings; a model it cannot encode
    raises rather than crossing some other way. A dspy type crosses as its fields, not its prompt rendering, so an
    ``Image`` still validates as an ``Image``. A value with no JSON form (a function, say) crosses as its string form.
    """
    if isinstance(value, Type):
        return {name: encode_value(getattr(value, name)) for name, f in type(value).model_fields.items() if not f.exclude}
    if isinstance(value, dict):
        out, seen = {}, set()
        for key, item in value.items():
            encoded = _key(key)
            if (written := encoded if isinstance(encoded, str) else json.dumps(encoded)) in seen:  # JSON's keys are strings
                raise ValueError(f"Two keys of a dict cross the sandbox boundary as {written!r}; give them distinct ones.")
            seen.add(written)
            out[encoded] = encode_value(item)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [encode_value(item) for item in value]
    return pydantic_core.to_jsonable_python(value, by_alias=False, round_trip=True, fallback=str)


def validate_wire(adapter: pydantic.TypeAdapter, value: Any) -> Any:
    """``value``, as ``encode_value`` sent it, validated by ``adapter`` by field name. It is validated as the JSON
    pydantic wrote it as, which a strict model accepts too (a tuple as a list, bytes as text)."""
    try:
        text = json.dumps(value)
    except (TypeError, ValueError):  # not a value from the wire
        return adapter.validate_python(value, by_name=True)
    return adapter.validate_json(text, by_name=True)


def _key(key: Any) -> Any:
    """A dict key as JSON can hold it: its JSON form (an enum's value), as a string if that is not a scalar (a tuple's
    is a list). Built from its JSON form, so a model key keeps its excluded fields out."""
    encoded = pydantic_core.to_jsonable_python(key, by_alias=False, round_trip=True, fallback=str)
    return encoded if encoded is None or isinstance(encoded, (str, int, float, bool)) else json.dumps(encoded)


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


# -- the host LM ----------------------------------------------------------------------------------------------------


class HostLM(BaseLM):
    """The host's LM as sandboxed code sees it. Requests are built and run on the host."""

    def __init__(self, host: Any, spec: dict[str, Any], role: str | None = None) -> None:
        super().__init__(model=spec["model"], model_type=spec["model_type"], cache=False, num_retries=0,
                         **spec["kwargs"])
        self._host = host
        self._role = role  # which host LM serves the calls: the main one, or a TwoStepAdapter's "extraction" one
        self._defaults = dict(self.kwargs)
        # The cache setting a copy asked for. This LM's own `cache` stays off: the host caches, not the sandbox.
        self._cache: bool | None = None
        self._capabilities = spec["capabilities"]

    supports_function_calling = property(lambda self: self._capabilities["supports_function_calling"])
    supports_reasoning = property(lambda self: self._capabilities["supports_reasoning"])
    supports_response_schema = property(lambda self: self._capabilities["supports_response_schema"])
    supported_params = property(lambda self: set(self._capabilities["supported_params"]))

    def _send(self, op: str, prompt: Any, options: dict[str, Any]) -> Any:
        return _raise_lm_error(self._host(op=op, prompt=encode_prompt(prompt), options=encode_options(options),
                                          role=self._role))

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
        return serde.response_from_dict(out) if isinstance(prompt, Request) else out

    async def acall(self, prompt: Any = None, **options: Any) -> Any:
        return self(prompt, **options)


def _raise_lm_error(out: Any) -> Any:
    if isinstance(out, dict) and LM_ERROR_KEY in out:
        error = out[LM_ERROR_KEY]
        raise getattr(exceptions, error["type"], exceptions.LMError)(error["message"])
    return out


class HostAdapter(ChatAdapter):
    """An adapter class of the user's as sandboxed predictors see it: each call runs on the host, with the user's
    adapter, as it would without a sandbox. It formats the request, calls the host's LM, metered, and parses the
    answer there. Only the signature, the demos and the inputs cross, and the parsed outputs back."""

    def __init__(self, host: Any) -> None:
        super().__init__()
        self._host = host

    def __call__(self, lm: Any, lm_kwargs: dict[str, Any], signature: Any, demos: list[Any], inputs: dict[str, Any]):
        return self._run(lm, lm_kwargs, signature, demos, inputs, run_async=False)

    async def acall(self, lm: Any, lm_kwargs: dict[str, Any], signature: Any, demos: list[Any], inputs: dict[str, Any]):
        return self._run(lm, lm_kwargs, signature, demos, inputs, run_async=True)  # the host runs the adapter's acall

    def _run(self, lm: Any, lm_kwargs: Any, signature: Any, demos: Any, inputs: Any, run_async: bool) -> Any:
        options = {**(lm._changed() if isinstance(lm, HostLM) else {}), **(lm_kwargs or {})}
        if isinstance(lm, HostLM) and lm._cache is not None:
            options.setdefault("cache", lm._cache)
        call = {"signature": signature_spec(signature), "inputs": encode_value(dict(inputs)), "async": run_async,
                "demos": [encode_value(dict(demo.items())) for demo in demos]}
        outputs = _raise_lm_error(self._host(op="adapter", prompt=call, options=encode_options(options), role=None))
        return [{name: _read(value, signature.output_fields[name].annotation) if name in signature.output_fields
                 else value for name, value in completion.items()} for completion in outputs]


def _read(value: Any, annotation: Any) -> Any:
    """``value`` from the host, as its declared type here, or as it came if it isn't one. The host formats a type it
    doesn't know, like one sandboxed code defined, as text, which is parsed here as an adapter would."""
    with contextlib.suppress(Exception):
        return validate_wire(pydantic.TypeAdapter(annotation), value)
    if isinstance(value, str):
        with contextlib.suppress(Exception):
            return parse_value(value, annotation)
    return value


# -- host tools -----------------------------------------------------------------------------------------------------


class HostTool(Tool):
    """A provided tool as sandboxed code sees it: a ``dspy.Tool`` that calls the host tool by name."""

    def __init__(self, func: Any, spec: dict[str, Any]) -> None:
        super().__init__(func, name=spec["name"], desc=spec["desc"], args=spec["args"],
                         arg_types=dict.fromkeys(spec["args"], Any))
        self.has_kwargs = spec["has_kwargs"]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Generated code may call a tool positionally, in the order the host declares its arguments.
        return super().__call__(**dict(zip(self.args, args, strict=False)), **kwargs)

    def _validate_and_parse_args(self, **kwargs: Any) -> dict[str, Any]:
        return {key: encode_value(value) for key, value in super()._validate_and_parse_args(**kwargs).items()}


# -- custom types ---------------------------------------------------------------------------------------------------

# Typing names a signature's annotations may use; sandboxed code is told not to import at module scope.
TYPING_NAMES = {"Any": Any, "Literal": Literal, "Optional": __import__("typing").Optional, "Union": Union}


# Settings of a pydantic model that decide what its values look like on the wire, which the sandbox's shape keeps.
_CONFIG_KEYS = (
    "extra", "frozen", "strict", "use_enum_values", "ser_json_bytes", "val_json_bytes", "ser_json_timedelta",
    "ser_json_inf_nan", "str_strip_whitespace", "str_to_lower", "str_to_upper", "str_min_length", "str_max_length",
    "populate_by_name", "validate_by_name", "validate_by_alias", "serialize_by_alias",
)


def rebuild_types(spec: dict[str, Any]) -> dict[str, Any]:
    """The custom types of the host signature, by name, as the host describes them (``facade.types_spec``): imported,
    or built here as plain classes of the same names and shapes, in a module of their own, so they can name each
    other. One whose shape can't be built here is a model that takes any fields."""
    module = types.ModuleType("__dspy_types")
    sys.modules[module.__name__] = module
    for key, shape in spec.get("shapes", {}).items():  # the types each holds first, as the host lists them
        try:
            built = _build(shape, module)
        except Exception:
            built = pydantic.create_model(str(shape.get("name", key)), __config__=pydantic.ConfigDict(extra="allow"))
        built.__module__ = module.__name__
        setattr(module, key, built)
    for built in list(vars(module).values()):  # now that the types each names are there
        if isinstance(built, type) and issubclass(built, pydantic.BaseModel):
            with contextlib.suppress(Exception):
                built.model_rebuild(force=True, _types_namespace=vars(module))
    return {name: _annotation(t, module) for name, t in spec.get("types", {}).items()}


def _annotation(spec: dict[str, Any], module: types.ModuleType) -> Any:
    try:
        if "import" in spec:
            return _import(spec["import"])
        if "shape" in spec:
            return vars(module).get(spec["shape"]) or ForwardRef(spec["shape"])
        if "none" in spec:
            return type(None)
        if "ellipsis" in spec:
            return ...
        if "literal" in spec:
            values = tuple(_literal(value, module) for value in spec["literal"])
            return Literal[values] if values else Any
        if "union" in spec:
            return Union[tuple(_annotation(arg, module) for arg in spec["union"])]  # noqa: UP007  # built at runtime
        if "annotated" in spec:
            inner = _annotation(spec["annotated"], module)
            metadata = [_constraint(c) for c in spec["metadata"]]
            return Annotated[(inner, *metadata)] if metadata else inner
        if "generic" in spec:
            args = tuple(_annotation(arg, module) for arg in spec["args"])
            return _import(spec["generic"])[args] if args else _import(spec["generic"])
    except Exception:
        pass
    return Any


def _literal(value: Any, module: types.ModuleType) -> Any:
    """A literal's value: an enum member is looked up on its enum, which is built before the types that name it."""
    if not isinstance(value, dict):
        return value
    owner = _annotation(value["member"], module)
    return getattr(owner, value["name"], value["name"]) if isinstance(owner, type) else value["name"]


def _constraint(spec: dict[str, Any]) -> Any:
    marker = _import(spec["import"])
    return marker if spec["values"] is None else marker(**spec["values"])


def _build(shape: dict[str, Any], module: types.ModuleType) -> type:
    """A plain class of ``shape``: a pydantic model, dataclass, NamedTuple, TypedDict or enum, with no code of its own."""
    name, kind = str(shape["name"]), shape["kind"]
    if kind in ("enum", "flag"):
        base = {"str": str, "int": int}.get(shape.get("base"))
        if kind == "flag":
            return (enum.IntFlag if base is int else enum.Flag)(name, shape["members"])
        if base is int:
            return enum.IntEnum(name, shape["members"])
        return enum.Enum(name, shape["members"], **({"type": str} if base is str else {}))
    fields = {key: (_field_type(field, module), field) for key, field in shape.get("fields", {}).items()
              if isinstance(key, str) and key.isidentifier() and not key.startswith("_")}
    if kind in ("model", "root", "dataclass"):
        namespace = {
            "__module__": module.__name__,
            "__annotations__": {key: annotation for key, (annotation, _) in fields.items()},
        }
        namespace.update({key: _field_info(field) for key, (_, field) in fields.items()
                          if _has_default(field)})
        if isinstance(shape.get("doc"), str):
            namespace["__doc__"] = shape["doc"]
        if any(field.get("host_default") for _, field in fields.values()):
            namespace["_leave_host_defaults"] = pydantic.model_serializer(mode="wrap")(_leave_host_defaults)
        # A RootModel takes no extra, and a dataclass is frozen by its own parameter.
        config = {key: value for key, value in shape.get("config", {}).items()
                  if key in _CONFIG_KEYS and (kind, key) not in (("root", "extra"), ("dataclass", "frozen"))}
        if kind == "dataclass":
            dataclass = pydantic.dataclasses.dataclass(frozen=bool(shape.get("frozen")), config=pydantic.ConfigDict(**config))
            return dataclass(type(name, (), namespace))
        namespace["model_config"] = pydantic.ConfigDict(**config)
        base = pydantic.RootModel if kind == "root" else pydantic.BaseModel
        return types.new_class(name, (base,), {}, lambda body: body.update(namespace))
    if kind == "namedtuple":
        built = NamedTuple(name, [(key, annotation) for key, (annotation, _) in fields.items()])
        defaults = [field["default"] for _, field in fields.values() if "default" in field]
        built.__new__.__defaults__ = tuple(defaults) or None
        return built
    if kind == "typeddict":
        required = set(shape.get("required", ()))
        return TypedDict(name, {key: annotation if key in required else NotRequired[annotation]
                                for key, (annotation, _) in fields.items()})
    return pydantic.create_model(name, __config__=pydantic.ConfigDict(extra="allow"))


def _field_type(field: dict[str, Any], module: types.ModuleType) -> Any:
    annotation = _annotation(field["type"], module)
    if "default" in field and field["default"] is not None:  # its JSON form, read back as the field's type
        with contextlib.suppress(Exception):
            field["default"] = validate_wire(pydantic.TypeAdapter(annotation), field["default"])
    metadata = [_constraint(c) for c in field.get("metadata", ())]
    if field.get("host_default"):  # holds HOST_DEFAULT until it is set, which it takes as it is and doesn't send
        metadata += [pydantic.WrapValidator(_or_host_default), pydantic.WrapSerializer(_unless_host_default)]
    return Annotated[(annotation, *metadata)] if metadata else annotation


def _field_info(field: dict[str, Any]) -> Any:
    settings = {key: field[key] for key in ("description", "alias", "serialization_alias", "exclude") if key in field}
    if "validation_alias" in field:
        settings["validation_alias"] = _alias_from(field["validation_alias"])
    if "factory" in field:
        return pydantic.Field(default_factory=_import(field["factory"]), **settings)
    if field.get("host_default"):
        return pydantic.Field(default_factory=lambda: HOST_DEFAULT, **settings)
    return pydantic.Field(field["default"] if "default" in field else ..., **settings)


def _has_default(field: dict[str, Any]) -> bool:
    return any(key in field for key in ("default", "factory", "host_default", "description", "alias",
                                        "serialization_alias", "validation_alias", "exclude"))


def _alias_from(alias: Any) -> Any:
    if isinstance(alias, dict) and "path" in alias:
        return pydantic.AliasPath(*alias["path"])
    if isinstance(alias, dict):
        return pydantic.AliasChoices(*(_alias_from(choice) for choice in alias["choices"]))
    return alias


class _HostDefault:
    def __repr__(self) -> str:
        return "<made on the host>"

    def __reduce__(self) -> str:  # copied and pickled as the one marker
        return "HOST_DEFAULT"


# What a field whose default only the host can make (with a factory of the user's) holds in the sandbox until it is
# set. A built model or dataclass doesn't send such a field, so the host makes its default.
HOST_DEFAULT = _HostDefault()


def _leave_host_defaults(self: Any, handler: Any) -> Any:
    data = handler(self)
    if isinstance(data, dict):
        for name, value in vars(self).items():
            if value is HOST_DEFAULT:
                data.pop(name, None)
    return data


def _or_host_default(value: Any, handler: Any) -> Any:
    return value if value is HOST_DEFAULT else handler(value)


def _unless_host_default(value: Any, handler: Any) -> Any:
    return None if value is HOST_DEFAULT else handler(value)


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
            entry["default"] = encode_value(p.default)
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
            op="execute", id=self._id, code=code, variables=encode_value(dict(variables or {})), tools=tools,
            output_fields=self.output_fields,
        )
        while "call" in out:
            call = out["call"]
            try:
                reply = {"value": encode_value(self.tools[call["name"]](*call["args"], **call["kwargs"]))}
            except Exception as e:
                reply = {"error": f"{type(e).__name__}: {e}"}
            out = HostInterpreter._host(op="resume", id=self._id, reply=reply)
        if "error" in out:
            error = out["error"]
            kind = {"SyntaxError": SyntaxError, "CodeExecutionError": CodeExecutionError}.get(
                error["type"], CodeInterpreterError
            )
            raise kind(error["message"])
        return FinalOutput(out["final"]) if "final" in out else out.get("output")

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


def _import(path: str) -> Any:
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def rebuild_adapter(spec: dict[str, Any], host: Any = None, custom_types: dict[str, Any] | None = None) -> Any:
    """The host's adapter, unpickled. ``host`` is the LM host tool, which serves a TwoStepAdapter's extraction model,
    and ``custom_types`` the user's types defined here, which the adapter may name."""

    if spec.get("host"):  # an adapter class of the user's, which runs on the host
        return HostAdapter(host)

    class Unpickler(pickle.Unpickler):
        def persistent_load(self, pid: str) -> Any:
            if pid.startswith("type:"):
                return (custom_types or {})[pid.removeprefix("type:")]
            return HostLM(host, spec["extraction_lm"], role="extraction")

    return Unpickler(io.BytesIO(base64.b64decode(spec["pickle"]))).load()


def install(namespace: dict[str, Any], spec: dict[str, Any]) -> None:
    """Wire this sandbox's dspy to the host; called once, right after sandboxed code first imports dspy."""
    settings: dict[str, Any] = {}
    if INTERPRETER_TOOL in namespace:
        HostInterpreter._host = namespace[INTERPRETER_TOOL]
        HostInterpreter.execution_instructions = spec.get("execution_instructions", "")
        settings["interpreter_factory"] = HostInterpreter
    if spec.get("lm") is not None and LM_TOOL in namespace:
        settings["lm"] = HostLM(namespace[LM_TOOL], spec["lm"])
    settings["trace"] = HostTrace(namespace[TRACE_TOOL]) if spec.get("trace") and TRACE_TOOL in namespace else None
    namespace["__dspy_types"] = rebuild_types(spec["types"])  # before the adapter, which may name some
    if spec.get("adapter") is not None:
        with contextlib.suppress(Exception):
            settings["adapter"] = rebuild_adapter(spec["adapter"], namespace.get(LM_TOOL), namespace["__dspy_types"])
    dspy.configure(**settings)
    namespace["__dspy_guest_spec"] = spec
    bind(namespace)


def bind(namespace: dict[str, Any]) -> None:
    """Bind the provided tools and custom types in ``namespace``.

    Interpreters may rebind their tools before each execution, so the Flex driver binds again before it runs.
    """
    spec = namespace["__dspy_guest_spec"]
    for name, tool in spec["tools"].items():
        func = namespace.get(name)
        if func is not None and not isinstance(func, HostTool):
            func.__dspy_host_tool__ = name  # so a nested interpreter calls the host tool directly
            namespace[name] = HostTool(func, tool)
    if "__dspy_types" not in namespace:
        namespace["__dspy_types"] = rebuild_types(spec["types"])
    for name, value in TYPING_NAMES.items():
        namespace.setdefault(name, value)
    namespace.update(namespace["__dspy_types"])


def run_forward(namespace: dict[str, Any], class_name: str, inputs: dict[str, Any]) -> str:
    """Run one Flex forward in the sandbox; return its output fields as JSON."""
    bind(namespace)
    spec = namespace["__dspy_guest_spec"]
    types = namespace["__dspy_types"]
    for name, annotation in spec.get("inputs", {}).items():
        if name in inputs:
            try:
                parsed = eval(annotation, {**vars(builtins), **vars(__import__("typing")), "dspy": dspy}, types)
                adapter = pydantic.TypeAdapter(parsed)
            except Exception:
                continue  # a type pydantic can't validate: the value as it came
            with contextlib.suppress(pydantic.ValidationError):  # a value its type rejects, passed as it came
                inputs[name] = validate_wire(adapter, inputs[name])
    prediction = namespace[class_name]().forward(**inputs)
    fields = getattr(prediction, "_store", prediction)
    return json.dumps({"fields": encode_value(dict(fields))})


def signature_spec(signature: Any) -> dict[str, Any]:
    """``signature`` as the host rebuilds it (``SandboxSession._signature``), with the user's types by name."""
    from dspy.predict.flex.ctx import FlexContext

    ctx = FlexContext(signature_cls=signature)
    # dspy's and the standard library's types (dspy.Reasoning, Decimal, ...) resolve on the host by import; others by name.
    types = {name: f"{t.__module__}.{t.__qualname__}" for name, t in ctx.custom_types().items()
             if t.__module__.split(".")[0] == "dspy" or t.__module__.split(".")[0] in sys.stdlib_module_names}
    return {"signature": ctx.render_signature_string(), "instructions": signature.instructions,
            "fields": ctx.field_kwargs(), "types": types,
            "inputs": list(signature.input_fields), "outputs": list(signature.output_fields)}


def _trace_entry(predictor: Any, inputs: dict[str, Any], prediction: Any) -> dict[str, Any]:
    signature = getattr(predictor, "signature", None)
    outputs = getattr(prediction, "_store", None)
    return {"signature": signature_spec(signature) if signature is not None else None,
            "inputs": encode_value(dict(inputs)),
            "outputs": encode_value(dict(outputs)) if outputs is not None else None}
