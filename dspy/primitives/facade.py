"""Host side of the sandbox dspy bridge.

Sandboxed code (a ``dspy.Flex`` module or a ``dspy.RLM`` REPL block) imports the real dspy in its interpreter.
``SandboxSession.install`` makes dspy importable there (``PythonInterpreter.provide_dspy``) and runs ``BOOTSTRAP``,
which installs ``dspy.primitives.sandbox_guest`` as soon as sandboxed code imports dspy. From then on, only JSON
crosses the boundary, over the interpreter's tool-call protocol (``CodeInterpreter.tools``), and values cross in the form ``sandbox_guest.Wire`` describes,
read and written on the host by the session's ``HostValues``:

- ``LM_TOOL``: every LM call made in the sandbox. The host builds the request from its own LM on a ``SandboxLM``,
  which admits only generation options (``SANDBOX_LM_KWARGS``) and meters each completion, then runs it. The model,
  credentials, cache, retries and history stay on the host. A predictor call runs whole on the host, with the
  host's adapter, which formats and parses it with the real types (``SandboxSession._run_adapter``).
- ``CALL_TOOL``: the provided tools, which the sandbox binds as ``dspy.Tool``s with their host metadata.
- The user's custom types stay here: a value of one crosses as plain data, the JSON its type writes, which the host
  reads back as the type (``HostValues``).
- ``INTERPRETER_TOOL``: a fresh interpreter from ``interpreter_factory`` for each code-executing module built in the
  sandbox, so its code shares nothing with the sandbox that built it.

An interpreter that runs code in the host's own memory (an in-process interpreter) uses the host's dspy as is:
the bootstrap leaves it alone, and the caller scopes its execution with ``SandboxSession.scope``.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import contextvars
import copy
import dataclasses
import enum
import functools
import inspect
import itertools
import json
import logging
import queue
import secrets
import sys
import types
import typing
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pydantic
import pydantic_core
from pydantic.json_schema import GenerateJsonSchema

import dspy
from dspy._vendor.lm15 import serde
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import _get_adapter, parse_value
from dspy.clients.base_lm import BaseLM
from dspy.lm15 import Response
from dspy.primitives.code_interpreter import (
    CodeExecutionError,
    CodeInterpreterError,
    FinalOutput,
    _create_interpreter,
    resolve_interpreter_factory,
)
from dspy.primitives.sandbox_bundle import sandbox_modules
from dspy.primitives.sandbox_guest import (
    CALL_TOOL,
    INTERPRETER_TOOL,
    LM_ERROR_KEY,
    LM_TOOL,
    TRACE_TOOL,
    TYPING_NAMES,
    HostInterpreter,
    InMemory,
    Wire,
    _jsonable,
    decode_options,
    decode_prompt,
    encode_options,
)
from dspy.primitives.sandbox_serializable import KEEP_IN_DUMP, SandboxSerializable
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import make_signature
from dspy.utils.exceptions import LMError, LMNotConfiguredError

logger = logging.getLogger(__name__)

# Identifies this process to the bootstrap, so it can tell when it runs in the host's own memory.
_HOST_TOKEN = secrets.token_hex(16)
# Bootstrapped namespaces that share the host's memory, by session id.
_in_host: dict[str, dict[str, Any]] = {}

BOOTSTRAP = """
def __dspy_bootstrap(spec):
    import sys

    host = sys.modules.get("dspy.primitives.facade")
    if getattr(host, "_HOST_TOKEN", None) == spec["host_token"]:
        # This namespace shares the host's memory (an in-process interpreter): it keeps the host's dspy.
        host._in_host[spec["session"]] = globals()
        return
    namespace = globals()

    def install():
        from dspy.primitives.sandbox_guest import install

        install(namespace, spec)

    if "dspy" in sys.modules:
        namespace.setdefault("dspy", sys.modules["dspy"])
        install()
        return

    class LazyDspy:
        # `dspy` is in scope before sandboxed code imports it, and imports it on first use.
        def __getattr__(self, name):
            import dspy

            namespace["dspy"] = dspy
            return getattr(dspy, name)

    namespace.setdefault("dspy", LazyDspy())

    class InstallOnImport:
        # Installs the guest right after sandboxed code first imports dspy.
        def find_spec(self, name, path=None, target=None):
            if name != "dspy":
                return None
            sys.meta_path.remove(self)
            import importlib.util

            found = importlib.util.find_spec(name)
            if found is not None and found.loader is not None:
                exec_module = found.loader.exec_module

                def exec_and_install(module):
                    exec_module(module)
                    install()

                found.loader.exec_module = exec_and_install
            return found

    sys.meta_path.insert(0, InstallOnImport())


__dspy_bootstrap(__dspy_spec)
del __dspy_bootstrap, __dspy_spec
"""


def is_reserved_sandbox_name(name: str) -> bool:
    """Names the bridge owns in the sandbox namespace: ``dspy`` and the ``_dspy``/``__dspy`` prefixes of its
    internals and host-injected variables."""
    return name == "dspy" or name.startswith(("_dspy", "__dspy"))


def _resolve_signature(signature: dict[str, Any], custom_types: dict[str, type] | None = None) -> Any:
    """Rebuild a signature described as a string signature, instructions and each field's kwargs."""
    sig = make_signature(signature["signature"], signature.get("instructions"), custom_types=custom_types)
    fields = signature.get("fields")
    if fields:
        sig = make_signature(
            {
                name: (f.annotation, (InputField if name in sig.input_fields else OutputField)(**fields.get(name, {})))
                for name, f in sig.fields.items()
            },
            sig.instructions,
        )
    return sig


# Generation parameters sandboxed code may set per LM call; routing, credential, and transport options are not.
SANDBOX_LM_KWARGS = frozenset({
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "n", "seed",
    "logprobs", "top_logprobs", "response_format",
    "reasoning", "reasoning_effort",
    "tools", "tool_choice", "parallel_tool_calls",
    "cache", "rollout_id", "prompt_cache", "prompt_cache_key",
})
_LM_CALL_INPUTS = frozenset({"prompt", "messages", "request"})
# The settings of a typed request that those options set. The rest of its config comes from the host LM.
_SANDBOX_CONFIG_FIELDS = frozenset({
    "max_tokens", "temperature", "top_p", "stop", "response_format", "tool_choice", "reasoning", "cache", "seed",
    "logprobs",
})
# The extensions key of a request handed to the sandbox that says which host-owned settings it was built with.
_SEALED_KEY = "__dspy_host_settings__"


class SandboxLM(BaseLM):
    """The host LM as sandboxed code may call it: only ``SANDBOX_LM_KWARGS`` per call, and ``reserve(n)`` before
    each, one slot per requested completion."""

    def __init__(
        self,
        lm: Any,
        reserve: Callable[[int], None] | None = None,
        built: list[dict[str, Any]] | None = None,
        refuse_host_files: bool = False,
    ) -> None:
        # lm may be any callable RLM accepts as sub_lm, not only a BaseLM. Adapters and Predict read the
        # generation defaults off the proxy; credentials and routing stay on the wrapped LM.
        super().__init__(model=getattr(lm, "model", None), model_type=getattr(lm, "model_type", "chat"))
        self.kwargs = {k: v for k, v in (getattr(lm, "kwargs", None) or {}).items() if k in SANDBOX_LM_KWARGS}
        self._lm = lm
        self._reserve = reserve
        # The host-owned settings of the requests built for sandboxed code, which its typed calls may carry back.
        self._built = built if built is not None else []
        self._refuse_host_files = refuse_host_files

    @staticmethod
    def _check(kwargs: dict[str, Any], *, allow_inputs: bool = True) -> None:
        allowed = SANDBOX_LM_KWARGS | _LM_CALL_INPUTS if allow_inputs else SANDBOX_LM_KWARGS
        rejected = sorted(kwargs.keys() - allowed)
        if rejected:
            raise TypeError(
                f"Sandboxed code may not set LM option(s) {rejected}; allowed per call: {sorted(SANDBOX_LM_KWARGS)}."
            )

    def _build_adapter_request(self, prompt, options):
        from dspy.clients.requests import build_request

        # Only overrides come from sandboxed code. Host defaults remain trusted,
        # including provider-specific settings that are not exposed on this proxy.
        self._check(options or {}, allow_inputs=False)
        request = build_request(self._lm, prompt, options)
        if (owned := _host_owned(request.config)) not in self._built:
            self._built.append(owned)
        return request

    def _admit(self, kwargs: dict[str, Any], args: tuple = ()) -> None:
        self._check(kwargs)
        prompt = args[0] if args else kwargs.get("request", kwargs.get("prompt"))
        if self._refuse_host_files:
            _refuse_host_files(prompt)
        if isinstance(prompt, dspy.lm15.Request):
            self._check_request(prompt)
        if self._reserve is not None:
            # Typed requests are singular even when the wrapped LM defaults to n > 1.
            count = 1 if isinstance(prompt, dspy.lm15.Request) else max(1, int(kwargs.get("n", self.kwargs.get("n")) or 1))
            self._reserve(count)

    def _check_request(self, request: Any) -> None:
        from dspy.clients.requests import build_request

        owned = _host_owned(request.config)
        if owned in self._built:
            return
        try:
            host = _host_owned(build_request(self._lm, ".", {}).config)
        except Exception:
            host = _host_owned(dspy.lm15.Config())
        rejected = sorted(name for name, value in owned.items() if value != host[name])
        if rejected:
            raise TypeError(f"Sandboxed code may not change request setting(s) {rejected}; they come from the host LM.")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs, args)
        return self._lm(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs, args)
        return await self._lm.acall(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs, args)
        return self._lm.forward(*args, **kwargs)

    async def aforward(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs, args)
        return await self._lm.aforward(*args, **kwargs)

    def copy(self, **kwargs: Any) -> SandboxLM:
        self._check(kwargs)
        return SandboxLM(self._lm.copy(**kwargs), self._reserve, self._built, self._refuse_host_files)

    # Adapters choose structured outputs and native tool calling by these; answer for the wrapped model.
    @property
    def supports_function_calling(self) -> bool:
        return getattr(self._lm, "supports_function_calling", False)

    @property
    def supports_reasoning(self) -> bool:
        return getattr(self._lm, "supports_reasoning", False)

    @property
    def supports_response_schema(self) -> bool:
        return getattr(self._lm, "supports_response_schema", False)

    @property
    def supported_params(self) -> set[str]:
        return getattr(self._lm, "supported_params", set())

    @property
    def supported_tool_result_media(self) -> set[str]:
        return getattr(self._lm, "supported_tool_result_media", set())


def _host_owned(config: Any) -> dict[str, Any]:
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config) if f.name not in _SANDBOX_CONFIG_FIELDS}


class _CancelCall:
    """Reply that ends a nested call the parent sandbox will no longer answer."""


class _NestedInterpreter:
    """A fresh interpreter that a code-executing module built in the sandbox asked for (see ``HostInterpreter``).

    It executes on a thread of its own, so an execution can pause whenever its code calls a function that lives in
    the parent sandbox: ``execute`` and ``resume`` return that call to the parent, which answers it with the next
    ``resume``. The parent's sandbox is never re-entered.
    """

    def __init__(self, interpreter: Any, host_tools: dict[str, Callable[..., Any]], values: HostValues) -> None:
        self.interpreter = interpreter
        self._host_tools = host_tools
        self._values = values  # how values cross to and from the parent sandbox
        self._thread = ThreadPoolExecutor(max_workers=1)  # an interpreter may be bound to one thread
        self._events: queue.Queue = queue.Queue()
        self._replies: queue.Queue = queue.Queue()
        self._busy = False

    def execute(self, code: str, variables: dict, tools: dict[str, dict], output_fields: list | None) -> dict:
        if self._busy:
            raise CodeInterpreterError("This interpreter is already executing code.")
        interpreter = self.interpreter
        interpreter.tools.update({
            name: self._host_tools[spec["host"]] if spec.get("host") in self._host_tools else self._parent_call(name, spec)
            for name, spec in tools.items()
        })
        if hasattr(interpreter, "output_fields"):
            interpreter.output_fields = output_fields
        if hasattr(interpreter, "_tools_registered"):
            interpreter._tools_registered = False  # register the tools anew, as RLM does for its interpreters
        self._busy = True
        self._thread.submit(contextvars.copy_context().run, self._run, code, variables)
        return self._next()

    def resume(self, reply: dict) -> dict:
        # Busy means paused at a call into the parent: execute and resume return only at a call or once it is done.
        if not self._busy:
            raise CodeInterpreterError("This interpreter has no call waiting for a reply.")
        self._replies.put(reply)
        return self._next()

    def close(self) -> None:
        if self._busy:
            self._replies.put(_CancelCall)
            while self._next().get("call"):
                self._replies.put(_CancelCall)
        try:
            self._thread.submit(self.interpreter.shutdown).result()
        finally:
            self._thread.shutdown()

    def _run(self, code: str, variables: dict) -> None:
        try:
            self._events.put(("done", self.interpreter.execute(code, variables)))
        except BaseException as e:
            self._events.put(("error", e))

    def _next(self) -> dict:
        kind, value = self._events.get()
        if kind == "call":
            return {"call": value}
        self._busy = False
        if kind == "error":
            name = type(value).__name__ if isinstance(value, (SyntaxError, CodeExecutionError)) else "CodeInterpreterError"
            return {"error": {"type": name, "message": str(value)}}
        if isinstance(value, FinalOutput):
            return {"final": self._values.encode(value.output)}
        return {"output": self._values.encode(value)}

    def _parent_call(self, name: str, spec: dict) -> Callable[..., Any]:
        """A stand-in for a parent-sandbox function, with its signature, that pauses the execution to call it."""

        def call(*args: Any, **kwargs: Any) -> Any:
            call = {"name": name, "args": self._values.encode(list(args)), "kwargs": self._values.encode(kwargs)}
            self._events.put(("call", call))
            reply = self._replies.get()
            if reply is _CancelCall:
                raise CodeInterpreterError(f"The sandbox that owns {name!r} stopped before answering the call.")
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return self._values.decode(reply["value"])

        call.__name__ = call.__qualname__ = name
        call.__signature__ = inspect.Signature([
            inspect.Parameter(
                p["name"],
                _PARAMETER_KINDS[p["kind"]],
                default=p.get("default", inspect.Parameter.empty),
                annotation=_SIMPLE_ANNOTATIONS.get(p.get("type"), inspect.Parameter.empty),
            )
            for p in spec.get("parameters", [])
        ])
        return call


_SIMPLE_ANNOTATIONS = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}
# A parameter's kind by the name the sandbox sends (``_parameters``), looked up rather than read off a host object.
_PARAMETER_KINDS = {kind.name: kind for kind in type(inspect.Parameter.POSITIONAL_ONLY)}


def _entrypoint(tool: Any) -> Callable[..., Any]:
    """A provided tool as an interpreter tool: called through its ``dspy.Tool``, with its function's signature."""
    if not isinstance(tool, Tool):
        return tool

    @functools.wraps(tool.func)
    def entrypoint(**kwargs: Any) -> Any:
        return tool(**kwargs)

    return entrypoint


class SandboxSession:
    """Host side of one interpreter session that runs the real dspy: one Flex forward, or one RLM call.

    Serves the sandbox's LM calls on a ``SandboxLM`` over ``lm`` (``dspy.settings.lm`` at call time when ``None``),
    with ``reserve`` metering every completion, and adds the predictor calls the sandbox records to the host's trace. Each
    code-executing module built in the sandbox gets a fresh interpreter from ``interpreter_factory``; ``close``
    shuts down any still open. Values cross to and from the sandbox through ``encode`` and ``decode``.
    """

    def __init__(
        self,
        tools: dict[str, Any],
        *,
        custom_types: dict[str, type] | None = None,
        lm: Any = None,
        reserve: Callable[[int], None] | None = None,
        interpreter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._tools = tools
        self._custom_types = custom_types or {}
        self._values = HostValues(tools)
        self._lm = lm
        self._reserve = reserve
        self._interpreter_factory = interpreter_factory
        self._nested: dict[str, _NestedInterpreter] = {}
        self._interpreter: Any = None
        self._adapter: Any = None
        self._calls = itertools.count(1)
        self._built: list[dict[str, Any]] = []  # see SandboxLM
        self._sealed: list[dict[str, Any]] = []  # see _seal
        self.lm_error: tuple[LMError, str] | None = None
        self.in_host = False
        self.id: str | None = None  # the session's id in the interpreter, once installed

    def install(self, interpreter: Any, *, import_dspy: bool) -> None:
        """Bootstrap the bridge in ``interpreter``; with ``import_dspy``, import dspy there right away."""
        provide = getattr(interpreter, "provide_dspy", None)
        if callable(provide):
            provide()
        self._interpreter = interpreter
        interpreter.tools[LM_TOOL] = self._serve_lm
        interpreter.tools[TRACE_TOOL] = self._serve_trace
        interpreter.tools[CALL_TOOL] = self._serve_tool
        # Inside a sandbox, the host already gave this interpreter the host's own interpreter tool.
        if self._interpreter_factory is not None and not isinstance(interpreter, HostInterpreter):
            interpreter.tools[INTERPRETER_TOOL] = self._serve_interpreter
        session = self.id = secrets.token_hex(8)
        spec = self._spec(session)
        interpreter.execute(BOOTSTRAP, variables={"__dspy_spec": spec})
        namespace = _in_host.pop(session, None)
        if namespace is not None:
            # Code in the host's memory has the host's own types and values: nothing crosses.
            self.in_host = True
            self._values = InMemory(namespace)
            namespace["__dspy_guest_spec"] = spec
            namespace["__dspy_values"] = self._values
            namespace["__dspy_types"] = dict(self._custom_types)
            namespace.update({**{k: v for k, v in TYPING_NAMES.items() if k not in namespace}, **self._custom_types})
        if import_dspy:
            interpreter.execute("import dspy")

    def encode(self, value: Any) -> Any:
        """``value`` as it crosses into the sandbox."""
        return self._values.encode(value)

    def decode(self, data: Any) -> Any:
        """A value that crossed from the sandbox, with the host's objects it sent back as themselves."""
        return self._values.decode(data)

    def scope(self) -> contextlib.AbstractContextManager[Any]:
        """Run code of an interpreter that shares the host's memory on the metered LM; a no-op otherwise."""
        lm = self._resolve_lm()
        if not self.in_host or lm is None:
            return contextlib.nullcontext()
        factory = {"interpreter_factory": self._interpreter_factory} if self._interpreter_factory else {}
        return dspy.context(lm=SandboxLM(lm, self._reserve, self._built), **factory)

    def close(self) -> None:
        """Shut down the interpreters made for code-executing modules in the sandbox that are still open."""
        nested, self._nested = list(self._nested.values()), {}
        for interpreter in nested:
            try:
                interpreter.close()
            except Exception:
                logger.warning("Shutting down a nested interpreter failed", exc_info=True)

    def _serve_interpreter(
        self,
        op: str,
        id: str | None = None,
        code: str | None = None,
        variables: dict | None = None,
        tools: dict | None = None,
        output_fields: list | None = None,
        reply: dict | None = None,
    ) -> Any:
        """The ``INTERPRETER_TOOL`` host tool behind ``HostInterpreter``: one fresh interpreter per ``create``."""
        if op == "create":
            interpreter = _create_interpreter(self._interpreter_factory)
            interpreter.tools[INTERPRETER_TOOL] = self._serve_interpreter  # for the modules its own code builds
            handle = secrets.token_hex(8)
            host_tools = {name: _entrypoint(tool) for name, tool in self._tools.items()}
            self._nested[handle] = _NestedInterpreter(interpreter, host_tools, self._values)
            return handle
        nested = self._nested.get(id)
        if nested is None:
            raise CodeInterpreterError(f"Unknown nested interpreter {id!r}")
        if op == "provide":
            provide = getattr(nested.interpreter, "provide_dspy", None)
            if callable(provide):
                provide()
            return None
        if op == "execute":
            return nested.execute(code or "", self._values.decode(variables or {}), tools or {}, output_fields)
        if op == "resume":
            return nested.resume(reply or {})
        if op == "shutdown":
            self._nested.pop(id).close()
            return None
        raise CodeInterpreterError(f"Unknown nested interpreter operation {op!r}")

    def _serve_tool(self, tool: str, arguments: dict | None = None) -> Any:
        """The ``CALL_TOOL`` host tool: call the provided tool named ``tool``. Its arguments are read off the wire, so
        a value the host sent comes back to it as the host's own object. It runs as the interpreter's own tool of that
        name if the caller registered one, like an RLM, which awaits an async tool and reports to the tool callbacks,
        so a call is the same whether sandboxed code makes it through dspy or directly."""
        if tool not in self._tools:
            raise CodeInterpreterError(f"Unknown tool {tool!r}")
        provided = self._tools[tool]
        registered = (getattr(self._interpreter, "tools", None) or {}).get(tool)
        function = registered if callable(registered) else _entrypoint(provided)
        kwargs = self._values.decode(arguments or {})
        # A dspy.Tool (an RLM's registered tools call theirs) reads its arguments as declared itself: reading them here
        # too would validate a model twice. Only a plain function gets them as declared from here.
        if not isinstance(provided, Tool):
            declared = Tool(provided).arg_types
            kwargs = {name: _as_declared(value, declared.get(name)) for name, value in kwargs.items()}
        result = function(**kwargs)
        if asyncio.iscoroutine(result):
            from dspy.primitives.python_interpreter import _await_in_sync

            result = _await_in_sync(result)
        return self._values.encode(result)

    def _serve_trace(self, entry: dict[str, Any]) -> None:
        """The ``TRACE_TOOL`` host tool: add a predictor call recorded in the sandbox to the host's trace, with the
        predictor rebuilt from its signature."""
        trace = dspy.settings.trace
        if trace is None or dspy.settings.max_trace_size <= 0:
            return
        signature = self._signature(entry["signature"]) if entry.get("signature") else None
        inputs, outputs = self._values.decode(entry["inputs"]), self._values.decode(entry.get("outputs") or {})
        if signature is not None:  # read as the declared types, as the user's arrive as plain data
            inputs = {name: _typed(value, signature.input_fields.get(name)) for name, value in inputs.items()}
            outputs = {name: _typed(value, signature.output_fields.get(name)) for name, value in outputs.items()}
        if len(trace) >= dspy.settings.max_trace_size:
            trace.pop(0)
        trace.append((dspy.Predict(signature) if signature is not None else None, inputs, dspy.Prediction(**outputs)))

    def _signature(self, signature: dict[str, Any]) -> Any:
        fields = {name: dict(kwargs) for name, kwargs in signature.get("fields", {}).items()}
        for name, value in signature.get("defaults", {}).items():
            fields.setdefault(name, {})["default"] = self._values.decode(value)
        signature = {**signature, "fields": fields}
        try:
            names = {}
            for name, where in signature.get("types", {}).items():
                with contextlib.suppress(KeyError, TypeError, ValueError):
                    names[name] = self._custom_types[where["host"]] if "host" in where else _loaded_class(where["import"])
            # A type the host doesn't know, like one sandboxed code defined itself, leaves its field untyped: the host
            # builds no class from what the sandbox sends.
            sig = _resolve_signature({**signature, "signature": _known_field_types(signature["signature"], names)}, names)
            for name, extra in signature.get("extra", {}).items():
                if name in sig.fields and isinstance(extra, dict):
                    sig = sig.with_updated_fields(name, **{str(key): value for key, value in extra.items()})
            return sig
        except Exception:  # a signature the host can't read at all keeps its fields, untyped
            fields = signature.get("fields", {})
            return make_signature(
                {
                    **{name: (str, InputField(**fields.get(name, {}))) for name in signature.get("inputs", ())},
                    **{name: (str, OutputField(**fields.get(name, {}))) for name in signature.get("outputs", ())},
                },
                signature.get("instructions"),
            )

    def _resolve_lm(self) -> Any:
        return self._lm if self._lm is not None else dspy.settings.lm

    def _spec(self, session: str) -> dict[str, Any]:
        lm = self._resolve_lm()
        self._adapter = dspy.settings.adapter
        return {
            "host_token": _HOST_TOKEN,
            "session": session,
            "lm": _lm_spec(lm) if lm is not None else None,
            "tools": {name: _tool_spec(name, tool) for name, tool in self._tools.items()},
            "types": {name: _type_binding(t) for name, t in self._custom_types.items()},
            "execution_instructions": self._execution_instructions(),
            "trace": dspy.settings.trace is not None and dspy.settings.max_trace_size > 0,
        }

    def _execution_instructions(self) -> str:
        if self._interpreter_factory is None:
            return ""
        factory = resolve_interpreter_factory(self._interpreter_factory)
        return getattr(factory, "execution_instructions", "") or ""

    def _serve_lm(self, op: str, prompt: Any = None, options: dict[str, Any] | None = None) -> Any:
        """The ``LM_TOOL`` host tool: build a request (``op="request"``) or run a call (``op="call"``) on the host's
        LM, or run a predictor's whole adapter call (``op="adapter"``, see ``_run_adapter``). Every call runs the
        sync path, an async one in the sandbox too (see ``HostLM.acall``)."""
        call = next(self._calls)
        lm = None
        try:
            lm = self._resolve_lm()
            if lm is None:
                raise LMNotConfiguredError("No LM is configured on the host. Use dspy.configure(lm=...).")
            if isinstance(lm, SandboxLM) and self._reserve is None:
                sandbox_lm = lm
            else:
                sandbox_lm = SandboxLM(lm, self._reserve, self._built, refuse_host_files=True)
            options = decode_options(options or {})
            if op == "adapter":
                return self._run_adapter(prompt, options, sandbox_lm)
            prompt = decode_prompt(prompt)
            _refuse_host_files(prompt)
            if op == "request":
                return serde.request_to_dict(self._seal(sandbox_lm._build_adapter_request(prompt, options)))
            if op == "call":
                out = sandbox_lm(self._unseal(prompt), **options)
                return serde.response_to_dict(out) if isinstance(out, Response) else self._values.encode(out)
            raise CodeInterpreterError(f"Unknown sandbox LM operation {op!r}")
        except LMError as e:
            # The sandbox raises this as the matching dspy error; the caller re-raises the original.
            tag = f"[dspy bridge lm-error #{call}]"
            self.lm_error = (e, tag)
            return {LM_ERROR_KEY: {"type": type(e).__name__, "message": _redact(f"{tag} {e}", lm)}}

    def _run_adapter(self, call: dict[str, Any], options: dict[str, Any], lm: SandboxLM) -> Any:
        """A predictor call in the sandbox, run here with the host's adapter, as it would run without a sandbox: the
        adapter formats the request with the real types, calls the metered LM and parses the answer. The sandbox
        sends the signature, the demos and the inputs. An LM the adapter holds itself, like a TwoStepAdapter's
        extraction model, is metered too."""
        signature = self._signature(call["signature"])
        # Read as the declared types, as the user's arrive as plain data.
        inputs = {name: _typed(value, signature.input_fields.get(name)) for name, value in self._values.decode(call["inputs"]).items()}
        demos = [dspy.Example(**{name: _typed(value, signature.fields.get(name)) for name, value in self._values.decode(demo).items()})
                 for demo in call.get("demos", ())]
        adapter = self._adapter if self._adapter is not None else _get_adapter()
        if getattr(adapter, "extraction_model", None) is not None:
            adapter = copy.copy(adapter)
            adapter.extraction_model = SandboxLM(adapter.extraction_model, self._reserve, [], refuse_host_files=True)
        outputs = adapter(lm, lm_kwargs=options, signature=signature, demos=demos, inputs=inputs)
        return self._values.encode(outputs)

    def _seal(self, request: Any) -> Any:
        """``request`` as sandboxed code gets it: the settings the host LM owns, such as provider options that may hold
        credentials (``aws_secret_access_key``, ``vertex_credentials``...), stay on the host. The request carries
        their defaults and, in its extensions, which of the host's settings to restore (``_unseal``)."""
        owned = _host_owned(request.config)
        if owned not in self._sealed:
            self._sealed.append(owned)
        placeholder = {**_host_owned(dspy.lm15.Config()), "extensions": {_SEALED_KEY: self._sealed.index(owned)}}
        return dataclasses.replace(request, config=dataclasses.replace(request.config, **placeholder))

    def _unseal(self, prompt: Any) -> Any:
        """A request ``_seal`` gave the sandbox, with the host LM's settings back in place; other prompts as they are."""
        if not isinstance(prompt, dspy.lm15.Request) or _SEALED_KEY not in (prompt.config.extensions or {}):
            return prompt
        index = prompt.config.extensions[_SEALED_KEY]
        placeholder = {**_host_owned(dspy.lm15.Config()), "extensions": {_SEALED_KEY: index}}
        changed = sorted(name for name, value in _host_owned(prompt.config).items() if value != placeholder[name])
        if changed or type(index) is not int or not 0 <= index < len(self._sealed):
            raise TypeError(
                f"Sandboxed code may not change request setting(s) {changed or ['extensions']}; they come from the host LM."
            )
        return dataclasses.replace(prompt, config=dataclasses.replace(prompt.config, **self._sealed[index]))


def _redact(text: str, lm: Any) -> str:
    """``text`` without the host LM's settings, credentials among them, which a provider's error may echo."""
    settings: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str) and len(value) >= 8:
            settings.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    while lm is not None:  # through the proxies, to the LM that holds the settings
        collect(getattr(lm, "kwargs", None) or {})
        lm = getattr(lm, "_lm", None)
    for setting in sorted(set(settings), key=len, reverse=True):
        text = text.replace(setting, "***")
    return text


def _refuse_host_files(prompt: Any) -> None:
    """Sandboxed code may not have the host read its files: media that names a host path would otherwise be read,
    and sent to the provider, as the host LM prepares the request."""

    def check(parts: Any) -> None:
        for part in parts if isinstance(parts, (tuple, list)) else ():
            if getattr(part, "path", None) is not None:
                raise PermissionError(
                    "Sandboxed code may not reference host files in LM requests; pass the content inline instead."
                )
            check(getattr(part, "content", ()))  # a tool result's parts

    check(getattr(prompt, "system", None))
    for message in getattr(prompt, "messages", ()):
        check(message.parts)


def from_wire(value: Any, annotation: Any) -> Any:
    """Read the wire's restored Python values as ``annotation``, by field name. Fall back to JSON validation for
    custom types whose round-trip data requires it, then to dspy's LM-output parser."""
    adapter = pydantic.TypeAdapter(annotation)
    try:
        return adapter.validate_python(value, by_name=True)
    except Exception:
        try:
            return adapter.validate_json(json.dumps(value), by_name=True)
        except Exception:
            return parse_value(value, annotation)


def _as_declared(value: Any, annotation: Any) -> Any:
    """A tool argument as the type the tool declares, when it came as plain data (as a value of the user's types
    crosses), or as it came: the tool then checks it as sent."""
    if annotation in (None, Any) or not isinstance(value, (dict, list)):
        return value
    try:
        return from_wire(value, annotation)
    except Exception:
        return value


def _typed(value: Any, field: Any) -> Any:
    """``value`` as its field's declared type, or as is if it does not parse."""
    if field is None:
        return value
    try:
        return from_wire(value, field.annotation)
    except Exception:
        return value


# -- custom types ---------------------------------------------------------------------------------------------------


class HostValues(Wire):
    """The host's side of the wire for one session.

    A value of one of the user's custom types crosses as plain data: the JSON its type writes for a round trip,
    pydantic's own contract, which the host reads back as the type where a signature declares it (``from_wire``).
    Sandboxed code can name the type (``sandbox_guest.HostType``), but the type and its code stay here. An object of a
    class both sides have (dspy's, the standard library's) crosses as that class, with a reference to it kept here, so
    one that comes back unchanged is the host's own object again. Types only go one way: the host never builds a class
    from what the sandbox sends.
    """

    def __init__(self, tools: dict[str, Any] | None = None) -> None:
        self._tools = tools or {}  # the provided tools, which a tool crossing back may name
        self._objects: list[Any] = []
        self._refs: dict[int, int] = {}  # by id, the reference of each object sent

    def encode_enum(self, member: enum.Enum) -> Any:
        return self.encode(_jsonable(member)) if _import_path(type(member)) is None else super().encode_enum(member)

    def encode_object(self, value: Any, fields: dict[str, Any]) -> Any:
        if _import_path(type(value)) is None:  # the user's: plain data, as its type writes it for a round trip
            return self.encode(_plain_data(value))
        return {**super().encode_object(value, fields), "ref": self._keep(value)}

    def find_class(self, node: dict[str, Any]) -> Any:
        with contextlib.suppress(ValueError):  # only a class already loaded here, so nothing is imported
            return _loaded_class(node["import"])
        return None

    def decode_object(self, node: dict[str, Any]) -> Any:
        value = super().decode_object(node)  # a class both sides have, by its fields, or else its fields
        if "ref" in node:
            original = self.decode_ref(node)
            if type(original) is type(value) and original == value:  # unchanged: the host's own object
                return original
        return value

    def decode_ref(self, node: dict[str, Any]) -> Any:
        ref = node.get("ref")
        if type(ref) is not int or not 0 <= ref < len(self._objects):
            raise ValueError(f"The host sent no object with the reference {ref!r}.")
        return self._objects[ref]

    def decode_tool(self, node: dict[str, Any]) -> Tool:
        if "host" not in node:
            return super().decode_tool(node)
        if node["host"] not in self._tools:
            raise ValueError(f"No provided tool is named {node['host']!r}.")
        tool = self._tools[node["host"]]
        return tool if isinstance(tool, Tool) else Tool(tool)

    def _keep(self, value: Any) -> int:
        ref = self._refs.get(id(value))
        if ref is None or self._objects[ref] is not value:
            ref = self._refs[id(value)] = len(self._objects)
            self._objects.append(value)  # kept for the session, so its id stays its own
        return ref


def _plain_data(value: Any) -> Any:
    """``value`` as plain data, as its type writes it for a round trip, except that a ``SandboxSerializable`` it holds
    stays itself where the dump has its str, to cross in its own form: its str can't rebuild it. Where a serializer of
    the user's reads what one writes (a wrap serializer of its field, say), the dump is as before, from the str."""
    kept: dict[str, Any] = {}

    def keep(item: Any) -> str:  # a placeholder no serializer can turn into another
        placeholder = secrets.token_hex(16)
        kept[placeholder] = item
        return placeholder

    try:
        marked = pydantic_core.to_jsonable_python(
            value,
            by_alias=False,
            round_trip=True,
            context={KEEP_IN_DUMP: keep},  # in a model's fields
            fallback=lambda item: keep(item) if isinstance(item, SandboxSerializable) else str(item),  # in other values
        )
    except pydantic_core.PydanticSerializationError:
        marked = None  # a serializer that fails on a placeholder, or fails anyway
    if marked is not None and not kept:
        return marked  # holds no SandboxSerializable: the dump as before
    plain = _jsonable(value)

    def merge(marked: Any, plain: Any) -> Any:
        # A placeholder no serializer changed, where the dump as before has the value's str, is the value.
        if isinstance(marked, str) and marked in kept and plain == str(kept[marked]):
            return kept[marked]
        if isinstance(marked, list) and isinstance(plain, list) and len(marked) == len(plain):
            return [merge(m, p) for m, p in zip(marked, plain, strict=True)]
        if isinstance(marked, dict) and isinstance(plain, dict) and marked.keys() == plain.keys():
            return {key: merge(marked[key], plain[key]) for key in plain}
        return plain

    return plain if marked is None else merge(marked, plain)


def _type_binding(t: Any) -> dict[str, Any]:
    """How sandboxed code names the custom type ``t``: imported there if it can be, like dspy's types, or else by name
    only (``sandbox_guest.HostType``), described by the JSON schema of what crosses of it."""
    if (path := _import_path(t)) is not None:
        return {"import": path}
    try:
        return {"schema": _inline_refs(pydantic.TypeAdapter(t).json_schema(schema_generator=_CrossingSchema))}
    except Exception:  # a type pydantic can't describe
        return {"schema": {}}


class _CrossingSchema(GenerateJsonSchema):
    """A JSON schema of what crosses to the sandbox: without the fields marked ``exclude=True``, or their defaults."""

    def field_is_present(self, field: Any) -> bool:
        return not field.get("serialization_exclude") and super().field_is_present(field)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with its ``$defs`` written where they are used, so it can be part of another schema. A type that
    holds itself is written out once, and is an object where it holds itself."""
    definitions = schema.get("$defs", {})

    def inline(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [inline(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            rest = {key: value for key, value in node.items() if key != "$ref"}
            if name in definitions and name not in seen:
                return inline({**definitions[name], **rest}, seen | {name})
            return {"type": "object", "title": name, **rest}
        return {key: inline(value, seen) for key, value in node.items() if key != "$defs"}

    return inline(schema, frozenset())


def _in_sandbox(module: str) -> bool:
    """Whether the sandbox can import ``module``: the standard library's, or a package of dspy's own install."""
    top = module.split(".")[0]
    return (top in sys.stdlib_module_names and not top.startswith("_")) or top in sandbox_modules()


def _import_path(t: Any) -> str | None:
    """Where the sandbox imports ``t`` from, if it can (dspy's types, the standard library's, pydantic's...)."""
    module, name = getattr(t, "__module__", None), getattr(t, "__qualname__", "")
    importable = isinstance(name, str) and all(part.isidentifier() for part in name.split("."))  # no <locals>
    return f"{module}.{name}" if isinstance(module, str) and importable and _in_sandbox(module) else None


def _loaded_class(path: str) -> type:
    """The class ``path`` names in a module the host has already imported, as a type a signature recorded in the
    sandbox names (``dspy.Reasoning``, ``decimal.Decimal``, ``dspy.ToolCalls.ToolCall``...) resolves on the host:
    without importing anything."""
    parts = path.split(".")
    split = next((n for n in range(len(parts) - 1, 0, -1) if ".".join(parts[:n]) in sys.modules), 0)
    module = ".".join(parts[:split])
    # Read off the namespaces: getattr would run a module __getattr__, which may load things lazily.
    value: Any = vars(sys.modules[module]) if split and _in_sandbox(module) else {}
    for name in parts[split:]:
        value = value.get(name) if isinstance(value, dict) else vars(value).get(name) if inspect.isclass(value) else None
    if not inspect.isclass(value):
        raise ValueError(f"{path!r} names no class the host has loaded")
    return value


_FIELD_TYPE_NAMES = frozenset(
    {t.__name__ for t in (int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray)}
    | {name for name, value in vars(typing).items() if not name.startswith("_") and not isinstance(value, types.ModuleType)}
)


def _known_field_types(signature: str, names: dict[str, Any]) -> str:
    """A signature string sent by the sandbox, without the type of a field that names anything but builtin and
    ``typing`` types and ``names``, or uses other syntax than names, subscripts, ``|`` and the values of a ``Literal``.
    ``make_signature`` resolves any other name by importing a module of that name, and an attribute by reading it, on
    the host. A quoted type would reach pydantic as a forward reference, which it evaluates as Python on the host."""
    allowed = _FIELD_TYPE_NAMES.union(names)
    # A Literal's values are values, not types: only the typing one, not a class the sandbox calls Literal.
    literal = names.get("Literal", typing.Literal) is typing.Literal
    sides = []
    for side in signature.split("->"):
        source = f"def f({side}): pass"
        fields = []
        for arg in ast.parse(source).body[0].args.args:  # the fields, as make_signature parses them
            known = arg.annotation is not None and _is_field_type(arg.annotation, allowed, literal)
            fields.append(f"{arg.arg}: {ast.get_source_segment(source, arg.annotation)}" if known else arg.arg)
        sides.append(", ".join(fields))
    return " -> ".join(sides)


def _is_field_type(node: ast.expr, allowed: frozenset[str], literal: bool, values: bool = False) -> bool:
    if isinstance(node, ast.Name):
        return node.id in allowed
    if isinstance(node, ast.Constant):
        return values or node.value is None or node.value is Ellipsis
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):  # Literal[-1]
        return values and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    if isinstance(node, ast.Tuple):
        return all(_is_field_type(item, allowed, literal, values) for item in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_field_type(node.left, allowed, literal) and _is_field_type(node.right, allowed, literal)
    if isinstance(node, ast.Subscript):
        of_literal = literal and isinstance(node.value, ast.Name) and node.value.id == "Literal"
        return _is_field_type(node.value, allowed, literal) and _is_field_type(node.slice, allowed, literal, of_literal)
    return False


def _lm_spec(lm: Any) -> dict[str, Any]:
    proxy = lm if isinstance(lm, SandboxLM) else SandboxLM(lm)
    return {
        "model": proxy.model,
        "model_type": proxy.model_type,
        "kwargs": encode_options(proxy.kwargs),
        "capabilities": {
            "supports_function_calling": bool(proxy.supports_function_calling),
            "supports_reasoning": bool(proxy.supports_reasoning),
            "supports_response_schema": bool(proxy.supports_response_schema),
            "supported_params": sorted(proxy.supported_params or ()),
            "supported_tool_result_media": sorted(proxy.supported_tool_result_media or ()),
        },
    }


def _tool_spec(name: str, tool: Any) -> dict[str, Any]:
    tool = tool if isinstance(tool, Tool) else Tool(tool)
    return {"name": name, "desc": tool.desc or "", "args": tool.args, "has_kwargs": tool.has_kwargs}
