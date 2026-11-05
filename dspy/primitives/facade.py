"""Host side of the sandbox dspy bridge.

Sandboxed code (a ``dspy.Flex`` module or a ``dspy.RLM`` REPL block) imports the real dspy in its interpreter.
``SandboxSession.install`` makes dspy importable there (``PythonInterpreter.provide_dspy``) and runs ``BOOTSTRAP``,
which installs ``dspy.primitives.sandbox_guest`` as soon as sandboxed code imports dspy, with the host's adapter and
the shapes of the custom types (``types_spec``). From then on, only JSON crosses the boundary, over the interpreter's
tool-call protocol (``CodeInterpreter.tools``), and values cross as ``encode_value`` writes them and ``from_wire``
reads them:

- ``LM_TOOL``: every LM call made in the sandbox. The host builds the request from its own LM on a ``SandboxLM``,
  which admits only generation options (``SANDBOX_LM_KWARGS``) and meters each completion, then runs it. The model,
  credentials, cache, retries and history stay on the host.
- Provided tools, registered by name; the sandbox binds them as ``dspy.Tool``s with their host metadata.
- ``INTERPRETER_TOOL``: a fresh interpreter from ``interpreter_factory`` for each code-executing module built in the
  sandbox, so its code shares nothing with the sandbox that built it.

An interpreter that runs code in the host's own memory (an in-process interpreter) uses the host's dspy as is:
the bootstrap leaves it alone, and the caller scopes its execution with ``SandboxSession.scope``.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import contextvars
import copy
import dataclasses
import enum
import functools
import inspect
import io
import itertools
import logging
import math
import pickle
import queue
import re
import secrets
import sys
import types
import typing
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pydantic
import pydantic_core

import dspy
from dspy._vendor.lm15 import serde
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value
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
    _CONFIG_KEYS,
    INTERPRETER_TOOL,
    LM_ERROR_KEY,
    LM_TOOL,
    TRACE_TOOL,
    TYPING_NAMES,
    HostInterpreter,
    decode_options,
    decode_prompt,
    encode_options,
    encode_value,
    validate_wire,
)
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

    def __init__(self, interpreter: Any, host_tools: dict[str, Callable[..., Any]]) -> None:
        self.interpreter = interpreter
        self._host_tools = host_tools
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
            return {"final": encode_value(value.output)}
        return {"output": encode_value(value)}

    def _parent_call(self, name: str, spec: dict) -> Callable[..., Any]:
        """A stand-in for a parent-sandbox function, with its signature, that pauses the execution to call it."""

        def call(*args: Any, **kwargs: Any) -> Any:
            self._events.put(("call", {"name": name, "args": encode_value(list(args)), "kwargs": encode_value(kwargs)}))
            reply = self._replies.get()
            if reply is _CancelCall:
                raise CodeInterpreterError(f"The sandbox that owns {name!r} stopped before answering the call.")
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply["value"]

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
    shuts down any still open.
    """

    def __init__(
        self,
        tools: dict[str, Any],
        *,
        custom_types: dict[str, type] | None = None,
        inputs: dict[str, str] | None = None,
        lm: Any = None,
        reserve: Callable[[int], None] | None = None,
        interpreter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._tools = tools
        self._custom_types = custom_types or {}
        self._inputs = inputs or {}
        self._lm = lm
        self._reserve = reserve
        self._interpreter_factory = interpreter_factory
        self._nested: dict[str, _NestedInterpreter] = {}
        self._extraction_lm: Any = None
        self._adapter: Any = None
        self._calls = itertools.count(1)
        self._built: dict[str | None, list[dict[str, Any]]] = {}  # per LM role, see SandboxLM
        self._sealed: dict[str | None, list[dict[str, Any]]] = {}  # per LM role, see _seal
        self.lm_error: tuple[LMError, str] | None = None
        self.in_host = False

    def install(self, interpreter: Any, *, import_dspy: bool) -> None:
        """Bootstrap the bridge in ``interpreter``; with ``import_dspy``, import dspy there right away."""
        provide = getattr(interpreter, "provide_dspy", None)
        if callable(provide):
            provide()
        interpreter.tools[LM_TOOL] = self._serve_lm
        interpreter.tools[TRACE_TOOL] = self._serve_trace
        # Inside a sandbox, the host already gave this interpreter the host's own interpreter tool.
        if self._interpreter_factory is not None and not isinstance(interpreter, HostInterpreter):
            interpreter.tools[INTERPRETER_TOOL] = self._serve_interpreter
        session = secrets.token_hex(8)
        spec = self._spec(session)
        interpreter.execute(BOOTSTRAP, variables={"__dspy_spec": spec})
        namespace = _in_host.pop(session, None)
        if namespace is not None:
            self.in_host = True
            namespace["__dspy_guest_spec"] = spec
            namespace["__dspy_types"] = dict(self._custom_types)
            namespace.update({**{k: v for k, v in TYPING_NAMES.items() if k not in namespace}, **self._custom_types})
        if import_dspy:
            interpreter.execute("import dspy")

    def scope(self) -> contextlib.AbstractContextManager[Any]:
        """Run code of an interpreter that shares the host's memory on the metered LM; a no-op otherwise."""
        lm = self._resolve_lm()
        if not self.in_host or lm is None:
            return contextlib.nullcontext()
        factory = {"interpreter_factory": self._interpreter_factory} if self._interpreter_factory else {}
        return dspy.context(lm=SandboxLM(lm, self._reserve, self._built.setdefault(None, [])), **factory)

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
            self._nested[handle] = _NestedInterpreter(interpreter, host_tools)
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
            return nested.execute(code or "", variables or {}, tools or {}, output_fields)
        if op == "resume":
            return nested.resume(reply or {})
        if op == "shutdown":
            self._nested.pop(id).close()
            return None
        raise CodeInterpreterError(f"Unknown nested interpreter operation {op!r}")

    def _serve_trace(self, entry: dict[str, Any]) -> None:
        """The ``TRACE_TOOL`` host tool: add a predictor call recorded in the sandbox to the host's trace, with the
        predictor rebuilt from its signature and its values parsed back to their declared types."""
        trace = dspy.settings.trace
        if trace is None or dspy.settings.max_trace_size <= 0:
            return
        signature = self._signature(entry["signature"]) if entry.get("signature") else None
        inputs, outputs = entry["inputs"], entry.get("outputs") or {}
        if signature is not None:
            inputs = {name: _typed(value, signature.input_fields.get(name)) for name, value in inputs.items()}
            outputs = {name: _typed(value, signature.output_fields.get(name)) for name, value in outputs.items()}
        if len(trace) >= dspy.settings.max_trace_size:
            trace.pop(0)
        trace.append((dspy.Predict(signature) if signature is not None else None, inputs, dspy.Prediction(**outputs)))

    def _signature(self, signature: dict[str, Any]) -> Any:
        try:
            names = dict(self._custom_types)
            for name, path in signature.get("types", {}).items():
                with contextlib.suppress(ValueError):
                    names.setdefault(name, _loaded_class(path))
            # A type the host doesn't know, like one sandboxed code defined itself, leaves its field untyped: the host
            # builds no class from what the sandbox sends.
            return _resolve_signature({**signature, "signature": _known_field_types(signature["signature"], names)}, names)
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
        adapter, adapter_types = _adapter_spec(dspy.settings.adapter)
        self._adapter = dspy.settings.adapter
        self._extraction_lm = getattr(dspy.settings.adapter, "extraction_model", None)
        return {
            "host_token": _HOST_TOKEN,
            "session": session,
            "lm": _lm_spec(lm) if lm is not None else None,
            "adapter": adapter,
            "tools": {name: _tool_spec(name, tool) for name, tool in self._tools.items()},
            "types": types_spec({**adapter_types, **self._custom_types}),
            "inputs": self._inputs,
            "execution_instructions": self._execution_instructions(),
            "trace": dspy.settings.trace is not None and dspy.settings.max_trace_size > 0,
        }

    def _execution_instructions(self) -> str:
        if self._interpreter_factory is None:
            return ""
        factory = resolve_interpreter_factory(self._interpreter_factory)
        return getattr(factory, "execution_instructions", "") or ""

    def _serve_lm(
        self,
        op: str,
        prompt: Any = None,
        options: dict[str, Any] | None = None,
        role: str | None = None,
    ) -> Any:
        """The ``LM_TOOL`` host tool: build a request (``op="request"``) or run a call (``op="call"``), on the host's
        LM, or with ``role="extraction"`` on the extraction model of the host's TwoStepAdapter. With ``op="adapter"``,
        run a predictor's whole adapter call on the host, for an adapter class of the user's (``_run_adapter``). Every
        call runs the sync path, an async one in the sandbox too (see ``HostLM.acall``)."""
        call = next(self._calls)
        lm = None
        try:
            lm = self._extraction_lm if role == "extraction" else self._resolve_lm()
            if lm is None:
                raise LMNotConfiguredError("No LM is configured on the host. Use dspy.configure(lm=...).")
            if isinstance(lm, SandboxLM) and self._reserve is None:
                sandbox_lm = lm
            else:
                sandbox_lm = SandboxLM(lm, self._reserve, self._built.setdefault(role, []), refuse_host_files=True)
            prompt, options = decode_prompt(prompt), decode_options(options or {})
            if op == "adapter":
                return encode_value(self._run_adapter(prompt, options, sandbox_lm))
            _refuse_host_files(prompt)
            if op == "request":
                return serde.request_to_dict(self._seal(sandbox_lm._build_adapter_request(prompt, options), role))
            if op == "call":
                out = sandbox_lm(self._unseal(prompt, role), **options)
                return serde.response_to_dict(out) if isinstance(out, Response) else encode_value(out)
            raise CodeInterpreterError(f"Unknown sandbox LM operation {op!r}")
        except LMError as e:
            # The sandbox raises this as the matching dspy error; the caller re-raises the original.
            tag = f"[dspy bridge lm-error #{call}]"
            self.lm_error = (e, tag)
            return {LM_ERROR_KEY: {"type": type(e).__name__, "message": _redact(f"{tag} {e}", lm)}}

    def _run_adapter(self, call: dict[str, Any], options: dict[str, Any], lm: SandboxLM) -> list[dict[str, Any]]:
        """A predictor call in the sandbox, run with the user's own adapter here, as it would run without a sandbox:
        the adapter formats the request, calls the metered LM and parses the answer. The sandbox sends the signature,
        the demos and the inputs, which are read back to their declared types first."""
        signature = self._signature(call["signature"])
        inputs = {name: _typed(value, signature.input_fields.get(name)) for name, value in call["inputs"].items()}
        demos = [dspy.Example(**{name: _typed(value, signature.fields.get(name)) for name, value in demo.items()})
                 for demo in call.get("demos", ())]
        return self._adapter(lm, lm_kwargs=options, signature=signature, demos=demos, inputs=inputs)

    def _seal(self, request: Any, role: str | None) -> Any:
        """``request`` as sandboxed code gets it: the settings the host LM owns, such as provider options that may hold
        credentials (``aws_secret_access_key``, ``vertex_credentials``...), stay on the host. The request carries
        their defaults and, in its extensions, which of the host's settings to restore (``_unseal``)."""
        sealed = self._sealed.setdefault(role, [])
        owned = _host_owned(request.config)
        if owned not in sealed:
            sealed.append(owned)
        placeholder = {**_host_owned(dspy.lm15.Config()), "extensions": {_SEALED_KEY: sealed.index(owned)}}
        return dataclasses.replace(request, config=dataclasses.replace(request.config, **placeholder))

    def _unseal(self, prompt: Any, role: str | None) -> Any:
        """A request ``_seal`` gave the sandbox, with the host LM's settings back in place; other prompts as they are."""
        if not isinstance(prompt, dspy.lm15.Request) or _SEALED_KEY not in (prompt.config.extensions or {}):
            return prompt
        sealed = self._sealed.get(role, [])
        index = prompt.config.extensions[_SEALED_KEY]
        placeholder = {**_host_owned(dspy.lm15.Config()), "extensions": {_SEALED_KEY: index}}
        changed = sorted(name for name, value in _host_owned(prompt.config).items() if value != placeholder[name])
        if changed or type(index) is not int or not 0 <= index < len(sealed):
            raise TypeError(
                f"Sandboxed code may not change request setting(s) {changed or ['extensions']}; they come from the host LM."
            )
        return dataclasses.replace(prompt, config=dataclasses.replace(prompt.config, **sealed[index]))


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
    """``value``, as ``encode_value`` sent it, read back as ``annotation``: by pydantic, which reads models by field
    name, or failing that as dspy parses an LM's output."""
    try:
        return validate_wire(pydantic.TypeAdapter(annotation), value)
    except Exception:
        return parse_value(value, annotation)


def _typed(value: Any, field: Any) -> Any:
    """``value`` as its field's declared type, or as is if it does not parse."""
    if field is None:
        return value
    try:
        return from_wire(value, field.annotation)
    except Exception:
        return value


# -- custom types ---------------------------------------------------------------------------------------------------


def types_spec(custom_types: dict[str, type]) -> dict[str, Any]:
    """How the sandbox gets the signature's custom types: as their shape, never their code.

    dspy's and the standard library's types are imported there. A pydantic model, dataclass, ``NamedTuple`` or
    ``TypedDict`` of the user's crosses as its fields, with their types, defaults and constraints, and an enum as its
    members, all read off the live class. The sandbox builds plain classes of the
    same names and shapes from that, so sandboxed code can read and build them. Their methods and validators stay on
    the host, which validates with the real types whatever the sandbox sends back.
    """
    shapes = _Shapes()
    return {"types": {name: shapes.ref(t) for name, t in custom_types.items()}, "shapes": shapes.out}


_ABSENT = object()


class _Shapes:
    """The shapes of user types, by a key per class, as ``types_spec`` sends them."""

    def __init__(self) -> None:
        self.out: dict[str, dict[str, Any]] = {}
        self._keys: dict[type, str] = {}
        self._owner: type | None = None  # the class whose fields are being described

    def ref(self, annotation: Any) -> dict[str, Any]:
        """``annotation`` as data the sandbox rebuilds it from (see ``sandbox_guest.rebuild_types``)."""
        if isinstance(annotation, (str, typing.ForwardRef)):
            annotation = self._forward(annotation)
        if annotation is Any or annotation is object:
            return {"any": True}
        if annotation is None or annotation is type(None):
            return {"none": True}
        origin, args = typing.get_origin(annotation), typing.get_args(annotation)
        if origin is typing.Literal:  # an enum member as its enum and name
            return {"literal": [{"member": self.ref(type(arg)), "name": arg.name} if isinstance(arg, enum.Enum)
                                else encode_value(arg) for arg in args]}
        if origin is typing.Annotated:
            return {"annotated": self.ref(args[0]), "metadata": [m for m in map(self._constraint, args[1:]) if m]}
        if origin in (typing.Union, types.UnionType):
            return {"union": [self.ref(arg) for arg in args]}
        if origin is not None:
            path = _import_path(origin)
            if path is None:
                return {"any": True}
            return {"generic": path, "args": [{"ellipsis": True} if arg is Ellipsis else self.ref(arg) for arg in args]}
        if not isinstance(annotation, type):
            return {"any": True}  # a TypeVar, an unresolved forward reference...
        if (path := _import_path(annotation)) is not None:
            return {"import": path}
        return {"shape": self._key(annotation)}

    def _forward(self, reference: Any) -> Any:
        """A forward reference left in an annotation, like ``"Node"`` in ``list["Node"]`` (pydantic keeps some on
        Python 3.10): a bare name is looked up in the module of the class it is in, or is that class. Nothing is run."""
        name = (reference.__forward_arg__ if isinstance(reference, typing.ForwardRef) else reference).strip()
        owner = self._owner
        if owner is None or not name.isidentifier():
            return Any
        found = owner if name == owner.__name__ else vars(sys.modules.get(owner.__module__, types)).get(name)
        return found if isinstance(found, type) else Any

    def _key(self, cls: type) -> str:
        if cls not in self._keys:
            key = base = re.sub(r"\W", "_", cls.__name__)
            while key in self.out:  # another class of the same name
                key = f"{base}_{len(self.out)}"
            self._keys[cls] = key
            self.out[key] = {}  # taken, for a type that holds itself
            owner, self._owner = self._owner, cls
            try:
                shape = {"name": cls.__name__, **self._shape(cls)}
            finally:
                self._owner = owner
            del self.out[key]
            self.out[key] = shape  # listed after the types it holds, so the sandbox builds those first
        return self._keys[cls]

    def _shape(self, cls: type) -> dict[str, Any]:
        if issubclass(cls, enum.Enum):
            base = next((kind.__name__ for kind in (str, int) if issubclass(cls, kind)), None)
            members = {name: encode_value(member.value) for name, member in cls.__members__.items()}
            return {"kind": "flag" if issubclass(cls, enum.Flag) else "enum", "base": base, "members": members}
        if issubclass(cls, pydantic.BaseModel):
            doc = vars(cls).get("__doc__")
            fields = {name: self._field(f.annotation, f) for name, f in cls.model_fields.items()}
            kind = "root" if issubclass(cls, pydantic.RootModel) else "model"
            return {"kind": kind, "doc": doc if isinstance(doc, str) else None, "config": _config(cls.model_config),
                    "fields": fields}
        hints = _hints(cls)
        if dataclasses.is_dataclass(cls):
            infos = getattr(cls, "__pydantic_fields__", None) or {}  # a pydantic dataclass's, as complete as a model's
            fields = {}
            for f in dataclasses.fields(cls):
                info = infos.get(f.name) or (f.default if isinstance(f.default, pydantic.fields.FieldInfo) else None)
                fields[f.name] = self._field(hints.get(f.name, Any), info, f.default, f.default_factory)
                if getattr(f, "kw_only", False) is True:  # as resolved for the field, the class's setting included
                    fields[f.name]["kw_only"] = True
            config = _config(getattr(cls, "__pydantic_config__", {}))  # a pydantic dataclass's settings, as a model's
            return {"kind": "dataclass", "frozen": cls.__dataclass_params__.frozen, "config": config, "fields": fields}
        if issubclass(cls, tuple) and hasattr(cls, "_fields"):
            defaults = getattr(cls, "_field_defaults", {})
            fields = {name: self._field(hints.get(name, Any), None, defaults.get(name, _ABSENT)) for name in cls._fields}
            return {"kind": "namedtuple", "fields": fields}
        if typing.is_typeddict(cls):
            return {"kind": "typeddict", "required": sorted(cls.__required_keys__),
                    "fields": {name: {"type": self.ref(hint)} for name, hint in hints.items()}}
        return {"kind": "opaque"}  # a class pydantic can't validate: values of it cross as what they encode to

    def _field(self, annotation: Any, info: Any = None, default: Any = _ABSENT, factory: Any = None) -> dict:
        """A field's shape: its type, constraints, description and aliases, and its default. A default that only the
        host can make, with a factory of the user's, is marked, and the host makes it when the value comes back."""
        field: dict[str, Any] = {"type": self.ref(annotation)}
        if info is not None:  # a pydantic field's settings
            default, factory = info.default, info.default_factory
            if constraints := [c for c in map(self._constraint, info.metadata) if c]:
                field["metadata"] = constraints
            if isinstance(info.description, str):
                field["description"] = info.description
            for key in ("alias", "serialization_alias"):
                if isinstance(value := getattr(info, key), str):
                    field[key] = value
            if info.validation_alias is not None and (alias := _alias(info.validation_alias)) is not None:
                field["validation_alias"] = alias
        if default in (dataclasses.MISSING, pydantic_core.PydanticUndefined) or isinstance(default, pydantic.fields.FieldInfo):
            default = _ABSENT
        if factory in (None, dataclasses.MISSING):
            factory = None
        if info is not None and info.exclude is True:
            field["exclude"] = True  # it never crosses either way, nor does a default it has, which may be private
            if default is not _ABSENT and (not _is_literal(default) or default):
                default, factory = None, None
        if default is not _ABSENT:
            field["default"] = encode_value(default)
        elif factory is not None:
            path = _import_path(factory)
            field.update({"factory": path} if path is not None else {"host_default": True})
        return field

    def _constraint(self, metadata: Any) -> dict[str, Any] | None:
        """A constraint pydantic keeps on a field, like ``annotated_types.Gt(5)`` or a pattern, as data. ``None`` for
        one the other side can't rebuild, like a validator function, which still applies where the type lives."""
        if isinstance(metadata, type):  # a marker class, like pydantic.Json
            path = _import_path(metadata)
            return {"import": path, "values": None} if path is not None else None
        path = _import_path(type(metadata))
        if dataclasses.is_dataclass(metadata):
            values = {f.name: getattr(metadata, f.name) for f in dataclasses.fields(metadata)}
        elif type(metadata).__name__ == "_PydanticGeneralMetadata":  # what Field(pattern=...) and the like keep
            values, path = dict(vars(metadata)), "pydantic.Field"
        elif not getattr(metadata, "__dict__", None) and not getattr(type(metadata), "__slots__", ()):
            values = {}  # a marker with no settings, like pydantic.Json()
        else:
            return None
        if path is None or not all(_is_literal(value) for value in values.values()):
            return None
        return {"import": path, "values": values}


def _config(settings: Any) -> dict[str, Any]:
    """The settings of a pydantic model or dataclass that decide what its values look like on the wire."""
    return {key: value for key in _CONFIG_KEYS if _is_literal(value := settings.get(key)) and value is not None}


def _alias(alias: Any) -> Any:
    """A validation alias as data: a name, an ``AliasPath`` as its path, an ``AliasChoices`` as its choices."""
    if isinstance(alias, str):
        return alias
    if isinstance(alias, pydantic.AliasPath):
        return {"path": list(alias.path)}
    if isinstance(alias, pydantic.AliasChoices):
        choices = [_alias(choice) for choice in alias.choices]
        return {"choices": choices} if None not in choices else None
    return None


def _hints(cls: type) -> dict[str, Any]:
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except Exception:
        return {}


def _is_literal(value: Any, _within: tuple[int, ...] = ()) -> bool:
    """Whether ``value`` is plain JSON-like data: builtin scalars, also inside builtin containers."""
    if type(value) in (tuple, list, set, frozenset, dict):
        if id(value) in _within:  # a container that holds itself
            return False
        items = [part for pair in value.items() for part in pair] if type(value) is dict else value
        return all(_is_literal(item, (*_within, id(value))) for item in items)
    if type(value) is float:
        return math.isfinite(value)
    return value is None or type(value) in (bool, int, str, bytes)


def _is_user_object(value: Any) -> bool:
    """A class or function of the user's, which the sandbox cannot import."""
    return (inspect.isclass(value) or inspect.isfunction(value)) and not _in_sandbox(value.__module__)


def _in_sandbox(module: str) -> bool:
    """Whether the sandbox can import ``module``: the standard library's, or a package of dspy's own install."""
    top = module.split(".")[0]
    return (top in sys.stdlib_module_names and not top.startswith("_")) or top in sandbox_modules()


def _import_path(t: Any) -> str | None:
    """Where the sandbox imports ``t`` from, if it can (dspy's types, the standard library's, pydantic's...)."""
    module, name = getattr(t, "__module__", None), getattr(t, "__qualname__", "")
    return f"{module}.{name}" if isinstance(module, str) and name.isidentifier() and _in_sandbox(module) else None


def _loaded_class(path: str) -> type:
    """The class ``path`` names in a module the host has already imported, as a type a signature recorded in the
    sandbox names (``dspy.Reasoning``, ``decimal.Decimal``...) resolves on the host: without importing anything."""
    module, _, name = path.rpartition(".")
    # Read off the module's namespace: getattr would run a module __getattr__, which may load things lazily.
    value = vars(sys.modules[module]).get(name) if _in_sandbox(module) and module in sys.modules else None
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
        },
    }


def _adapter_spec(adapter: Any) -> tuple[dict[str, Any] | None, dict[str, type]]:
    """dspy's own adapters cross pickled, with their settings: without their callbacks, which stay on the host, with
    a TwoStepAdapter's extraction model replaced by one the host serves, and with the user's types they name (such
    as native response types) built in the sandbox like the signature's. An adapter class of the user's stays on the
    host and runs there, as it would without a sandbox (see ``SandboxSession._run_adapter``). Returns the spec and the
    user's types it names."""
    if adapter is None:
        return None, {}
    if _is_user_object(type(adapter)):
        return {"host": True}, {}
    adapter = copy.copy(adapter)
    adapter.callbacks = []
    extraction = getattr(adapter, "extraction_model", None)
    named: dict[str, type] = {}

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj: Any) -> str | None:
            if extraction is not None and obj is extraction:
                return "extraction_lm"
            if isinstance(obj, BaseLM):  # another LM would take its credentials along
                raise TypeError(f"{type(adapter).__name__} holds an LM besides its extraction model")
            if isinstance(obj, type) and _is_user_object(obj):
                named[obj.__name__] = obj
                return f"type:{obj.__name__}"
            return None

    buffer = io.BytesIO()
    try:
        Pickler(buffer).dump(adapter)
    except Exception as e:
        logger.warning("Sandboxed predictors use the default adapter, not %s: %s", type(adapter).__name__, e)
        return None, {}
    spec: dict[str, Any] = {"pickle": base64.b64encode(buffer.getvalue()).decode()}
    if extraction is not None:
        spec["extraction_lm"] = _lm_spec(extraction)
    return spec, named


def _tool_spec(name: str, tool: Any) -> dict[str, Any]:
    tool = tool if isinstance(tool, Tool) else Tool(tool)
    return {"name": name, "desc": tool.desc or "", "args": tool.args, "has_kwargs": tool.has_kwargs}
