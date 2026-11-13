from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from typing import Any, Callable

from dspy.adapters.types.tool import Tool
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.settings import settings
from dspy.predict.flex.bridge import BridgeRuntime
from dspy.predict.flex.ctx import FlexContext
from dspy.predict.parameter import Parameter
from dspy.predict.predict import _sanitize_lm_state
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, _validate_interpreter_factory
from dspy.primitives.libraries import spec_of
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.signatures.signature import Signature, ensure_signature

logger = logging.getLogger(__name__)

# Asks the interpreter about its own environment: what it reads there is what the code will run against.
_ASK = "from dspy.primitives import libraries as __dspy_libraries\n__dspy_libraries.{question}(__dspy_argument)"
# The recorded environments this process has compared with its own, so that each is compared, and reported, once.
_checked: set[str] = set()
_checked_lock = threading.Lock()


class Flex(Module, Parameter):
    """A module whose implementation is optimizable code, not just a prompt.

    Construct it like any module (``dspy.Flex(MySignature)``). It starts as a baseline that delegates
    to a single ``dspy.Predict`` over the signature — or ``dspy.RLM`` when ``tools`` are given, so the
    baseline can call them. ``dspy.GEPA`` recognizes ``Flex`` instances by type and rewrites their
    source — a single ``dspy.Module`` subclass, exposed as ``module_src`` — into decomposed predictors
    plus plain Python instead of only tuning instructions.

    The optimizer-authored code runs inside an interpreter. ``Flex`` never
    runs it in the host Python process. ``interpreter_factory`` defaults to
    ``dspy.PythonInterpreter`` (Deno/Pyodide) and must be a zero-argument callable returning a new
    ``CodeInterpreter``. ``dspy.configure(interpreter_factory=...)`` replaces that default.
    The source imports the real dspy in the interpreter; a custom interpreter therefore defines the
    Python and standard-library subset available to it, and must let it import the host's dspy.
    The optimizer-authored code runs isolated; only provided-tool calls and LM calls go back to the
    host, which makes the real LM calls.

    The code can also import your own code, such as your project's abstractions or an SDK: pass it as ``libraries``.
    The interpreter makes them importable, the optimizer gets their API as the interpreter has them, next to DSPy's
    modules, and writes code that uses what fits. What the code was validated against is recorded with it
    (``environment``): each package's version and a hash of each of your own source files the libraries reach, read
    in the interpreter. Running it next to something else warns once, and ``check_environment`` lists what differs.

    Args:
        signature: A ``dspy.Signature`` class or string declaring inputs/outputs.
        tools: ``dspy.Tool`` instances or named callables.
        libraries: What the code may import besides dspy and the standard library: modules, or single classes and
            functions, to point it at one part of a large package. A name works too (``"shopkit.pricing"``, or
            ``"shopkit.cart:Cart"`` for a class), for a library only the interpreter has. A value of a library's
            class is an object of that class in the interpreter. Whatever needs your network, credentials or state
            stays a tool.
        module_src: Source to start from instead of the baseline, such as code that only uses ``libraries``.
        interpreter_factory: Zero-argument callable returning a fresh ``CodeInterpreter`` for each
            interpreter session. Defaults to ``dspy.PythonInterpreter`` (sandbox, requires Deno).
            ``dspy.configure(interpreter_factory=...)`` replaces the default.
        max_predictor_calls: Maximum number of LM calls the optimizer-authored code can make per
            ``forward``. ``None`` removes the limit.
    """

    def __init__(
        self,
        signature: Any,
        *,
        tools: list[Any] | None = None,
        libraries: list[Any] | None = None,
        module_src: str | None = None,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
        max_predictor_calls: int | None = 100,
    ):
        super().__init__()

        self._signature_cls = ensure_signature(signature)
        self._name = getattr(self._signature_cls, "__name__", None) or "Flex"
        self._flex_ctx = FlexContext(
            signature_cls=self._signature_cls,
            tools=list(tools or []),
            libraries=list(dict.fromkeys(spec_of(library) for library in libraries or [])),
        )

        self._module_src: str | None = None
        self._environment: dict[str, Any] | None = None
        self.lm = None

        _validate_interpreter_factory(interpreter_factory)
        self._interpreter_factory = interpreter_factory
        self._max_predictor_calls = max_predictor_calls

        self._rebuild_bridge()
        self._bind_code(module_src or self._baseline_src())

    def _rebuild_bridge(self) -> None:
        self._bridge: BridgeRuntime = BridgeRuntime(self, self._interpreter_factory, self._max_predictor_calls)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("_bridge", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault("_environment", None)  # a module pickled before it recorded one
        self._flex_ctx.__dict__.setdefault("libraries", [])
        self._rebuild_bridge()
        if self._module_src is not None:
            self._bridge.bind(self._module_src)

    @property
    def signature(self) -> type[Signature]:
        return self._signature_cls

    @property
    def module_src(self) -> str | None:
        return self._module_src

    @property
    def environment(self) -> dict[str, Any] | None:
        """What this module's code was validated against, or ``None`` if nothing is recorded: the interpreter's
        Python version, each package the libraries reach with its version there (``requires``), a hash of each source
        file of your own code they reach (``sources``), and a fingerprint of each tool (``tools``).

        ``dspy.GEPA`` records it for the code it returns, and ``pin`` records it for the code as it is now. It is
        saved with the module and forgotten when the code changes.
        """
        return copy.deepcopy(self._environment)

    def pin(self) -> dict[str, Any] | None:
        """Record the environment the code runs in now as the one it was validated against, and return it.

        Call it once your evaluation passes: after writing or editing ``module_src`` yourself, or after a change
        that ``check_environment`` reported turned out to be harmless. With ``libraries``, this asks an interpreter.
        """
        self._environment = self._resolve_environment() or None
        if self._environment:
            with _checked_lock:
                _checked.add(self._environment_key())
        return self.environment

    def check_environment(self) -> list[str]:
        """What differs between the recorded environment and the one the code runs in now, in words, such as
        ``"shopkit/vat.py changed"`` or ``"regex is 2026.7.19, was 2024.11.6"``. Empty if nothing differs or nothing
        is recorded. With ``libraries``, this asks an interpreter.

        The first call of the module after loading does the same and warns. In CI, assert that this is empty.
        """
        return self._changes()

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        state = {
            "module_src": self._module_src,
            "lm": self.lm.dump_state() if self.lm else None,
        }
        if self._environment:
            state["environment"] = copy.deepcopy(self._environment)
        return state

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        module_src = state.get("module_src")
        if module_src:
            self._bind_code(module_src)
        lm_state = state.get("lm")
        if lm_state:
            sanitized = _sanitize_lm_state(lm_state, allow_unsafe_lm_state)
            self.lm = (
                BaseLM.load_state(sanitized, allow_custom_lm_class=allow_unsafe_lm_state) if sanitized else None
            )
        else:
            self.lm = None
        environment = state.get("environment")
        self._environment = copy.deepcopy(environment) if isinstance(environment, dict) and environment else None

    def _ask(self, question: str, argument: Any, interpreter: Any = None) -> str:
        """What ``dspy.primitives.libraries`` answers to ``question`` in the interpreter this module's code runs in:
        in ``interpreter``, one set up for a forward, or else in a new one."""
        code, variables = _ASK.format(question=question), {"__dspy_argument": argument}
        if interpreter is None:
            answer = self._bridge.run(code, variables)
        else:
            answer = interpreter.execute(code, variables=variables)
        if not isinstance(answer, str):
            raise CodeInterpreterError(f"The interpreter did not describe its environment (it returned {answer!r}).")
        return answer

    def _library_reference(self) -> str | None:
        """The libraries' API as the interpreter has them, for the code optimizer."""
        libraries = self._flex_ctx.libraries
        return self._ask("reference", libraries) if libraries else None

    def _resolve_environment(self) -> dict[str, Any]:
        """What the code runs against now: what the libraries resolve to in the interpreter, and each tool's
        fingerprint."""
        libraries = self._flex_ctx.libraries
        record: dict[str, Any] = json.loads(self._ask("describe", libraries)) if libraries else {}
        if tools := {name: _fingerprint(tool) for name, tool in self._flex_ctx.context_names().items()}:
            record["tools"] = tools
        return record

    def _changes(self, interpreter: Any = None) -> list[str]:
        """How the environment the code runs in now differs from the recorded one."""
        record = self._environment
        if not record:
            return []
        changes: list[str] = []
        if "python" in record and not self._flex_ctx.libraries:
            changes.append("it was validated with libraries, and now has none")
        elif "python" in record:
            found = json.loads(self._ask("verify", record, interpreter))
            changes += differences(record, found["python"], found["versions"], found["hashes"])
        tools = {name: _fingerprint(tool) for name, tool in self._flex_ctx.context_names().items()}
        for name, was in (record.get("tools") or {}).items():
            if name not in tools:
                changes.append(f"tool {name} is missing")
            elif tools[name] != was:
                changes.append(f"tool {name} changed")
        return changes

    def _environment_key(self) -> str:
        described = [self._module_src, self._environment, self._flex_ctx.libraries]
        return hashlib.sha256(json.dumps(described, sort_keys=True, default=str).encode()).hexdigest()

    def _check_environment_once(self, interpreter: Any) -> None:
        """Compare the recorded environment with the one of ``interpreter``, set up for a forward, and warn about
        what differs: once per process for each recorded environment, and never failing the forward."""
        if not self._environment:
            return
        try:
            key = self._environment_key()
            with _checked_lock:
                if key in _checked:
                    return
                _checked.add(key)
            changes = self._changes(interpreter)
        except Exception:  # a record that cannot be read, or an interpreter that cannot answer
            logger.debug("dspy.Flex %s: could not compare its recorded environment", self._name, exc_info=True)
            return
        if changes:
            logger.warning(
                "dspy.Flex %s: its code was validated against another environment (%s). Run your evaluation again, "
                "then call pin(), or optimize it again.",
                self._name,
                "; ".join(changes),
            )

    def reset(self) -> None:
        """Clear the LM; ``module_src`` is kept, and predictors are rebuilt from it each forward."""
        self.lm = None

    def named_predictors(self):
        """A Flex's update unit is only its ``module_src``."""
        return []

    def set_lm(self, lm) -> None:
        self.lm = lm

    def get_lm(self):
        return self.lm

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name})"

    def _baseline_src(self) -> str:
        """Starting source: one predictor over the whole signature, wrapped in a ``dspy.Module``.

        Uses ``dspy.RLM`` when tools are provided; otherwise a single ``dspy.Predict``.
        """
        cls: type[Signature] = self._signature_cls
        returns = ", ".join(f"{name}=result[{name!r}]" for name in cls.output_fields)
        preamble = ""
        if any(self._flex_ctx.field_kwargs().values()):
            # A class-based signature keeps the field descriptions and constraints that the string form drops.
            sig_arg = self._class_name().removesuffix("Module")
            preamble = self._flex_ctx.render_signature_class(sig_arg) + "\n\n\n"
        else:
            sig_str = self._flex_ctx.render_signature_string()
            instructions = (getattr(cls, "instructions", "") or "").strip()
            if custom := self._flex_ctx.custom_types():
                # Caller-frame discovery does not find types nested inside list/dict annotations.
                bindings = ", ".join(f"{name!r}: {name}" for name in custom)
                sig_arg = f"dspy.Signature({sig_str!r}, {instructions!r}, custom_types={{{bindings}}})"
            else:
                sig_arg = f"dspy.Signature({sig_str!r}, {instructions!r})" if instructions else repr(sig_str)
        tool_names = list(self._flex_ctx.context_names())
        if tool_names:
            attr, ctor = "rlm", f"dspy.RLM({sig_arg}, tools=[{', '.join(tool_names)}])"
        else:
            attr, ctor = "predict", f"dspy.Predict({sig_arg})"
        return preamble + (
            f"class {self._class_name()}(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.{attr} = {ctor}\n"
            "\n"
            "    def forward(self, **inputs):\n"
            f"        result = self.{attr}(**inputs)\n"
            f"        return dspy.Prediction({returns})"
        )

    def _class_name(self) -> str:
        base = self._name if (self._name and self._name.isidentifier()) else "Flex"
        return f"{base}Module"

    def _bind_code(self, module_src: str) -> None:
        self._bridge.bind(module_src)
        if module_src != self._module_src:
            self._environment = None  # recorded for other code
        self._module_src = module_src

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound ``forward`` inside the interpreter."""
        if args:
            raise TypeError("dspy.Flex accepts keyword inputs only")
        if self.lm is not None:
            with settings.context(lm=self.lm):
                return self._bridge.forward(kwargs)
        return self._bridge.forward(kwargs)


def differences(
    record: dict[str, Any], python: str | None, versions: dict[str, Any], hashes: dict[str, Any], limit: int = 10
) -> list[str]:
    """How an environment with this ``python``, these ``versions`` of packages and ``hashes`` of source files differs
    from the libraries' part of ``record``, in words. At most ``limit`` files are named."""
    changes = []
    if python != record.get("python"):
        changes.append(f"Python is {python}, was {record.get('python')}")
    for requirement in record.get("requires") or ():
        name, _, was = requirement.partition("==")
        if versions.get(name) is None:
            changes.append(f"{name} is not installed")
        elif versions[name] != was:
            changes.append(f"{name} is {versions[name]}, was {was}")
    files: list[str] = []
    for top, recorded in (record.get("sources") or {}).items():
        gone = [key for key in recorded if hashes.get(key) is None]
        if recorded and len(gone) == len(recorded):
            changes.append(f"{top} cannot be found")
            continue
        files += [f"{key} is gone" for key in gone]
        files += [f"{key} changed" for key, was in recorded.items() if hashes.get(key) not in (None, was)]
    if len(files) > limit:
        files = [*files[:limit], f"and {len(files) - limit} more files"]
    return changes + files


def _fingerprint(tool: Any) -> str:
    """A hash of what a tool looks like to the code: its name, description and arguments."""
    tool = tool if isinstance(tool, Tool) else Tool(tool)
    described = json.dumps([tool.name, tool.desc, tool.args], sort_keys=True, default=str)
    return hashlib.sha256(described.encode()).hexdigest()[:12]
