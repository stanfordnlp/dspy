from __future__ import annotations

import hashlib
import json
import logging
import types
from typing import Any, Callable

from dspy.adapters.types.tool import Tool
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.settings import settings
from dspy.predict.flex.bridge import BridgeRuntime
from dspy.predict.flex.ctx import FlexContext
from dspy.predict.flex.library import Library
from dspy.predict.parameter import Parameter
from dspy.predict.predict import _sanitize_lm_state
from dspy.primitives.code_interpreter import CodeInterpreter, _validate_interpreter_factory
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)


@experimental(version="3.3.0")
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

    The code can also use your own libraries, such as an SDK: pass them as ``libraries`` and the optimizer gets an
    API reference for each (see ``dspy.Library``), next to DSPy's modules, and writes code that uses what fits. The
    libraries' versions are saved with the program, and loading it next to other versions warns.

    Args:
        signature: A ``dspy.Signature`` class or string declaring inputs/outputs.
        tools: ``dspy.Tool`` instances or named callables.
        libraries: Modules the code may import, as ``dspy.Library`` objects, modules, or module names. The
            interpreter must be able to import them: ``dspy.LocalInterpreter`` imports them from this environment.
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
        libraries: list[Library | types.ModuleType | str] | None = None,
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
            libraries=[library if isinstance(library, Library) else Library(library) for library in libraries or []],
        )

        self._module_src: str | None = None
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
        state["_saved_environment"] = self._environment()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        saved = state.pop("_saved_environment", None)
        super().__setstate__(state)
        self._rebuild_bridge()
        if self._module_src is not None:
            self._bridge.bind(self._module_src)
        self._check_environment(saved)

    @property
    def signature(self) -> type[Signature]:
        return self._signature_cls

    @property
    def module_src(self) -> str | None:
        return self._module_src

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        return {
            "module_src": self._module_src,
            "lm": self.lm.dump_state() if self.lm else None,
            "environment": self._environment(),
        }

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
        self._check_environment(state.get("environment"))

    def _environment(self) -> dict[str, Any]:
        """What ``module_src`` was written against: each library's version and API fingerprint, and each tool's."""
        libraries = {}
        for library in self._flex_ctx.libraries:
            try:
                libraries[library.name] = {"version": library.version, "api": library.fingerprint}
            except ImportError:
                libraries[library.name] = None
        tools = {name: _fingerprint(tool) for name, tool in self._flex_ctx.context_names().items()}
        return {"libraries": libraries, "tools": tools}

    def _check_environment(self, saved: dict[str, Any] | None) -> None:
        """Warn when this module's libraries or tools differ from the ones its code was optimized with."""
        if not saved:
            return
        current = self._environment()
        changes = []
        for name, was in (saved.get("libraries") or {}).items():
            now = current["libraries"].get(name, False)
            if now is False:
                changes.append(f"{name} is not one of its libraries")
            elif now is None:
                changes.append(f"{name} cannot be imported")
            elif was and now["version"] != was.get("version"):
                changes.append(f"{name} is {now['version']}, not {was.get('version')}")
            elif was and now["api"] != was.get("api"):
                changes.append(f"{name}'s API changed")
        for name, was in (saved.get("tools") or {}).items():
            now = current["tools"].get(name)
            if now is None:
                changes.append(f"tool {name} is missing")
            elif now != was:
                changes.append(f"tool {name} changed")
        if changes:
            logger.warning(
                "dspy.Flex %s: its code was optimized against other libraries or tools (%s). Install the versions it "
                "was optimized with, or optimize it again.",
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
        self._module_src = module_src

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound ``forward`` inside the interpreter."""
        if args:
            raise TypeError("dspy.Flex accepts keyword inputs only")
        if self.lm is not None:
            with settings.context(lm=self.lm):
                return self._bridge.forward(kwargs)
        return self._bridge.forward(kwargs)


def _fingerprint(tool: Any) -> str:
    """A hash of what a tool looks like to the code: its name, description and arguments."""
    tool = tool if isinstance(tool, Tool) else Tool(tool)
    described = json.dumps([tool.name, tool.desc, tool.args], sort_keys=True, default=str)
    return hashlib.sha256(described.encode()).hexdigest()[:12]
