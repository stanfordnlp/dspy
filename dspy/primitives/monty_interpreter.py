from __future__ import annotations

import asyncio
import contextvars
import inspect
import keyword
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from pydantic_core import PydanticSerializationError, to_jsonable_python

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput


class MontyInterpreter:
    """Persistent, sandboxed Python-subset execution using ``dspy[monty]``.

    Each instance owns a Monty worker session. Only explicitly registered tools
    run on the host; filesystem, network, and environment access are not granted.
    ``limits`` accepts Monty's ResourceLimits. These bound guest execution, not
    the duration or memory consumption of host tools.
    """

    execution_instructions = (
        "Code runs in a persistent Monty Python sandbox. Variables and functions persist. "
        "Host tools and SUBMIT are available as global functions. Use print to inspect values. "
        "Monty supports a Python subset: no third-party packages, class inheritance, generator functions, "
        "globals(), or method decorators. Standard-library modules have limited APIs. "
        "Call native methods directly (text.strip()); use named helper functions for callbacks. "
        "The dspy facade adapts dspy.Module subclasses; call their forward method explicitly. "
        "The _dspy, _Dspy, and __dspy prefixes are reserved. "
        "Image inputs are URL/data-URI strings, not Pillow objects. Pass them to llm_query(..., images=[image]). "
        "If supplied, process_images(code, images) runs Pillow/OpenCV/NumPy code in a separate sandbox; "
        "those imports and DSPyImage methods are available only inside that tool's code. "
        "Use supplied tools for external access."
    )

    flex_execution_instructions = """Monty Flex authoring rules:
Write one dspy.Module subclass with __init__ and forward. Ordinary helper classes,
nested helpers, locals(), and async functions use Monty's native support.
Predictors, supplied tools, and compiled Flex methods
may be referenced by name and passed as values. Only supplied tools may be given to
bridged predictors. Call native methods directly: text.strip(), not fn = text.strip.
When a callback is needed, write a helper: def normalize(text): return text.strip().
Only direct super().__init__(...) in __init__ is supported;
do not alias or shadow super or access __class__. Call module.forward(...) explicitly.
The _dspy, _Dspy, and __dspy identifier prefixes are reserved. Standard-library APIs
are limited to Monty's supported subset; third-party imports are unavailable.
Use supplied host tools for external libraries; return plain data across the boundary.
Image inputs and sub-predictor image outputs are URL/data-URI strings. If supplied,
process_images(code, images) runs Pillow/OpenCV/NumPy in a separate sandbox;
only that code string can use DSPyImage.to_pil()/to_cv2() and library imports.
Runtime failures include diagnostics; revise the source to address
them rather than catching an unsupported operation and returning a dummy answer.
"""

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        *,
        limits: dict[str, Any] | None = None,
    ) -> None:
        self.tools = dict(tools or {})
        self.output_fields = output_fields
        self.limits = dict(limits or {})
        self._resources = ExitStack()
        self._session = None
        self._ended = False
        self._facade_installed = False
        self._compiled_code = None

    def prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Use image references rather than the Pyodide-specific image reconstruction code."""
        from dspy.adapters.types.image import Image

        def prepare(value):
            if isinstance(value, Image):
                return value.url
            if isinstance(value, dict):
                return {key: prepare(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [prepare(item) for item in value]
            return value

        return prepare(inputs)

    def start(self) -> None:
        if self._ended:
            raise CodeInterpreterError("MontyInterpreter session has been shut down.")
        if self._session is not None:
            return
        try:
            import pydantic_monty
        except ImportError as e:
            raise ImportError("Install Monty support with `pip install 'dspy[monty]'`.") from e
        try:
            pool = self._resources.enter_context(pydantic_monty.Monty())
            self._session = self._resources.enter_context(pool.checkout(limits=self.limits))
        except Exception as e:
            self.shutdown()
            raise CodeInterpreterError(f"Unable to start Monty: {e}") from e

    def _call_tool(self, name: str, args: tuple, kwargs: dict) -> Any:
        tool = self.tools[name]
        # DSPy tools expose their original signature but may accept only kwargs.
        signature = inspect.signature(tool)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        positional = []
        keywords = {}
        has_varargs = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values())
        for key, value in bound.arguments.items():
            kind = signature.parameters[key].kind
            if kind is inspect.Parameter.POSITIONAL_ONLY or (
                has_varargs and kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                positional.append(value)
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                positional.extend(value)
            elif kind is inspect.Parameter.VAR_KEYWORD:
                keywords.update(value)
            else:
                keywords[key] = value
        value = tool(*positional, **keywords)
        if inspect.isawaitable(value):
            async def wait():
                return await value

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(wait())
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(contextvars.copy_context().run, asyncio.run, wait()).result()
        return value

    def _submission(self, args: tuple, kwargs: dict) -> FinalOutput:
        if self.output_fields is None:
            if len(args) != 1 or kwargs:
                raise TypeError("SUBMIT requires one output value")
            return FinalOutput({"output": args[0]})
        names = [field["name"] for field in self.output_fields]
        if args and kwargs:
            raise TypeError("SUBMIT accepts positional or keyword values, not both")
        values = dict(zip(names, args, strict=False)) if args else dict(kwargs)
        if set(values) != set(names) or len(args) > len(names):
            raise TypeError("SUBMIT fields do not match the configured output fields")
        return FinalOutput(values)

    def _install_dspy_facade(self, tool_names) -> None:
        """Install the guest facade; predictor dispatch remains owned by FacadeInvocation."""
        self._execute(Path(__file__).with_name("_monty_shim.py").read_text(encoding="utf-8"))
        entries = ", ".join(f"({name}, {name!r})" for name in tool_names)
        self._execute(f"_dspy_tool_entries = [{entries}]")
        self._facade_installed = True

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        if self._facade_installed:
            from dspy.primitives._monty import compile_source

            self._compiled_code = compile_source(code)
            code = self._compiled_code.source
        return self._execute(code, variables)

    def _execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute native Monty code, including trusted facade setup and Flex driver code."""
        self.start()
        import pydantic_monty as monty

        for name in self.tools:
            if not name.isidentifier() or keyword.iskeyword(name) or name in {"SUBMIT", "__builtins__"}:
                raise CodeInterpreterError(f"Invalid tool name: {name!r}")
        if {"SUBMIT", "__builtins__", *self.tools} & (variables or {}).keys():
            raise CodeInterpreterError("Variables cannot replace interpreter-owned globals")
        try:
            inputs = to_jsonable_python(variables or {})
        except PydanticSerializationError as e:
            raise CodeInterpreterError(f"Unable to serialize interpreter inputs: {e}") from e
        output = monty.CollectString()
        submitted = None
        # Callable entries resolve names to external functions; calls themselves
        # are handled below to implement tools and the SUBMIT control signal.
        lookup = {**self.tools, "SUBMIT": self._submission}
        try:
            step = self._session.feed_start(
                code, inputs=inputs, external_lookup=lookup, print_callback=output,
            )
            while not isinstance(step, monty.MontyComplete):
                if isinstance(step, monty.FunctionSnapshot) and not step.is_os_function:
                    if step.function_name == "SUBMIT":
                        try:
                            submitted = self._submission(step.args, step.kwargs)
                        except TypeError as e:
                            step = step.resume({"exception": e})
                            continue
                        # Like the CPython worker's private BaseException, this
                        # unwinds the feed without losing its persistent globals.
                        step = step.resume({"exception": BaseException("__dspy_submit__")})
                        continue
                    if step.function_name in self.tools:
                        try:
                            value = to_jsonable_python(self._call_tool(step.function_name, step.args, step.kwargs))
                        except Exception as e:
                            step = step.resume({"exception": RuntimeError(f"{type(e).__name__}: {e}")})
                        else:
                            step = step.resume({"return_value": value})
                        continue
                step = step.resume_auto()
            return step.output if step.output is not None else (output.output.rstrip("\n") or None)
        except monty.MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except monty.MontyRuntimeError as e:
            if submitted is not None and str(e.exception()) == "__dspy_submit__":
                return submitted
            if isinstance(e.exception(), (MemoryError, TimeoutError)):
                self.shutdown()
                raise CodeInterpreterError(f"Monty resource limit ended the session: {e}") from e
            if self._compiled_code is not None:
                raise self._compiled_code.annotate(e) from e
            raise CodeExecutionError(str(e)) from e
        except (monty.MontyError, RuntimeError) as e:
            self.shutdown()
            raise CodeInterpreterError(str(e)) from e
        except BaseException:
            # A host interruption must not leave a suspended feed reusable.
            self.shutdown()
            raise

    def shutdown(self) -> None:
        self._ended = True
        self._session = None
        self._resources.close()
