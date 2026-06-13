from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, get_args

import pydantic

import dspy
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental
from dspy.utils.exceptions import AdapterParseError, ContextWindowExceededError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


@experimental
class ReActV2(Module):
    # Field names ReActV2 attaches to every returned `Prediction`. A user output field with one
    # of these names would collide with the keyword arguments below and raise an opaque TypeError
    # mid-`forward()`, so we reject it up front.
    _RESERVED_OUTPUT_FIELDS = ("history", "termination_reason")

    def __init__(self, signature: type[Signature], tools: list[Callable | Tool], max_iters: int = 20):
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters

        reserved = [name for name in self._RESERVED_OUTPUT_FIELDS if name in self.signature.output_fields]
        if reserved:
            raise ValueError(
                f"ReActV2 reserves the output field name(s) {', '.join(reserved)} for its own metadata. "
                "Please rename the conflicting field(s) in your signature."
            )

        user_tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
        self.tools = {tool.name: tool for tool in user_tools}
        if "submit" in self.tools:
            raise ValueError("`submit` is reserved by ReActV2 as the final-output tool.")
        self.tools["submit"] = self._make_submit_tool()

        self.react = dspy.Predict(self._make_react_signature())

    def _make_submit_tool(self) -> Tool:
        output_fields = self.signature.output_fields
        output_names = list(output_fields)

        def submit(**kwargs):
            missing = [name for name in output_names if name not in kwargs]
            if missing:
                raise ValueError(f"Missing required final output field(s): {', '.join(missing)}")
            return {name: kwargs[name] for name in output_names}

        args = {
            name: _json_schema_for_annotation(field.annotation)
            for name, field in output_fields.items()
        }
        arg_types = {name: field.annotation for name, field in output_fields.items()}
        return Tool(
            submit,
            name="submit",
            desc="Submit the final outputs for the task.",
            args=args,
            arg_types=arg_types,
        )

    def _make_react_signature(self) -> type[Signature]:
        fields = {}
        for name, field in self.signature.input_fields.items():
            fields[name] = (
                _optional_annotation(field.annotation),
                dspy.InputField(desc=field.json_schema_extra.get("desc")),
            )

        fields["history"] = (dspy.History, dspy.InputField())
        fields["tools"] = (list[dspy.Tool], dspy.InputField())
        fields["next_thought"] = (dspy.Reasoning, dspy.OutputField())
        fields["tool_calls"] = (dspy.ToolCalls, dspy.OutputField())

        inputs = ", ".join(f"`{name}`" for name in self.signature.input_fields)
        outputs = ", ".join(f"`{name}`" for name in self.signature.output_fields)
        tool_names = ", ".join(f"`{name}`" for name in self.tools)
        instructions = "\n".join(
            [
                self.signature.instructions,
                f"You are an Agent. Use the supplied tools to produce {outputs} from {inputs}.",
                "Call tools when more information is needed.",
                f"When the final answer is ready, call `submit` with {outputs}.",
                f"The available tools are: {tool_names}.",
            ]
        ).strip()

        return dspy.Signature(fields, instructions)

    def forward(self, **input_args):
        max_iters = input_args.pop("max_iters", self.max_iters)
        history = _coerce_history(input_args.pop("history", None))
        pending_inputs = {name: input_args[name] for name in self.signature.input_fields if name in input_args}

        break_reason = "max_iters"
        for turn_index in range(max_iters):
            try:
                pred = self.react(
                    history=history,
                    tools=list(self.tools.values()),
                    **pending_inputs,
                )
                tool_calls = _coerce_tool_calls(getattr(pred, "tool_calls", None))
            except (AdapterParseError, ValueError) as err:
                logger.warning("Ending ReActV2 loop after parse failure: %s", _fmt_exc(err))
                break_reason = "parse_error"
                break
            except ContextWindowExceededError:
                logger.warning("Ending ReActV2 loop after context window exceeded.")
                break_reason = "context_window_exceeded"
                break

            if not tool_calls.tool_calls:
                break_reason = "empty_tool_calls"
                break

            tool_calls = _ensure_tool_call_ids(tool_calls, turn_index)
            tool_call_results, final_outputs = self._execute_tool_calls(tool_calls)
            event = self._history_event(pending_inputs, pred, tool_calls, tool_call_results)
            if final_outputs is not None:
                event.update(final_outputs)
            _append_history_event(history, event)
            pending_inputs = {}

            if final_outputs is not None:
                return Prediction(**final_outputs, history=history, termination_reason="submit")

        return self._forced_submit(history, pending_inputs, break_reason, max_iters)

    def _execute_tool_calls(self, tool_calls: ToolCalls) -> tuple[ToolCallResults, dict[str, Any] | None]:
        values = []
        is_errors = []
        final_outputs = None

        for tool_call in tool_calls.tool_calls:
            if tool_call.name not in self.tools:
                values.append(f"Unknown tool: {tool_call.name}")
                is_errors.append(True)
                continue

            try:
                value = self.tools[tool_call.name](**(tool_call.args or {}))
                values.append(value)
                is_errors.append(False)
                if tool_call.name == "submit" and isinstance(value, dict):
                    final_outputs = value
            except Exception as err:
                values.append(f"Execution error in {tool_call.name}: {_fmt_exc(err)}")
                is_errors.append(True)

        return ToolCallResults.from_tool_calls_and_values(tool_calls, values, is_errors), final_outputs

    def _history_event(
        self,
        pending_inputs: dict[str, Any],
        pred: Prediction,
        tool_calls: ToolCalls,
        tool_call_results: ToolCallResults,
    ) -> dict[str, Any]:
        event = dict(pending_inputs)
        if hasattr(pred, "next_thought") and pred.next_thought is not None:
            event["next_thought"] = pred.next_thought
        if tool_calls.tool_calls:
            if tool_call_results.tool_call_results:
                tool_calls = tool_calls.model_copy(update={"tool_call_results": tool_call_results})
            event["tool_calls"] = tool_calls
        return event

    def _forced_submit(
        self,
        history: dspy.History,
        pending_inputs: dict[str, Any],
        break_reason: str,
        turn_index: int,
    ) -> Prediction:
        try:
            pred = self.react(
                history=history,
                tools=list(self.tools.values()),
                config={
                    "tool_choice": {"type": "function", "function": {"name": "submit"}},
                    "reasoning_effort": None,
                },
                **pending_inputs,
            )
            tool_calls = _ensure_tool_call_ids(_coerce_tool_calls(getattr(pred, "tool_calls", None)), turn_index)
        except (AdapterParseError, ValueError, ContextWindowExceededError) as err:
            logger.warning("Forced submit failed: %s", _fmt_exc(err))
            return Prediction(history=history, termination_reason=break_reason or "failed")

        submit_calls = ToolCalls(tool_calls=[call for call in tool_calls.tool_calls if call.name == "submit"])
        if not submit_calls.tool_calls:
            return Prediction(history=history, termination_reason=break_reason or "failed")

        tool_call_results, final_outputs = self._execute_tool_calls(submit_calls)
        event = self._history_event(pending_inputs, pred, submit_calls, tool_call_results)
        if final_outputs is not None:
            event.update(final_outputs)
        _append_history_event(history, event)

        if final_outputs is not None:
            return Prediction(**final_outputs, history=history, termination_reason="forced_submit")

        return Prediction(history=history, termination_reason=break_reason or "failed")


def _json_schema_for_annotation(annotation: Any) -> dict[str, Any]:
    try:
        return pydantic.TypeAdapter(annotation).json_schema()
    except Exception:
        return {"type": "string"}


def _optional_annotation(annotation: Any) -> Any:
    if type(None) in get_args(annotation):
        return annotation
    try:
        return annotation | None
    except TypeError:
        return annotation


def _coerce_history(history: Any) -> dspy.History:
    if history is None:
        return dspy.History(messages=[])
    if isinstance(history, dspy.History):
        return history
    return dspy.History.model_validate(history)


def _coerce_tool_calls(tool_calls: Any) -> ToolCalls:
    if tool_calls is None:
        return ToolCalls(tool_calls=[])
    return ToolCalls.model_validate(tool_calls)


def _ensure_tool_call_ids(tool_calls: ToolCalls, turn_index: int) -> ToolCalls:
    ensured = []
    for call_index, tool_call in enumerate(tool_calls.tool_calls):
        if tool_call.id is None:
            tool_call = tool_call.model_copy(update={"id": f"call_{turn_index}_{call_index}"})
        ensured.append(tool_call)
    return ToolCalls(tool_calls=ensured)


def _append_history_event(history: dspy.History, event: dict[str, Any]) -> None:
    if event:
        history.messages.append(event)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
