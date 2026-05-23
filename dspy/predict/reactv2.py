import logging
import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import dspy
from dspy.adapters.types.history import History
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.exceptions import AdapterParseError, ContextWindowExceededError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

_ANNOTATION_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}
_RECOVERABLE_FORCED_SUBMIT_ERRORS = (AdapterParseError, ContextWindowExceededError, ValueError)


@dataclass(frozen=True)
class ToolExecutionResult:
    value: object
    is_error: bool = False


def _build_submit_tool(signature: type["Signature"]) -> Tool:
    outputs = ", ".join([f"`{k}`" for k in signature.output_fields])
    output_args = {}
    output_arg_types = {}
    for name, field in signature.output_fields.items():
        annotation = getattr(field, "annotation", str)
        output_args[name] = {"type": _ANNOTATION_TO_JSON_TYPE.get(annotation, "string")}
        output_arg_types[name] = annotation

    return Tool(
        func=lambda **kwargs: kwargs,
        name="submit",
        desc=f"Call this tool to end the task and return your final answer. Takes: {outputs}.",
        args=output_args,
        arg_types=output_arg_types,
    )


def _optional_input_fields(signature: type["Signature"]) -> dict[str, object]:
    fields = deepcopy(signature.input_fields)
    for field in fields.values():
        field.default = None
    return fields


class ReActV2(Module):
    def __init__(self, signature: type["Signature"] | str, tools: list[Callable], max_iters: int = 20):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [tool if isinstance(tool, Tool) else Tool(tool) for tool in tools]
        self.tools = {tool.name: tool for tool in tools}
        self.tools["submit"] = _build_submit_tool(signature)

        react_signature = (
            dspy.Signature(_optional_input_fields(signature), self._build_instructions())
            .append("history", dspy.InputField(), type_=dspy.History)
            .append("tools", dspy.InputField(), type_=list[dspy.Tool])
            .append(
                "tool_call_results",
                dspy.InputField(default=None),
                type_=dspy.ToolCallResults | None,
            )
            .append("next_thought", dspy.OutputField(), type_=dspy.Reasoning)
            .append("tool_calls", dspy.OutputField(), type_=dspy.ToolCalls)
        )
        extract_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append(
            "trajectory",
            dspy.InputField(desc="The agent's history of thoughts, actions, and tool call results"),
            type_=str,
        )
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(extract_signature)

    def _build_instructions(self) -> str:
        inputs = ", ".join([f"`{k}`" for k in self.signature.input_fields])
        outputs = ", ".join([f"`{k}`" for k in self.signature.output_fields])
        instructions = [f"{self.signature.instructions}\n"] if self.signature.instructions else []
        instructions.extend(
            [
                f"You are an Agent. Given {inputs}, use tools to produce {outputs}.",
                "Each turn: think, then call one or more tools. After each tool call you receive a tool result.",
                "When you have enough information to answer, call `submit` to finish.",
                "\nAvailable tools:\n",
            ]
        )
        instructions.extend(f"({idx + 1}) {tool}" for idx, tool in enumerate(self.tools.values()))
        return "\n".join(instructions)

    def forward(self, **input_args):
        history = input_args.pop("history", dspy.History(messages=[]))
        max_iters = input_args.pop("max_iters", self.max_iters)
        tool_list = list(self.tools.values())
        pending_inputs = dict(input_args)

        break_reason = None
        for idx in range(max_iters):
            try:
                pred = self.react(history=history, tools=tool_list, **pending_inputs)
            except ContextWindowExceededError:
                history.compact_if_needed()
                try:
                    pred = self.react(history=history, tools=tool_list, **pending_inputs)
                except ContextWindowExceededError:
                    logger.warning("Context window exceeded after compaction, ending loop.")
                    break_reason = "context_overflow"
                    break
            except (AdapterParseError, ValueError) as err:
                logger.warning(f"Agent iteration {idx} failed: {_fmt_exc(err)}")
                break_reason = "parse_error"
                break

            if pred.tool_calls is None or not pred.tool_calls.tool_calls:
                logger.warning("Agent returned no tool calls, ending loop.")
                break_reason = "no_tool_calls"
                break

            tool_calls = pred.tool_calls.with_call_ids(f"call_{len(history.messages)}")
            tool_results = [self._execute_tool_call(tool_call) for tool_call in tool_calls.tool_calls]
            self._append_tool_turn(
                history,
                inputs=pending_inputs,
                next_thought=pred.next_thought,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            pending_inputs = {}

            for tool_call, tool_result in zip(tool_calls.tool_calls, tool_results, strict=True):
                if tool_call.name == "submit" and not tool_result.is_error:
                    history.messages[-1].update(tool_result.value)
                    return dspy.Prediction(history=history, termination_reason="submit", **tool_result.value)

        return self._forced_submit(history, pending_inputs, break_reason=break_reason)

    def _forced_submit(self, history: History, input_args: dict[str, object], break_reason: str | None = None):
        tool_list = list(self.tools.values())
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        call_config = adapter.force_tool_call_config("submit")

        try:
            pred = self.react(history=history, tools=tool_list, config=call_config, **input_args)
        except _RECOVERABLE_FORCED_SUBMIT_ERRORS as err:
            logger.debug(f"Forced submit tier 1 (react) failed: {_fmt_exc(err)}")
            pred = None

        if pred and pred.tool_calls and pred.tool_calls.tool_calls:
            for tool_call in pred.tool_calls.tool_calls:
                if tool_call.name != "submit":
                    continue
                try:
                    result = self.tools["submit"](**tool_call.args)
                except ValueError as err:
                    logger.debug(f"Forced submit tool execution failed: {_fmt_exc(err)}")
                    continue

                tool_calls = ToolCalls(tool_calls=[tool_call]).with_call_ids(f"call_{len(history.messages)}")
                self._append_tool_turn(
                    history,
                    inputs=input_args,
                    next_thought=pred.next_thought,
                    tool_calls=tool_calls,
                    tool_results=[ToolExecutionResult(value=result, is_error=False)],
                )
                try:
                    history.messages[-1].update(result)
                    return dspy.Prediction(history=history, termination_reason="forced_submit", **result)
                except TypeError as err:
                    logger.debug(f"Forced submit result was not a valid output mapping: {_fmt_exc(err)}")

        try:
            trajectory_text = self._render_history_as_text(history)
            extract = self.extract(trajectory=trajectory_text, **input_args)
            result = {k: getattr(extract, k) for k in self.signature.output_fields if hasattr(extract, k)}
            if any(value is not None for value in result.values()):
                history.append(result)
                return dspy.Prediction(history=history, termination_reason="extract", **result)
        except _RECOVERABLE_FORCED_SUBMIT_ERRORS as err:
            logger.debug(f"Forced submit tier 2 (extract) failed: {_fmt_exc(err)}")

        return dspy.Prediction(history=history, termination_reason=break_reason or "failed")

    def _execute_tool_call(self, tool_call: ToolCalls.ToolCall) -> ToolExecutionResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolExecutionResult(value=f"Unknown tool: {tool_call.name}", is_error=True)
        try:
            return ToolExecutionResult(value=tool(**tool_call.args), is_error=False)
        except Exception as err:
            return ToolExecutionResult(value=f"Execution error in {tool_call.name}: {_fmt_exc(err)}", is_error=True)

    @staticmethod
    def _append_tool_turn(
        history: History,
        *,
        inputs: dict[str, object] | None = None,
        next_thought,
        tool_calls: ToolCalls,
        tool_results: list[ToolExecutionResult],
    ) -> None:
        history.append(
            {
                **(inputs or {}),
                "next_thought": next_thought,
                "tool_calls": tool_calls,
                "tool_call_results": ToolCallResults.from_tool_calls_and_values(
                    tool_calls.tool_calls,
                    [tool_result.value for tool_result in tool_results],
                    [tool_result.is_error for tool_result in tool_results],
                ),
            }
        )

    def _render_history_as_text(self, history: History) -> str:
        lines = []
        for event in history.messages:
            thought = event.get("next_thought")
            tool_calls = event.get("tool_calls")
            tool_call_results = event.get("tool_call_results")
            if thought:
                lines.append(f"[Thought] {thought}")
            if isinstance(tool_calls, ToolCalls):
                for tool_call in tool_calls.tool_calls:
                    args = ", ".join(f"{key}={value!r}" for key, value in (tool_call.args or {}).items())
                    lines.append(f"[Action] {tool_call.name}({args})")
            if isinstance(tool_call_results, ToolCallResults):
                for result in tool_call_results.tool_call_results:
                    prefix = "[Error]" if result.is_error else "[Observation]"
                    lines.append(f"{prefix} {result.name}: {result.value}")
            for key, value in event.items():
                if key in {"next_thought", "tool_calls", "tool_call_results"}:
                    continue
                label = "[Input]" if key in self.signature.input_fields else "[Output]"
                lines.append(f"{label} {key}: {value}")
        return "\n".join(lines)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
