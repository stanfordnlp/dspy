import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class ReAct(Module):
    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 20):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

        Example:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct(signature="question->answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                (
                    f"You are an Agent. In each episode, you will be given the fields {inputs} as input. "
                    "And you can see your prior tool-use turns via `history`."
                ),
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to history as a tool event.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("history", dspy.InputField(), type_=dspy.History)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("history", dspy.InputField(), type_=dspy.History)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _prepare_execution(
        self, input_args: dict[str, Any]
    ) -> tuple[dict[str, Any], dspy.History, dspy.HistoryResumeState, int]:
        run_args = dict(input_args)
        resume_mode = run_args.pop("resume", "off")
        if resume_mode not in {"off", "auto", "strict"}:
            raise ValueError("ReAct `resume` must be one of: 'off', 'auto', 'strict'.")

        history = self._coerce_history(run_args.pop("history", None))
        resume_state = self._initialize_resume_state(history, run_args, resume_mode)
        max_iters = run_args.pop("max_iters", self.max_iters)
        return run_args, history, resume_state, max_iters

    def _coerce_history(self, provided_history: Any) -> dspy.History:
        if provided_history is None:
            return dspy.History(messages=[])
        if isinstance(provided_history, dspy.History):
            return provided_history.trimmed()
        raise TypeError("ReAct `history` must be a dspy.History instance.")

    def _initialize_resume_state(
        self,
        history: dspy.History,
        input_args: dict[str, Any],
        resume_mode: Literal["off", "auto", "strict"],
    ) -> dspy.HistoryResumeState:
        return history.parse_resume_state(
            input_args=input_args,
            input_keys=set(self.signature.input_fields.keys()),
            available_tools=set(self.tools.keys()),
            resume_mode=resume_mode,
            assistant_required_fields={"next_thought", "next_tool_name", "next_tool_args"},
            assistant_tool_name_field="next_tool_name",
            assistant_tool_args_field="next_tool_args",
            tool_message_tool_name_field="tool_name",
            tool_message_observation_field="observation",
        )

    async def _aexecute_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        try:
            observation = await self.tools[tool_name].acall(**tool_args)
        except Exception as err:
            observation = f"Execution error in {tool_name}: {_fmt_exc(err)}"
        return observation

    def _execute_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        try:
            observation = self.tools[tool_name](**tool_args)
        except Exception as err:
            observation = f"Execution error in {tool_name}: {_fmt_exc(err)}"
        return observation

    def _append_tool_observation(self, history: dspy.History, tool_name: str, observation: Any) -> dspy.History:
        return history.append(
            dspy.HistoryMessage(
                role="tool",
                fields={"tool_name": tool_name, "observation": observation},
            )
        )

    def _apply_pending_tool_call(
        self, history: dspy.History, resume_state: dspy.HistoryResumeState
    ) -> tuple[dspy.History, int, str | None]:
        completed_turns = resume_state.completed_turns
        last_completed_tool_name = resume_state.last_completed_tool_name
        pending_tool_call = resume_state.pending_tool_call

        if pending_tool_call is None:
            return history, completed_turns, last_completed_tool_name

        pending_tool_name, pending_tool_args = pending_tool_call
        observation = self._execute_tool_call(pending_tool_name, pending_tool_args)
        history = self._append_tool_observation(history, pending_tool_name, observation)
        completed_turns += 1
        last_completed_tool_name = pending_tool_name
        return history, completed_turns, last_completed_tool_name

    async def _aapply_pending_tool_call(
        self, history: dspy.History, resume_state: dspy.HistoryResumeState
    ) -> tuple[dspy.History, int, str | None]:
        completed_turns = resume_state.completed_turns
        last_completed_tool_name = resume_state.last_completed_tool_name
        pending_tool_call = resume_state.pending_tool_call

        if pending_tool_call is None:
            return history, completed_turns, last_completed_tool_name

        pending_tool_name, pending_tool_args = pending_tool_call
        observation = await self._aexecute_tool_call(pending_tool_name, pending_tool_args)
        history = self._append_tool_observation(history, pending_tool_name, observation)
        completed_turns += 1
        last_completed_tool_name = pending_tool_name
        return history, completed_turns, last_completed_tool_name

    def _call_inputs_from_history(self, history: dspy.History, **input_args) -> dict[str, Any]:
        call_inputs = dict(input_args)
        if history.messages:
            call_inputs["history"] = history

        return call_inputs

    def _step_to_history_messages(self, pred: dspy.Prediction, observation: Any) -> list[dspy.HistoryMessage]:
        return [
            dspy.HistoryMessage(
                role="assistant",
                fields={
                    "next_thought": pred.next_thought,
                    "next_tool_name": pred.next_tool_name,
                    "next_tool_args": pred.next_tool_args,
                },
            ),
            dspy.HistoryMessage(
                role="tool",
                fields={
                    "tool_name": pred.next_tool_name,
                    "observation": observation,
                },
            ),
        ]

    def _warn_invalid_tool_selection(self, err: ValueError) -> None:
        logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")

    def forward(self, **input_args):
        input_args, history, resume_state, max_iters = self._prepare_execution(input_args)
        history, completed_turns, last_completed_tool_name = self._apply_pending_tool_call(history, resume_state)

        for _ in range(completed_turns, max_iters):
            if last_completed_tool_name == "finish":
                break

            try:
                pred = self.react(**self._call_inputs_from_history(history, **input_args))
            except ValueError as err:
                self._warn_invalid_tool_selection(err)
                break

            observation = self._execute_tool_call(pred.next_tool_name, pred.next_tool_args)

            history = history.append_many(self._step_to_history_messages(pred, observation))
            completed_turns += 1
            last_completed_tool_name = pred.next_tool_name

            if pred.next_tool_name == "finish":
                break

        extract = self.extract(**self._call_inputs_from_history(history, **input_args))
        return dspy.Prediction(history=history, **extract)

    async def aforward(self, **input_args):
        input_args, history, resume_state, max_iters = self._prepare_execution(input_args)
        history, completed_turns, last_completed_tool_name = await self._aapply_pending_tool_call(history, resume_state)

        for _ in range(completed_turns, max_iters):
            if last_completed_tool_name == "finish":
                break

            try:
                pred = await self.react.acall(**self._call_inputs_from_history(history, **input_args))
            except ValueError as err:
                self._warn_invalid_tool_selection(err)
                break

            observation = await self._aexecute_tool_call(pred.next_tool_name, pred.next_tool_args)

            history = history.append_many(self._step_to_history_messages(pred, observation))
            completed_turns += 1
            last_completed_tool_name = pred.next_tool_name

            if pred.next_tool_name == "finish":
                break

        extract = await self.extract.acall(**self._call_inputs_from_history(history, **input_args))
        return dspy.Prediction(history=history, **extract)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


"""
Thoughts and Planned Improvements for dspy.ReAct.

TOPIC 01: How Trajectories are Formatted, or rather when they are formatted.

Right now, both sub-modules are invoked with a `trajectory` argument, which is a string formatted in `forward`. Though
the formatter uses a general adapter.format_fields, the tracing of DSPy only sees the string, not the formatting logic.

What this means is that, in demonstrations, even if the user adjusts the adapter for a fixed program, the demos' format
will not update accordingly, but the inference-time trajectories will.

One way to fix this is to support `format=fn` in the dspy.InputField() for "trajectory" in the signatures. But this
means that care must be taken that the adapter is accessed at `forward` runtime, not signature definition time.

Another potential fix is to more natively support a "variadic" input field, where the input is a list of dictionaries,
or a big dictionary, and have each adapter format it accordingly.

Trajectories also affect meta-programming modules that view the trace later. It's inefficient O(n^2) to view the
trace of every module repeating the prefix.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?


TOPIC 06: Idiomatically allowing tools that maintain state across iterations, but not across different `forward` calls.
    * So the tool would be newly initialized at the start of each `forward` call, but maintain state across iterations.
    * This is pretty useful for allowing the agent to keep notes or count certain things, etc.
"""
