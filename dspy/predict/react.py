import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Literal

from litellm import ContextWindowExceededError

import dspy
from dspy.adapters.types.history import History
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class ReAct(Module):
    """ReAct (Reasoning and Acting) agent module.

    ReAct iteratively reasons about the current situation and takes actions using tools.
    The trajectory is stored as a History in raw LM message format.

    Args:
        signature: The signature defining input and output fields.
        tools: List of callable tools the agent can use.
        max_iters: Maximum reasoning iterations (default: 10).

    Example:
        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct("question -> answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        print(pred.answer)
        print(pred.trajectory)  # History object with tool call messages
        ```
    """

    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 10):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend([
            f"You are an Agent. In each episode, you will be given the fields {inputs} as input. "
            "And you can see your past trajectory so far.",
            f"Your goal is to use one or more of the supplied tools to collect any necessary information "
            f"for producing {outputs}.\n",
            "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, "
            "and also when finishing the task.",
            "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
            "When writing next_thought, you may reason about the current situation and plan for future steps.",
            "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
        ])

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. Signals that all information for producing {outputs} is available.",
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=History)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        extract_instructions = (
            f"You are an extraction agent. Extract the fields: {outputs} from the given trajectory.\n"
            f"The original task was:\n{signature.instructions}\n"
            "An executor agent has used tools to generate the conversation below. "
            f"Given this trajectory, extract the fields: {outputs}."
        )
        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            extract_instructions,
        ).append(
            "trajectory",
            dspy.InputField(desc="The conversation history with enough context to produce the output"),
            type_=History,
        )

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def forward(self, *, trajectory: History | None = None, **input_args):
        max_iters = input_args.pop("max_iters", self.max_iters)
        trajectory = trajectory or History.from_raw([])

        for _ in range(max_iters):
            try:
                pred, trajectory = self._call_with_retry(self.react, trajectory, **input_args)
            except (ValueError, ContextWindowExceededError) as err:
                logger.warning(f"Ending trajectory: {_fmt_exc(err)}")
                break

            observation = self._run_tool(pred.next_tool_name, pred.next_tool_args)
            trajectory = self._record_step(trajectory, pred, observation)

            if pred.next_tool_name == "finish":
                break

        extract, trajectory = self._call_with_retry(self.extract, trajectory, **input_args)
        trajectory = self._record_extract(trajectory, extract)

        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, *, trajectory: History | None = None, **input_args):
        max_iters = input_args.pop("max_iters", self.max_iters)
        trajectory = trajectory or History.from_raw([])

        for _ in range(max_iters):
            try:
                pred, trajectory = await self._acall_with_retry(self.react, trajectory, **input_args)
            except (ValueError, ContextWindowExceededError) as err:
                logger.warning(f"Ending trajectory: {_fmt_exc(err)}")
                break

            observation = await self._arun_tool(pred.next_tool_name, pred.next_tool_args)
            trajectory = self._record_step(trajectory, pred, observation)

            if pred.next_tool_name == "finish":
                break

        extract, trajectory = await self._acall_with_retry(self.extract, trajectory, **input_args)
        trajectory = self._record_extract(trajectory, extract)

        return dspy.Prediction(trajectory=trajectory, **extract)

    # -------------------------------------------------------------------------
    # Tool execution
    # -------------------------------------------------------------------------

    def _run_tool(self, name: str, args: dict) -> str:
        try:
            result = self.tools[name](**args)
            return self._serialize(result)
        except Exception as err:
            return f"Execution error in {name}: {_fmt_exc(err)}"

    async def _arun_tool(self, name: str, args: dict) -> str:
        try:
            result = await self.tools[name].acall(**args)
            return self._serialize(result)
        except Exception as err:
            return f"Execution error in {name}: {_fmt_exc(err)}"

    def _serialize(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)

    # -------------------------------------------------------------------------
    # Trajectory recording
    # -------------------------------------------------------------------------

    def _record_step(self, trajectory: History, pred, observation: str) -> History:
        """Record a single agent step (action + observation) to the trajectory."""
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

        action_msg = {
            "role": "assistant",
            "content": pred.next_thought,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": pred.next_tool_name,
                    "arguments": json.dumps(pred.next_tool_args),
                },
            }],
        }

        observation_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": observation,
        }

        return trajectory.with_messages([action_msg, observation_msg])

    def _record_extract(self, trajectory: History, extract) -> History:
        """Record the final extraction result to the trajectory."""
        extract_dict = dict(extract)
        reasoning = extract_dict.pop("reasoning", None)

        parts = []
        if reasoning:
            parts.append(f"Reasoning: {reasoning}")
        for key, value in extract_dict.items():
            parts.append(f"{key}: {self._serialize(value)}")

        return trajectory.with_messages([{"role": "assistant", "content": "\n".join(parts)}])

    # -------------------------------------------------------------------------
    # LM calls with truncation retry
    # -------------------------------------------------------------------------

    def _call_with_retry(self, module, trajectory: History, **input_args) -> tuple[Any, History]:
        last_err = None
        for _ in range(3):
            try:
                return module(**input_args, trajectory=trajectory), trajectory
            except ContextWindowExceededError as err:
                last_err = err
                logger.warning("Context window exceeded, truncating oldest step.")
                trajectory = self.truncate_trajectory(trajectory)
        raise ValueError(
            "The context window was exceeded even after 3 attempts to truncate the trajectory."
        )

        raise ContextWindowExceededError(f"Context window exceeded after 3 truncation attempts: {last_err}")

    async def _acall_with_retry(self, module, trajectory: History, **input_args) -> tuple[Any, History]:
        last_err = None
        for _ in range(3):
            try:
                return await module.acall(**input_args, trajectory=trajectory), trajectory
            except ContextWindowExceededError as err:
                last_err = err
                logger.warning("Context window exceeded, truncating oldest step.")
                trajectory = self.truncate_trajectory(trajectory)
        raise ValueError(
            "The context window was exceeded even after 3 attempts to truncate the trajectory."
        )

        raise ContextWindowExceededError(f"Context window exceeded after 3 truncation attempts: {last_err}")

    def truncate_trajectory(self, trajectory: History) -> History:
        """Remove the oldest tool call pair from the trajectory.

        Override this method to implement custom truncation logic.
        """
        messages = list(trajectory.messages)

        if len(messages) < 2:
            raise ValueError("Trajectory too long but cannot truncate: only one step remains.")

        # Remove assistant + following tool response(s)
        if messages[0].get("role") == "assistant" and messages[0].get("tool_calls"):
            messages.pop(0)
            while messages and messages[0].get("role") == "tool":
                messages.pop(0)
        else:
            messages.pop(0)

        return History(messages=messages, mode=trajectory.mode)


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
