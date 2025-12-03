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
    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 10):
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
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
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
            .append("trajectory", dspy.InputField(), type_=History)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        extract_instructions = (
            "You are an extraction Agent whose job it is to extract the fields: {outputs} from the given trajectory."
            + "The original task was:\n"
            + signature.instructions
            + "\nIn trying to solve this task, an executor agent with has used tools to generate the conversation below."
            + "\nGiven this trajectory, your only job is to extract the fields: {outputs}."
        )
        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            extract_instructions,
        ).append("trajectory", dspy.InputField(desc="The history of the conversation. There is enough context to produce the final output"), type_=History)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def forward(self, **input_args):
        max_iters = input_args.pop("max_iters", self.max_iters)

        # Check for existing history in input_args, otherwise start empty
        trajectory = input_args.pop("trajectory", None)
        if trajectory is None:
            trajectory = History(messages=[], mode="raw")

        for _ in range(max_iters):
            try:
                pred, trajectory = self._call_with_potential_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            # Add the agent's action to trajectory
            trajectory, tool_call_id = self._append_action(
                trajectory,
                thought=pred.next_thought,
                tool_name=pred.next_tool_name,
                tool_args=pred.next_tool_args,
            )

            # Execute tool and get observation
            try:
                observation = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                observation = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            # Add observation to trajectory
            trajectory = self._append_observation(trajectory, observation, tool_call_id)

            if pred.next_tool_name == "finish":
                break

        extract, trajectory = self._call_with_potential_truncation(self.extract, trajectory, **input_args)

        # Add the extract step to the trajectory
        trajectory = self._append_extract(trajectory, extract)

        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args):
        max_iters = input_args.pop("max_iters", self.max_iters)

        # Check for existing history in input_args, otherwise start empty
        trajectory = input_args.pop("trajectory", None)
        if trajectory is None:
            trajectory = History(messages=[], mode="raw")

        for _ in range(max_iters):
            try:
                pred, trajectory = await self._async_call_with_potential_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            # Add the agent's action to trajectory
            trajectory, tool_call_id = self._append_action(
                trajectory,
                thought=pred.next_thought,
                tool_name=pred.next_tool_name,
                tool_args=pred.next_tool_args,
            )

            # Execute tool and get observation
            try:
                observation = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                observation = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            # Add observation to trajectory
            trajectory = self._append_observation(trajectory, observation, tool_call_id)

            if pred.next_tool_name == "finish":
                break

        extract, trajectory = await self._async_call_with_potential_truncation(self.extract, trajectory, **input_args)

        # Add the extract step to the trajectory
        trajectory = self._append_extract(trajectory, extract)

        return dspy.Prediction(trajectory=trajectory, **extract)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _append_action(self, trajectory: History, thought: str, tool_name: str, tool_args: dict) -> tuple[History, str]:
        """Append an action (thought + tool call) to the trajectory.
        
        Returns:
            Tuple of (updated trajectory, tool_call_id for matching with observation)
        """
        tool_call_id = self._generate_tool_call_id()
        new_msg = {
            "role": "assistant",
            "content": thought,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }
            ],
        }
        return trajectory.with_messages([new_msg]), tool_call_id

    def _append_observation(self, trajectory: History, observation: Any, tool_call_id: str) -> History:
        """Append a tool response to the trajectory."""
        if isinstance(observation, str):
            content = observation
        else:
            try:
                content = json.dumps(observation)
            except (TypeError, ValueError):
                content = str(observation)

        new_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        return trajectory.with_messages([new_msg])

    def _append_extract(self, trajectory: History, extract) -> History:
        """Append the extract step (final reasoning and outputs) to the trajectory."""
        extract_dict = dict(extract)
        reasoning = extract_dict.pop("reasoning", None)

        content_parts = []
        if reasoning:
            content_parts.append(f"Reasoning: {reasoning}")
        for key, value in extract_dict.items():
            if isinstance(value, str):
                content_parts.append(f"{key}: {value}")
            else:
                try:
                    content_parts.append(f"{key}: {json.dumps(value)}")
                except (TypeError, ValueError):
                    content_parts.append(f"{key}: {value}")

        new_msg = {
            "role": "assistant",
            "content": "\n".join(content_parts),
        }
        return trajectory.with_messages([new_msg])

    def _call_with_potential_truncation(self, module, trajectory: History, **input_args) -> tuple[Any, History]:
        """Call module with trajectory, truncating if context window exceeded.
        
        Returns:
            Tuple of (module result, potentially truncated trajectory)
        """
        for _ in range(3):
            try:
                return module(**input_args, trajectory=trajectory), trajectory
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)
        return None, trajectory

    async def _async_call_with_potential_truncation(self, module, trajectory: History, **input_args) -> tuple[Any, History]:
        """Call module with trajectory, truncating if context window exceeded.
        
        Returns:
            Tuple of (module result, potentially truncated trajectory)
        """
        for _ in range(3):
            try:
                return await module.acall(**input_args, trajectory=trajectory), trajectory
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)
        return None, trajectory

    def truncate_trajectory(self, trajectory: History) -> History:
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        For tool call format, we remove pairs of messages (assistant + tool) together.
        """
        if len(trajectory.messages) < 2:
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        # Remove the oldest pair (assistant message with tool_calls + tool response)
        messages = list(trajectory.messages)
        if messages and messages[0].get("role") == "assistant" and messages[0].get("tool_calls"):
            # Remove assistant + following tool message(s)
            messages = messages[1:]
            while messages and messages[0].get("role") == "tool":
                messages = messages[1:]
        else:
            # Fallback: just remove the first message
            messages = messages[1:]

        return History(messages=messages, mode="raw")


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
