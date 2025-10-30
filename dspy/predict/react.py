import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

from litellm import ContextWindowExceededError

import dspy
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
                "To do this, you will interleave next_thought and next_tool_calls in each turn, and also when finishing the task.",
                "You can call multiple tools in parallel by providing multiple tool calls in next_tool_calls.",
                "After each set of tool calls, you receive resulting observations, which get appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting next_tool_calls, each tool must be one of:\n",
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
        instr.append(
            "When providing `next_tool_calls`, provide a list of tool calls. Each tool call should be a dictionary with 'name' and 'args' keys. "
            "The 'name' must be one of the tool names listed above, and 'args' must be a dictionary in JSON format containing the arguments for that tool."
        )

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_calls", dspy.OutputField(), type_=list[dict[str, Any]])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought

            # Parse tool calls - handle both list format and backward compatibility
            tool_calls = self._parse_tool_calls(pred.next_tool_calls)
            trajectory[f"tool_calls_{idx}"] = tool_calls

            # Execute tools in parallel
            observations = self._execute_tools_parallel(tool_calls)

            # Store observations as a structured format that includes tool names
            # This makes it easier for the LLM to understand which observation corresponds to which tool
            formatted_observations = []
            for tool_call, observation in zip(tool_calls, observations, strict=True):
                formatted_observations.append({
                    "tool": tool_call["name"],
                    "result": observation
                })
            trajectory[f"observations_{idx}"] = formatted_observations

            # Check if any tool call is "finish"
            if any(tc["name"] == "finish" for tc in tool_calls):
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought

            # Parse tool calls - handle both list format and backward compatibility
            tool_calls = self._parse_tool_calls(pred.next_tool_calls)
            trajectory[f"tool_calls_{idx}"] = tool_calls

            # Execute tools in parallel
            observations = await self._execute_tools_parallel_async(tool_calls)

            # Store observations as a structured format that includes tool names
            # This makes it easier for the LLM to understand which observation corresponds to which tool
            formatted_observations = []
            for tool_call, observation in zip(tool_calls, observations, strict=True):
                formatted_observations.append({
                    "tool": tool_call["name"],
                    "result": observation
                })
            trajectory[f"observations_{idx}"] = formatted_observations

            # Check if any tool call is "finish"
            if any(tc["name"] == "finish" for tc in tool_calls):
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _parse_tool_calls(self, tool_calls_data):
        """Parse tool calls from the prediction output.

        Handles both the new list format and provides backward compatibility.
        """
        # If it's already a list of dicts with 'name' and 'args', use it directly
        if isinstance(tool_calls_data, list):
            return tool_calls_data

        # Handle single dict case (shouldn't normally happen but for robustness)
        if isinstance(tool_calls_data, dict) and "name" in tool_calls_data and "args" in tool_calls_data:
            return [tool_calls_data]

        # If we got something unexpected, raise an error
        raise ValueError(f"Invalid tool_calls format: {tool_calls_data}")

    def _execute_tools_parallel(self, tool_calls: list[dict[str, Any]]) -> list[Any]:
        """Execute multiple tools in parallel using ThreadPoolExecutor.

        Args:
            tool_calls: List of tool call dicts, each with 'name' and 'args' keys

        Returns:
            List of observations in the same order as tool_calls
        """
        def execute_single_tool(tool_call: dict[str, Any]) -> Any:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            try:
                return self.tools[tool_name](**tool_args)
            except Exception as err:
                return f"Execution error in {tool_name}: {_fmt_exc(err)}"

        # Execute tools in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            observations = list(executor.map(execute_single_tool, tool_calls))

        return observations

    async def _execute_tools_parallel_async(self, tool_calls: list[dict[str, Any]]) -> list[Any]:
        """Execute multiple tools in parallel using asyncio.gather.

        Args:
            tool_calls: List of tool call dicts, each with 'name' and 'args' keys

        Returns:
            List of observations in the same order as tool_calls
        """
        async def execute_single_tool(tool_call: dict[str, Any]) -> Any:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            try:
                return await self.tools[tool_name].acall(**tool_args)
            except Exception as err:
                return f"Execution error in {tool_name}: {_fmt_exc(err)}"

        # Execute tools in parallel using asyncio.gather
        observations = await asyncio.gather(*[execute_single_tool(tc) for tc in tool_calls])

        return observations

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if len(keys) < 3:
            # Every iteration has 3 keys: thought, tool_calls, and observations.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one iteration."
            )

        for key in keys[:3]:
            trajectory.pop(key)

        return trajectory


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
