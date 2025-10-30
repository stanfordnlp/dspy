import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

from litellm import ContextWindowExceededError

import dspy
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class ReAct(Module):
    def __init__(
        self,
        signature: type["Signature"],
        tools: list[Callable],
        max_iters: int = 20,
        parallel_tool_calls: bool = False,
    ):
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
            parallel_tool_calls (Optional[bool]): Whether to enable parallel tool execution. Defaults to False.
                When True, allows the LLM to request multiple tool calls in a single turn that execute concurrently.
                When False, maintains sequential execution (one tool per turn).

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
        self.parallel_tool_calls = parallel_tool_calls

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        if parallel_tool_calls:
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
        else:
            instr.extend(
                [
                    f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                    f"Your goal is to use one of the supplied tools to collect any necessary information for producing {outputs}.\n",
                    "To do this, you will interleave next_thought and next_tool_name in each turn, and also when finishing the task.",
                    "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                    "When writing next_thought, you may reason about the current situation and plan for future steps.",
                    "When selecting next_tool_name, the value must be one of:\n",
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

        if parallel_tool_calls:
            instr.append(
                "When providing `next_tool_calls`, provide a list of tool calls. Each tool call should be a dictionary with 'name' and 'args' keys. "
                "The 'name' must be one of the tool names listed above, and 'args' must be a dictionary in JSON format containing the arguments for that tool."
            )

            react_signature = (
                dspy.Signature({**signature.input_fields}, "\n".join(instr))
                .append("trajectory", dspy.InputField(), type_=str)
                .append("next_thought", dspy.OutputField(), type_=str)
                .append("next_tool_calls", dspy.OutputField(), type_=ToolCalls)
            )
        else:
            instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

            react_signature = (
                dspy.Signature({**signature.input_fields}, "\n".join(instr))
                .append("trajectory", dspy.InputField(), type_=str)
                .append("next_thought", dspy.OutputField(), type_=str)
                .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
                .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
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

    def _split_finish_tools(self, tool_calls: list) -> tuple[list, list]:
        """Split tool calls into non-finish and finish calls.

        Returns:
            Tuple of (non_finish_calls, finish_calls)
        """
        non_finish = [tc for tc in tool_calls if tc.name != "finish"]
        finish_calls = [tc for tc in tool_calls if tc.name == "finish"]
        return non_finish, finish_calls

    def _store_parallel_trajectory(self, trajectory: dict, idx: int, tool_calls: list, observations: list):
        """Store tool calls and observations in the trajectory for a parallel step."""
        trajectory[f"tool_calls_{idx}"] = [{"name": tc.name, "args": tc.args} for tc in tool_calls]
        trajectory[f"observations_{idx}"] = [
            {"tool": tc.name, "result": obs}
            for tc, obs in zip(tool_calls, observations, strict=True)
        ]

    def _run_sequential_step(self, pred: Any, trajectory: dict, idx: int) -> bool:
        """Execute one sequential iteration step (sync) and update trajectory.

        Returns:
            True if the loop should break (finish tool was called), False otherwise.
        """
        trajectory[f"tool_name_{idx}"] = pred.next_tool_name
        trajectory[f"tool_args_{idx}"] = pred.next_tool_args

        try:
            observation = self.tools[pred.next_tool_name](**pred.next_tool_args)
        except Exception as err:
            observation = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"
        trajectory[f"observation_{idx}"] = observation

        return pred.next_tool_name == "finish"

    async def _run_sequential_step_async(self, pred: Any, trajectory: dict, idx: int) -> bool:
        """Execute one sequential iteration step (async) and update trajectory.

        Returns:
            True if the loop should break (finish tool was called), False otherwise.
        """
        trajectory[f"tool_name_{idx}"] = pred.next_tool_name
        trajectory[f"tool_args_{idx}"] = pred.next_tool_args

        try:
            observation = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
        except Exception as err:
            observation = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"
        trajectory[f"observation_{idx}"] = observation

        return pred.next_tool_name == "finish"

    def _run_parallel_step(self, pred: Any, trajectory: dict, idx: int) -> bool:
        """Execute one parallel iteration step (sync) and update trajectory.

        Returns:
            True if the loop should break (finish tool was called), False otherwise.
        """
        tool_calls = pred.next_tool_calls.tool_calls
        non_finish, finish_calls = self._split_finish_tools(tool_calls)

        if finish_calls:
            pre_obs = self._execute_tools_parallel(non_finish) if non_finish else []
            finish_obs = self._execute_tools_parallel(finish_calls)
            self._store_parallel_trajectory(trajectory, idx, non_finish + finish_calls, pre_obs + finish_obs)
            return True

        observations = self._execute_tools_parallel(tool_calls)
        self._store_parallel_trajectory(trajectory, idx, tool_calls, observations)
        return False

    async def _run_parallel_step_async(self, pred: Any, trajectory: dict, idx: int) -> bool:
        """Execute one parallel iteration step (async) and update trajectory.

        Returns:
            True if the loop should break (finish tool was called), False otherwise.
        """
        tool_calls = pred.next_tool_calls.tool_calls
        non_finish, finish_calls = self._split_finish_tools(tool_calls)

        if finish_calls:
            pre_obs = await self._execute_tools_parallel_async(non_finish) if non_finish else []
            finish_obs = await self._execute_tools_parallel_async(finish_calls)
            self._store_parallel_trajectory(trajectory, idx, non_finish + finish_calls, pre_obs + finish_obs)
            return True

        observations = await self._execute_tools_parallel_async(tool_calls)
        self._store_parallel_trajectory(trajectory, idx, tool_calls, observations)
        return False

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

            if self.parallel_tool_calls:
                if self._run_parallel_step(pred, trajectory, idx):
                    break
            else:
                if self._run_sequential_step(pred, trajectory, idx):
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

            if self.parallel_tool_calls:
                if await self._run_parallel_step_async(pred, trajectory, idx):
                    break
            else:
                if await self._run_sequential_step_async(pred, trajectory, idx):
                    break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _execute_tools_parallel(self, tool_calls: list) -> list[Any]:
        """Execute tools using ToolCall.execute() method.

        Args:
            tool_calls: List of ToolCall objects

        Returns:
            List of observations in the same order as tool_calls
        """
        if not tool_calls:
            return []

        # If there's only one tool call, execute directly without parallel overhead
        if len(tool_calls) == 1:
            try:
                return [tool_calls[0].execute(self.tools)]
            except Exception as err:
                return [f"Execution error in {tool_calls[0].name}: {_fmt_exc(err)}"]

        # Use ParallelExecutor which handles context propagation automatically
        # straggler_limit=0 disables retry of slow tasks
        executor = ParallelExecutor(num_threads=len(tool_calls), disable_progress_bar=True, straggler_limit=0)

        def execute_single_tool(tool_call) -> Any:
            try:
                return tool_call.execute(self.tools)
            except Exception as err:
                return f"Execution error in {tool_call.name}: {_fmt_exc(err)}"

        # Execute tools in parallel - ParallelExecutor handles context propagation
        observations = executor.execute(execute_single_tool, tool_calls)

        return observations

    async def _execute_tools_parallel_async(self, tool_calls: list) -> list[Any]:
        """Execute tools asynchronously using Tool.acall() method.

        Args:
            tool_calls: List of ToolCall objects

        Returns:
            List of observations in the same order as tool_calls
        """
        if not tool_calls:
            return []

        # If there's only one tool call, execute directly without gather overhead
        if len(tool_calls) == 1:
            try:
                tool = self.tools[tool_calls[0].name]
                return [await tool.acall(**tool_calls[0].args)]
            except Exception as err:
                return [f"Execution error in {tool_calls[0].name}: {_fmt_exc(err)}"]

        async def execute_single_tool(tool_call) -> Any:
            try:
                tool = self.tools[tool_call.name]
                return await tool.acall(**tool_call.args)
            except Exception as err:
                return f"Execution error in {tool_call.name}: {_fmt_exc(err)}"

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
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

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
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())

        # Determine keys per iteration based on mode
        keys_per_iteration = 3 if self.parallel_tool_calls else 4

        if len(keys) < keys_per_iteration:
            # Every iteration has either 3 keys (parallel: thought, tool_calls, observations)
            # or 4 keys (sequential: thought, tool_name, tool_args, observation)
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one iteration."
            )

        for key in keys[:keys_per_iteration]:
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
