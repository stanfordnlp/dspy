import logging
from typing import Any, Callable, Literal, get_origin

from litellm import ContextWindowExceededError
from pydantic import BaseModel

import dspy
from dspy.primitives.program import Module
from dspy.primitives.tool import Tool
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)


class ReAct(Module):
    def __init__(self, signature, tools: list[Callable], max_iters=5):
        """
        `tools` is either a list of functions, callable classes, or `dspy.Tool` instances.
        """

        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You will be given {inputs} and your goal is to finish with {outputs}.\n",
                "To do this, you will interleave Thought, Tool Name, and Tool Args, and receive a resulting Observation.\n",
                "Thought can reason about the current situation, and Tool Name can be the following types:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda **kwargs: "Completed.",
            name="finish",
            desc=f"Signals that the final outputs, i.e. {outputs}, are now available and marks the task as complete.",
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            args = getattr(tool, "args")
            desc = (f", whose description is <desc>{tool.desc}</desc>." if tool.desc else ".").replace("\n", "  ")
            desc += f" It takes arguments {args} in JSON format."
            instr.append(f"({idx+1}) {tool.name}{desc}")

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
        return adapter.format_fields(trajectory_signature, trajectory, role="user")

    def forward(self, **input_args):
        trajectory = {}
        for idx in range(self.max_iters):
            pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                parsed_tool_args = {}
                tool = self.tools[pred.next_tool_name]
                for k, v in pred.next_tool_args.items():
                    if hasattr(tool, "arg_types") and k in tool.arg_types:
                        arg_type = tool.arg_types[k]
                        if isinstance((origin := get_origin(arg_type) or arg_type), type) and issubclass(
                            origin, BaseModel
                        ):
                            parsed_tool_args[k] = arg_type.model_validate(v)
                            continue
                    parsed_tool_args[k] = v
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**parsed_tool_args)
            except Exception as e:
                trajectory[f"observation_{idx}"] = f"Failed to execute: {e}"

            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        while True:
            try:
                return module(
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
        if len(keys) < 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory


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
or a big dictionary, and have each adatper format it accordingly.

Trajectories also affect meta-programming modules that view the trace later. It's inefficient O(n^2) to view the
trace of every module repeating the prefix.


TOPIC 02: Handling default arguments in the Tool class.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling descriptions and casting.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?


TOPIC 06: Idiomatically allowing tools that maintain state across iterations, but not across different `forward` calls.
    * So the tool would be newly initialized at the start of each `forward` call, but maintain state across iterations.
    * This is pretty useful for allowing the agent to keep notes or count certain things, etc.

TOPIC 07: Make max_iters a bit more expressive.
    * Allow passing `max_iters` in forward to overwrite the default.
    * Get rid of `last_iteration: bool` in the format function. It's not necessary now.
"""
