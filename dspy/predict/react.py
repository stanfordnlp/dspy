from typing import Any, Callable, Literal, get_origin

from jsonschema import ValidationError, validate
from litellm import supports_function_calling
from pydantic import BaseModel

import dspy
from dspy.primitives.program import Module
from dspy.primitives.tool import Tool
from dspy.signatures.signature import ensure_signature


class ReAct(Module):
    def __init__(self, signature, tools: list[Callable], max_iters=5, use_litellm_tool_calling=None):
        """
        `tools` is either a list of functions, callable classes, or `dspy.Tool` instances.
        """

        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.use_litellm_tool_calling = use_litellm_tool_calling
        tools = [t if isinstance(t, Tool) else Tool.from_function(t) for t in tools]

        self.tools_in_litellm_format = [tool.convert_to_litellm_tool_format() for tool in tools]
        self.tools = {tool.name: tool for tool in tools}
        self.tools["finish"] = Tool(
            name="finish",
            desc="Signals that the final outputs, i.e. {outputs}, are now available and marks the task as complete.",
            parameters={},
            func=lambda **kwargs: "Completed.",
        )

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instruction = [f"{signature.instructions}\n"] if signature.instructions else []

        instruction_custom_tool_calling = list(instruction)
        instruction_native_tool_calling = list(instruction)

        instruction_custom_tool_calling.extend(
            [
                f"You will be given {inputs} and your goal is to finish with {outputs}.\n",
                "To do this, you will interleave Thought, Tool Name, and Tool Args, and receive a resulting Observation.\n",
                "Thought can reason about the current situation, and Tool Name can be the following types:\n",
            ]
        )
        instruction_native_tool_calling.extend(
            [
                f"You will be given {inputs} and your goal is to finish with {outputs}.\n",
                "To help you reach this goal, you will be given a list of tools, and you will need to think about "
                "which tool to use in order to progress towards the goal.\n",
                "Your selected tool will be executed with your suggested arguments, and the result will be sent along "
                "in the next hop to you.\n",
                "You will then repeat this process until you decide no more tools are needed and provide the final "
                "outputs.\n",
            ]
        )
        for idx, tool in enumerate(self.tools.values()):
            args = getattr(tool, "parameters", [])
            desc = (f", whose description is <desc>{tool.desc}</desc>." if tool.desc else ".").replace("\n", "  ")
            desc += f" It takes arguments {args} in JSON format."
            instruction_custom_tool_calling.append(f"({idx+1}) {tool.name}{desc}")

        custom_react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instruction_custom_tool_calling))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(self.tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )
        native_react_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields}, "\n".join(instruction_native_tool_calling)
        ).append("trajectory", dspy.InputField(), type_=str)

        self.react_with_custom_tool_calling = dspy.Predict(custom_react_signature)
        self.react_with_native_tool_calling = dspy.PredictWithTools(
            native_react_signature, tools=self.tools_in_litellm_format
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields}, signature.instructions
        ).append("trajectory", dspy.InputField(), type_=str)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any], last_iteration: bool):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_fields(trajectory_signature, trajectory, role="user")

    def _validate_and_parse_tool_arg(self, tool, arg_name, arg_value):
        tool_parameters = tool.parameters
        if arg_name not in tool_parameters:
            raise ValueError(f"Recevied unexpected argument {arg_name} for tool {tool.name}")

        try:
            validate(instance=arg_value, schema=tool_parameters[arg_name])
        except ValidationError as e:
            raise ValueError(f"Received invalid argument {arg_name} for tool {tool.name}: {e}")

        if arg_name in tool.arg_types:
            arg_type = tool.arg_types[arg_name]
            if isinstance((origin := get_origin(arg_type) or arg_type), type) and issubclass(origin, BaseModel):
                return arg_type.model_validate(arg_value)
            else:
                return arg_value
        else:
            return arg_value

    def _forward_with_custom_tool_calling(self, **input_args):
        trajectory = {}
        for idx in range(self.max_iters):
            pred = self.react_with_custom_tool_calling(
                **input_args, trajectory=self._format_trajectory(trajectory, last_iteration=(idx == self.max_iters - 1))
            )

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                parsed_tool_args = {}
                tool = self.tools[pred.next_tool_name]
                for k, v in pred.next_tool_args.items():
                    parsed_tool_args[k] = self._validate_and_parse_tool_arg(tool, k, v)
                trajectory[f"observation_{idx}"] = tool(**parsed_tool_args)
            except Exception as e:
                trajectory[f"observation_{idx}"] = f"Failed to execute: {e}"

            if pred.next_tool_name == "finish":
                break

        extract = self.extract(**input_args, trajectory=self._format_trajectory(trajectory, last_iteration=False))
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _forward_with_litellm_tool_calling(self, **input_args):
        trajectory = {}
        for idx in range(self.max_iters):
            pred = self.react_with_native_tool_calling(
                **input_args, trajectory=self._format_trajectory(trajectory, last_iteration=(idx == self.max_iters - 1))
            )

            if not hasattr(pred, "tool_calls"):
                # No more tools are needed, which means LLM decides that we have reached the final outputs.
                trajectory["finish"] = pred.toDict()
                break

            trajectory[f"tool_name_{idx}"] = []
            trajectory[f"tool_args_{idx}"] = []
            trajectory[f"observation_{idx}"] = []

            for tool_call in pred.tool_calls:
                tool_name = tool_call["tool_name"]
                tool_args = tool_call["tool_args"]
                trajectory[f"tool_name_{idx}"].append(tool_name)
                trajectory[f"tool_args_{idx}"].append(tool_args)

                parsed_tool_args = {}
                tool = self.tools[tool_name]
                for k, v in tool_args.items():
                    parsed_tool_args[k] = self._validate_and_parse_tool_arg(tool, k, v)

                try:
                    trajectory[f"observation_{idx}"].append(tool(**parsed_tool_args))
                except Exception as e:
                    trajectory[f"observation_{idx}"].append(f"Failed to execute: {e}")

        extract = self.extract(**input_args, trajectory=self._format_trajectory(trajectory, last_iteration=False))
        return dspy.Prediction(trajectory=trajectory, **extract)

    def forward(self, **input_args):
        lm = dspy.settings.lm
        if self.use_litellm_tool_calling and not supports_function_calling(lm.model):
            raise ValueError(
                f"Your lm {lm.model} doesn't support litellm tool calling. Please set `use_litellm_tool_calling=False` "
                "or choose an lm that supports litellm tool calling."
            )
        use_litellm_tool_calling = self.use_litellm_tool_calling
        if use_litellm_tool_calling is None:
            use_litellm_tool_calling = supports_function_calling(lm.model)

        if use_litellm_tool_calling:
            return self._forward_with_litellm_tool_calling(**input_args)
        else:
            return self._forward_with_custom_tool_calling(**input_args)


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
