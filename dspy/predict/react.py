import dspy
import inspect

from pydantic import BaseModel
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature
from dspy.adapters.json_adapter import get_annotation_name
from typing import Callable, Any, get_type_hints, get_origin, Literal

class Tool:
    def __init__(self, func: Callable, name: str = None, desc: str = None, args: dict[str, Any] = None):
        annotations_func = func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        self.func = func
        self.name = name or getattr(func, '__name__', type(func).__name__)
        self.desc = desc or getattr(func, '__doc__', None) or getattr(annotations_func, '__doc__', "")
        self.args = {
            k: v.schema() if isinstance((origin := get_origin(v) or v), type) and issubclass(origin, BaseModel)
            else get_annotation_name(v)
            for k, v in (args or get_type_hints(annotations_func)).items() if k != 'return'
        }

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ReAct(Module):
    def __init__(self, signature, tools: list[Callable], max_iters=5):
        """
        Tools is either a list of functions, callable classes, or dspy.Tool instances.
        """

        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) or hasattr(t, 'input_variable') else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs_ = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs_ = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend([
            f"You will be given {inputs_} and your goal is to finish with {outputs_}.\n",
            "To do this, you will interleave Thought, Tool Name, and Tool Args, and receive a resulting Observation.\n",
            "Thought can reason about the current situation, and Tool Name can be the following types:\n",
        ])

        finish_desc = f"Signals that the final outputs, i.e. {outputs_}, are now available and marks the task as complete."
        finish_args = {} #k: v.annotation for k, v in signature.output_fields.items()}
        tools["finish"] = Tool(func=lambda **kwargs: "Completed.", name="finish", desc=finish_desc, args=finish_args)

        for idx, tool in enumerate(tools.values()):
            args = tool.args if hasattr(tool, 'args') else str({tool.input_variable: str})
            desc = (f", whose description is <desc>{tool.desc}</desc>." if tool.desc else ".").replace('\n', "  ")
            desc += f" It takes arguments {args} in JSON format."
            instr.append(f"({idx+1}) {tool.name}{desc}")

        signature_ = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = (
            dspy.Signature({**signature.input_fields, **signature.output_fields}, signature.instructions)
            .append("trajectory", dspy.InputField(), type_=str)
        )

        self.tools = tools
        self.react = dspy.Predict(signature_)
        self.extract = dspy.ChainOfThought(fallback_signature)
    
    def forward(self, **input_args):
        trajectory = {}

        def format(trajectory_: dict[str, Any], last_iteration: bool):
            adapter = dspy.settings.adapter or dspy.ChatAdapter()
            signature_ = dspy.Signature(f"{', '.join(trajectory_.keys())} -> x")
            return adapter.format_fields(signature_, trajectory_, role='user')

        for idx in range(self.max_iters):
            pred = self.react(**input_args, trajectory=format(trajectory, last_iteration=(idx == self.max_iters-1)))

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as e:
                trajectory[f"observation_{idx}"] = f"Failed to execute: {e}"

            if pred.next_tool_name == "finish":
                break
        
        extract = self.extract(**input_args, trajectory=format(trajectory, last_iteration=False))
        return dspy.Prediction(trajectory=trajectory, **extract)


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


TOPIC 02: Handling default arguments in the Tool class.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling descriptions and casting.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?
"""
