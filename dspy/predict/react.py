import dspy
import inspect

from pydantic import BaseModel
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature
from dspy.adapters.json_adapter import get_annotation_name
from typing import Callable, Any, get_type_hints, get_origin, Literal

class Tool:
    def __init__(self, func: Callable, name: str = None, desc: str = None, args: dict[str, Any] = None):
        annotations_func = func if inspect.isfunction(func) else func.__call__
        self.func = func
        self.name = name or getattr(func, '__name__', type(func).__name__)
        self.desc = desc or getattr(func, '__doc__', None) or getattr(annotations_func, '__doc__', "No description")
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
        tools["finish"] = Tool(func=lambda **kwargs: kwargs, name="finish", desc=finish_desc, args=finish_args)

        for idx, tool in enumerate(tools.values()):
            desc = tool.desc.replace("\n", "  ")
            args = tool.args if hasattr(tool, 'args') else str({tool.input_variable: str})
            desc = f"whose description is <desc>{desc}</desc>. It takes arguments {args} in JSON format."
            instr.append(f"({idx+1}) {tool.name}, {desc}")

        signature_ = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = (
            dspy.Signature({**signature.input_fields, **signature.output_fields})
            .append("trajectory", dspy.InputField(), type_=str)
        )

        self.tools = tools
        self.react = dspy.Predict(signature_)
        self.extract = dspy.ChainOfThought(fallback_signature)
    
    def forward(self, **input_args):
        trajectory = {}

        def format(trajectory_: dict[str, Any], last_iteration: bool):
            adapter = dspy.settings.adapter or dspy.ChatAdapter()
            blob = adapter.format_fields(dspy.Signature(f"{', '.join(trajectory_.keys())} -> x"), trajectory_)
            warning = f"\n\nWarning: The maximum number of iterations ({self.max_iters}) has been reached."
            warning += " You must now produce the finish action."
            return blob + (warning if last_iteration else "")

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
