import dsp
import dspy
from dspy.signatures.signature import ensure_signature

from ..primitives.program import Module
from .predict import Predict

# TODO: Simplify a lot.
# TODO: Divide Action and Action Input like langchain does for ReAct.

# TODO: There's a lot of value in having a stopping condition in the LM calls at `\n\nObservation:`


class ReAct(Module):
    def __init__(self, signature, max_iters=5, num_results=3, tools=None):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        self.tools = tools or [dspy.Retrieve(k=num_results)]
        self.tools = {tool.name: tool for tool in self.tools}

        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        assert len(self.output_fields) == 1, "ReAct only supports one output field."

        inputs_ = ", ".join([f"`{k}`" for k in self.input_fields.keys()])
        outputs_ = ", ".join([f"`{k}`" for k in self.output_fields.keys()])

        instr = []
        
        if self.signature.instructions is not None:
            instr.append(f"{self.signature.instructions}\n")
        
        instr.extend([
            f"You will be given {inputs_} and you will respond with {outputs_}.\n",
            "To do this, you will interleave Thought, Action, and Observation steps.\n",
            "Thought can reason about the current situation, and Action can be the following types:\n",
        ])

        self.tools["Finish"] = dspy.Example(
            name="Finish",
            input_variable=outputs_.strip("`"),
            desc=f"returns the final {outputs_} and finishes the task",
        )

        for idx, tool in enumerate(self.tools):
            tool = self.tools[tool]
            instr.append(
                f"({idx+1}) {tool.name}[{tool.input_variable}], which {tool.desc}",
            )

        instr = "\n".join(instr)
        self.react = [
            Predict(dspy.Signature(self._generate_signature(i), instr))
            for i in range(1, max_iters + 1)
        ]

    def _generate_signature(self, iters):
        signature_dict = {}
        for key, val in self.input_fields.items():
            signature_dict[key] = val

        for j in range(1, iters + 1):
            signature_dict[f"Thought_{j}"] = dspy.OutputField(
                prefix=f"Thought {j}:",
                desc="next steps to take based on last observation",
            )

            tool_list = " or ".join(
                [
                    f"{tool.name}[{tool.input_variable}]"
                    for tool in self.tools.values()
                    if tool.name != "Finish"
                ],
            )
            signature_dict[f"Action_{j}"] = dspy.OutputField(
                prefix=f"Action {j}:",
                desc=f"always either {tool_list} or, when done, Finish[answer]",
            )

            if j < iters:
                signature_dict[f"Observation_{j}"] = dspy.OutputField(
                    prefix=f"Observation {j}:",
                    desc="observations based on action",
                    format=dsp.passages2text,
                )

        return signature_dict

    def act(self, output, hop):
        try:
            action = output[f"Action_{hop+1}"]
            action_name, action_val = action.strip().split("\n")[0].split("[", 1)
            action_val = action_val.rsplit("]", 1)[0]

            if action_name == "Finish":
                return action_val

            result = self.tools[action_name](action_val)
            # Handle the case where 'passages' attribute is missing
            output[f"Observation_{hop+1}"] = getattr(result, "passages", result)

        except Exception:
            output[f"Observation_{hop+1}"] = (
                "Failed to parse action. Bad formatting or incorrect action name."
            )
            # raise e

    def forward(self, **kwargs):
        args = {key: kwargs[key] for key in self.input_fields.keys() if key in kwargs}

        for hop in range(self.max_iters):
            # with dspy.settings.context(show_guidelines=(i <= 2)):
            output = self.react[hop](**args)

            if action_val := self.act(output, hop):
                break
            args.update(output)

        observations = [args[key] for key in args if key.startswith("Observation")]
        
        # assumes only 1 output field for now - TODO: handling for multiple output fields
        return dspy.Prediction(observations=observations, **{list(self.output_fields.keys())[0]: action_val or ""})
