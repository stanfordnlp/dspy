from copy import deepcopy

from pydantic.fields import FieldInfo

import dspy
from dspy.predict.avatar.models import Action, ActionOutput, Tool
from dspy.predict.avatar.signatures import Actor
from dspy.signatures.signature import ensure_signature


def get_number_with_suffix(number: int) -> str:
    """Converts a number to its ordinal string representation (1st, 2nd, 3rd, etc.)."""
    if number == 1:
        return "1st"
    elif number == 2:
        return "2nd"
    elif number == 3:
        return "3rd"
    else:
        return f"{number}th"


class Avatar(dspy.Module):
    """An agent-like DSPy module that uses tools iteratively to accomplish a task defined by a signature."""
    
    def __init__(
        self,
        signature,
        tools,
        max_iters=3,
        verbose=False,
    ):
        """Initializes the Avatar agent with a task signature and available tools.
        
        Sets up the actor signature by appending input fields from the provided signature,
        adds a special "Finish" tool to the tool list, and creates a TypedPredictor for the actor.
        The actor signature is built dynamically by prepending input fields in reverse order.
        """
        self.signature = ensure_signature(signature)
        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        self.finish_tool = Tool(
            tool=None,
            name="Finish",
            desc="returns the final output and finishes the task",
        )

        self.tools = tools + [self.finish_tool]
        self.actor_signature = Actor

        for field in list(self.input_fields.keys())[::-1]:
            self.actor_signature = self.actor_signature.append(
                field,
                self._get_field(self.input_fields[field]),
                type_=self.input_fields[field].annotation,
            )

        self.verbose = verbose
        self.max_iters = max_iters
        self.actor = dspy.TypedPredictor(self.actor_signature)

        self.actor_clone = deepcopy(self.actor)

    def _get_field(self, field_info: FieldInfo):
        """Reconstructs a DSPy field (InputField or OutputField) from a FieldInfo object.
        
        Extracts the field type, prefix, description, and optional format from the json_schema_extra
        metadata and creates the appropriate DSPy field instance.
        
        Raises:
            ValueError: If the field type in json_schema_extra is not 'input' or 'output'.
        """
        if field_info.json_schema_extra["__dspy_field_type"] == "input":
            return dspy.InputField(
                prefix=field_info.json_schema_extra["prefix"],
                desc=field_info.json_schema_extra["desc"],
                format=field_info.json_schema_extra["format"] if "format" in field_info.json_schema_extra else None,
            )
        elif field_info.json_schema_extra["__dspy_field_type"] == "output":
            return dspy.OutputField(
                prefix=field_info.json_schema_extra["prefix"],
                desc=field_info.json_schema_extra["desc"],
                format=field_info.json_schema_extra["format"] if "format" in field_info.json_schema_extra else None,
            )
        else:
            raise ValueError(f"Unknown field type: {field_info.json_schema_extra['__dspy_field_type']}")

    def _update_signature(self, idx: int, omit_action: bool = False):
        """Dynamically extends the actor signature with action and result fields for the current iteration.
        
        Converts the previous action field from output to input, adds a result field for that action,
        and either appends the next action field (if omit_action is False) or appends the final output
        fields (if omit_action is True, indicating task completion).
        """
        self.actor.signature = self.actor.signature.with_updated_fields(
            f"action_{idx}", Action, __dspy_field_type="input"
        )

        self.actor.signature = self.actor.signature.append(
            f"result_{idx}",
            dspy.InputField(
                prefix=f"Result {idx}:",
                desc=f"{get_number_with_suffix(idx)} result",
                type_=str,
            ),
        )

        if omit_action:
            for field in list(self.output_fields.keys()):
                self.actor.signature = self.actor.signature.append(
                    field,
                    self._get_field(self.output_fields[field]),
                    type_=self.output_fields[field].annotation,
                )
        else:
            self.actor.signature = self.actor.signature.append(
                f"action_{idx+1}",
                dspy.OutputField(
                    prefix=f"Action {idx+1}:",
                    desc=f"{get_number_with_suffix(idx+1)} action to taken",
                ),
            )
            self.actor.signature = self.actor.signature.with_updated_fields(
                f"action_{idx+1}",
                Action,
            )

    def _call_tool(self, tool_name: str, tool_input_query: str) -> str:
        """Executes a tool by name with the provided input query and returns its output."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.tool.run(tool_input_query)

    def forward(self, **kwargs):
        """Executes the agent's main loop, iteratively selecting and running tools until task completion.
        
        The agent repeatedly calls the actor to select an action, executes the corresponding tool,
        and updates the signature with the action and result. This continues until the "Finish" tool
        is selected or max_iters is reached. The signature is dynamically extended with each iteration's
        action and result fields, building up the context for subsequent decisions.
        """
        if self.verbose:
            print("Starting the task...")

        args = {
            "goal": self.signature.__doc__,
            "tools": [tool.name for tool in self.tools],
        }

        for key in self.input_fields.keys():
            if key in kwargs:
                args[key] = kwargs[key]

        idx = 1
        tool_name = None
        action_results: list[ActionOutput] = []
        max_iters = None if "max_iters" not in kwargs else kwargs["max_iters"]

        while tool_name != "Finish" and (max_iters > 0 if max_iters else True):
            actor_output = self.actor(**args)
            action = getattr(actor_output, f"action_{idx}")

            tool_name = action.tool_name
            tool_input_query = action.tool_input_query

            if self.verbose:
                print(f"Action {idx}: {tool_name} ({tool_input_query})")

            if tool_name != "Finish":
                tool_output = self._call_tool(tool_name, tool_input_query)
                action_results.append(
                    ActionOutput(tool_name=tool_name, tool_input_query=tool_input_query, tool_output=tool_output)
                )

                self._update_signature(idx)

                args[f"action_{idx}"] = action
                args[f"result_{idx}"] = tool_output
            else:
                self._update_signature(idx, omit_action=True)

                args[f"action_{idx}"] = action
                args[f"result_{idx}"] = "Gathered all information needed to finish the task."
                break

            idx += 1

            if max_iters:
                max_iters -= 1

        final_answer = self.actor(**args)
        self.actor = deepcopy(self.actor_clone)

        return dspy.Prediction(
            **{key: getattr(final_answer, key) for key in self.output_fields.keys()},
            actions=action_results,
        )
