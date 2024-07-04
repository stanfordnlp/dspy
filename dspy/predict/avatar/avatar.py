import dspy

from copy import deepcopy

from typing import Mapping
from pydantic.fields import FieldInfo
from pydantic import create_model
from dspy.predict.avatar.models import Action, Tool
from dspy.predict.avatar.signatures import Actor
from dspy.signatures.signature import ensure_signature


class Avatar(dspy.Module):
    def __init__(
        self,
        signature,
        knowledge_base,
        tools,
        max_iters=3,
    ):
        self.signature = ensure_signature(signature)
        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        self.finish_tool = Tool(
            tool=None,
            name="Finish",
            input_variable=[],
            desc="returns the final output and finishes the task",
        )

        self.tools = tools + [self.finish_tool]
        self.actor_signature = Actor
        self.knowledge_base = knowledge_base

        for field in list(self.input_fields.keys())[::-1]:
            self.actor_signature = self.actor_signature.insert(
                -2,
                field,
                self._get_field(self.input_fields[field]),
            )

        self.max_iters = max_iters
        self.actor = dspy.TypedPredictor(self.actor_signature)


    def _get_field(self, field_info: FieldInfo):
        match field_info.json_schema_extra['__dspy_field_type']:
            case 'input':
                return dspy.InputField(
                    prefix=field_info.json_schema_extra['prefix'],
                    desc=field_info.json_schema_extra['desc'],
                    format=field_info.json_schema_extra['format'] if 'format' in field_info.json_schema_extra else None,
                )
            case 'output':
                return dspy.OutputField(
                    prefix=field_info.json_schema_extra['prefix'],
                    desc=field_info.json_schema_extra['desc'],
                    format=field_info.json_schema_extra['format'] if 'format' in field_info.json_schema_extra else None,
                )
            case _:
                raise ValueError(f"Unknown field type: {field_info.json_schema_extra['__dspy_field_type']}") 


    def _update_signature(self, idx: int, omit_action: bool = False):
        self.actor_signature = self.actor_signature.with_updated_fields(
            f"action_{idx}", 
            Action, 
            __dspy_field_type="input"
        )

        self.actor_signature = self.actor_signature.append(
            f"result_{idx}",
            dspy.InputField(
                prefix=f"Result {idx}:",
                desc=f"{idx}th result",
            )
        )
        match omit_action:
            case True:
                for field in list(self.output_fields.keys()):
                    self.actor_signature = self.actor_signature.append(
                        field,
                        self._get_field(self.output_fields[field]),
                    )
                    
            case False:
                self.actor_signature = self.actor_signature.append(
                    f"action_{idx+1}",
                    dspy.OutputField(
                        prefix=f"Action {idx}:",
                        desc=f"{idx}th action to taken",
                    )
                )

        self.actor = self._get_actor(self.actor_signature.fields)

    
    def _get_actor(self, fields: Mapping[str, FieldInfo]) -> dspy.Signature:
        return create_model(
            "Actor",
            __base__=dspy.Signature,
            __doc__=Actor.__doc__,
            **fields,
        )


    def _call_tool(self, tool_name: str, tool_input_query: str) -> str:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.tool.run(tool_input_query)


    def forward(self, **kwargs):
        max_iters = None if "max_iters" not in kwargs else kwargs["max_iters"]
        args = {key: kwargs[key] for key in self.input_fields.keys() if key in kwargs}
        idx = 1

        while tool_name != "Finish" and (max_iters > 0 if max_iters else True):
            action = self.actor(tools=self.tools, **args)

            tool_name = action.tool_name
            tool_input_query = action.tool_input_query

            if tool_name != "Finish":
                tool_output = self._call_tool(tool_name, tool_input_query)
                self._update_signature(idx)

                args[f"action_{idx}"] = action
                args[f"result_{idx}"] = tool_output
            else:
                self._update_signature(idx, omit_action=True)

                args[f"action_{idx}"] = action
                break

            idx += 1

            if max_iters:
                max_iters -= 1

        final_answer = self.actor(tools=self.tools, **args)
        return dspy.Prediction(
            **{key: getattr(final_answer, key) for key in self.output_fields.keys()}
        )