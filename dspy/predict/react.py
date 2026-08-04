import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

import pydantic

import dspy
from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.types.tool import Tool, _resolve_json_schema_reference
from dspy.adapters.utils import get_annotation_name, parse_value
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.callback import with_callbacks
from dspy.utils.exceptions import ContextWindowExceededError, format_error_for_lm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class ReAct(Module):
    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 20):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism.

        The auto-registered `finish` tool takes the signature's output fields as its arguments, so the model
        delivers the final outputs with the same call that ends the loop. Those arguments are coerced to the
        fields' declared annotations and returned directly, with the final thought as `reasoning`, saving an
        LM call. If an output field is missing from the arguments or fails to validate, or the loop ends
        without a `finish` call, a `dspy.ChainOfThought` extractor reads the trajectory and produces the
        outputs instead.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

        Examples:

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
                f"When you have all information needed, call the `finish` tool with the final value for each of {outputs} passed as next_tool_args.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        self._output_type_adapters = {
            name: _type_adapter_for_annotation(field.annotation) for name, field in signature.output_fields.items()
        }

        tools["finish"] = Tool(
            func=lambda **kwargs: "Completed.",
            name="finish",
            desc=(
                f"Marks the task as complete and provides the final outputs. Call this with the final value for "
                f"each of {outputs} passed as arguments, once all information for producing them is available."
            ),
            args={
                name: _finish_arg_schema(name, field, self._output_type_adapters[name])
                for name, field in signature.output_fields.items()
            },
            arg_types={name: field.annotation for name, field in signature.output_fields.items()},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
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

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ContextWindowExceededError as err:
                logger.warning(f"Ending the trajectory: {format_error_for_lm(err, traceback_frames=5)}")
                break
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {format_error_for_lm(err, traceback_frames=5)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            if pred.next_tool_name == "finish":
                finish_prediction = self._finish_prediction(trajectory, idx, pred)
                if finish_prediction is not None:
                    return finish_prediction
                break

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {format_error_for_lm(err, traceback_frames=5)}"

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ContextWindowExceededError as err:
                logger.warning(f"Ending the trajectory: {format_error_for_lm(err, traceback_frames=5)}")
                break
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {format_error_for_lm(err, traceback_frames=5)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            if pred.next_tool_name == "finish":
                finish_prediction = self._finish_prediction(trajectory, idx, pred)
                if finish_prediction is not None:
                    return finish_prediction
                break

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {format_error_for_lm(err, traceback_frames=5)}"

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _finish_prediction(self, trajectory: dict[str, Any], idx: int, pred) -> "dspy.Prediction | None":
        """Record the `finish` call in the trajectory and build the final prediction from its args.

        Returns None when the args do not yield every output field, in which case the caller breaks out
        of the loop and falls back to the extract step.
        """
        finish_args = pred.next_tool_args if isinstance(pred.next_tool_args, dict) else {}
        trajectory[f"observation_{idx}"] = _run_finish_tool(self.tools["finish"], **finish_args)
        parsed_outputs = self._extract_outputs_from_finish_args(pred.next_tool_args)
        if parsed_outputs is None:
            return None
        return dspy.Prediction(trajectory=trajectory, **{"reasoning": pred.next_thought, **parsed_outputs})

    def _extract_outputs_from_finish_args(self, next_tool_args: dict[str, Any]) -> dict[str, Any] | None:
        """Parse the signature's output fields from the args of a `finish` tool call.

        Values are coerced with the same `parse_value` the extract step's adapter uses, so anything
        the fallback would have accepted for a field is accepted here too (e.g. a bare `3` for a
        `str` output, an enum member's name, or a JSON-encoded string of the annotated type).

        Returns a mapping from output field names to values validated against the fields' declared
        annotations, or None when any output field is absent from the args or fails validation — in
        which case the caller falls back to the legacy extract step. An explicit None is a valid
        value for a nullable annotation (e.g. `str | None`), not a missing field; it is validated
        against the annotation itself so that a non-nullable field falls back rather than being
        coerced into the string "None".
        """
        if not isinstance(next_tool_args, dict) or not next_tool_args:
            return None

        outputs = {}
        for name, field in self.signature.output_fields.items():
            if name not in next_tool_args:
                return None
            value = next_tool_args[name]
            type_adapter = self._output_type_adapters.get(name)
            if type_adapter is None:
                return None
            try:
                if value is None:
                    outputs[name] = type_adapter.validate_python(None)
                else:
                    outputs[name] = parse_value(value, field.annotation)
            except Exception as err:
                logger.debug(f"Falling back to extract: `finish` arg `{name}` failed validation: {err}")
                return None
        return outputs

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        last_error = None
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError as err:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                last_error = err
                trajectory = self.truncate_trajectory(trajectory)
        raise ContextWindowExceededError(
            message="The context window was exceeded even after 3 attempts to truncate the trajectory."
        ) from last_error

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        last_error = None
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError as err:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                last_error = err
                trajectory = self.truncate_trajectory(trajectory)
        raise ContextWindowExceededError(
            message="The context window was exceeded even after 3 attempts to truncate the trajectory."
        ) from last_error

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if len(keys) <= 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ContextWindowExceededError(
                message="The trajectory is too long so your prompt exceeded the context window, but the trajectory "
                "cannot be truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory


@with_callbacks
def _run_finish_tool(instance: Tool, **kwargs) -> str:
    """Execute the `finish` tool without `Tool.__call__`'s strict argument validation.

    The fast path accepts laxly-coercible args that the tool's jsonschema validation would
    reject, so the function is invoked directly; routing it through `with_callbacks` (with the
    tool as `instance`) preserves the `on_tool_start`/`on_tool_end` lifecycle that tracing and
    accounting integrations rely on.
    """
    return instance.func(**kwargs)


def _type_adapter_for_annotation(annotation: Any) -> "pydantic.TypeAdapter | None":
    try:
        return pydantic.TypeAdapter(annotation)
    except Exception as err:
        logger.debug(f"Could not build a pydantic TypeAdapter for the annotation `{annotation}`: {err}")
        return None


def _json_schema_for_type_adapter(type_adapter: "pydantic.TypeAdapter | None", annotation: Any) -> dict[str, Any]:
    if type_adapter is not None:
        try:
            return _resolve_json_schema_reference(type_adapter.json_schema())
        except Exception as err:
            logger.debug(
                f"Could not build a JSON schema for the annotation `{annotation}`, advertising it to the LM as a "
                f"string instead: {err}"
            )
    return {"type": "string"}


def _finish_arg_schema(name: str, field: Any, type_adapter: "pydantic.TypeAdapter | None") -> dict[str, Any]:
    """Build the `finish` tool's JSON schema for one output field.

    The schema carries the field's `desc`, custom type descriptions, and constraints so that the LM
    writing the final answer sees the same guidance the extract step's signature renders for it.
    """
    schema = dict(_json_schema_for_type_adapter(type_adapter, field.annotation))
    description = _output_field_description(name, field, schema.get("description"))
    if description:
        schema["description"] = description
    return schema


def _output_field_description(name: str, field: Any, inherited_description: str | None = None) -> str:
    parts = []
    extra = field.json_schema_extra or {}

    desc = extra.get("desc")
    if desc and desc != f"${{{name}}}":
        parts.append(desc)

    for custom_type in DspyType.extract_custom_type_from_annotation(field.annotation):
        type_description = custom_type.description()
        if type_description:
            parts.append(f"Type description of {get_annotation_name(custom_type)}: {type_description}")

    constraints = extra.get("constraints")
    if constraints:
        parts.append(f"Constraints: {constraints}")

    if inherited_description and inherited_description not in parts:
        parts.append(inherited_description)

    return "; ".join(parts)


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
