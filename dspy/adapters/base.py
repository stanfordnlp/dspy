import json
import logging
from dataclasses import dataclass
from typing import Any, get_origin

import json_repair

from dspy.adapters._legacy_type_markers import (
    _expand_legacy_custom_type_markers_in_chat_message,
)
from dspy.adapters.types import History, Type
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import apply_output_field_defaults, serialize_for_json
from dspy.clients._deprecation import adapter_message_call
from dspy.clients.base_lm import BaseLM
from dspy.clients.capabilities import with_capability_planning
from dspy.dsp.utils.settings import settings
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]
_TOOL_CALL_RESULTS_SIGNATURE = Signature({"tool_call_results": (ToolCallResults, InputField())})


@dataclass(frozen=True)
class _AdapterPlan:
    """What one adapter call sends to the LM, and how to map the reply back to the declared signature.

    `Adapter._plan()` makes every decision that depends on the LM before rendering; `format()` renders
    `render_signature`, `demos`, and `inputs`; `Adapter._parse_outputs()` reads the reply with the same plan.
    """

    # The caller's signature. Results are keyed by its output fields.
    signature: type[Signature]
    # The signature the prompt describes and `parse()` reads. Fields served natively (tools, reasoning,
    # citations) are removed, and fallback reasoning fields are renamed.
    render_signature: type[Signature]
    demos: list[dict[str, Any]]
    inputs: dict[str, Any]
    lm_kwargs: dict[str, Any]
    # The `dspy.ToolCalls` output, filled from provider tool calls whenever the reply has them.
    tool_calls_field: str | None
    # Outputs a provider reply can fill through `Type.parse_lm_response()`.
    native_fields: tuple[str, ...]
    # Declared output name -> name used in the prompt, for fallback reasoning fields.
    wire_names: dict[str, str]


def _drop_tool_kwargs(lm_kwargs: dict[str, Any]) -> None:
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        lm_kwargs.pop(key, None)


class Adapter:
    """Base Adapter class.

    The Adapter serves as the interface layer between DSPy module/signature and Language Models (LMs). It handles the
    complete transformation pipeline from DSPy inputs to LM calls and back to structured outputs.

    Key responsibilities:
        - Transform user inputs and signatures into properly formatted LM prompts, which also instructs the LM to format
            the response in a specific format.
        - Parse LM outputs into dictionaries matching the signature's output fields.
        - Enable/disable native LM features (function calling, citations, etc.) based on configuration.
        - Handle conversation history, few-shot examples, and custom type processing.

    The adapter pattern allows DSPy to work with different LM interfaces while maintaining a consistent programming
    model for users.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        parallel_tool_calls: bool | None = None,
    ):
        """
        Args:
            callbacks: List of callback functions to execute during `format()` and `parse()` methods. Callbacks can be
                used for logging, monitoring, or custom processing. Defaults to None (empty list).
            use_native_function_calling: Whether to enable native function calling capabilities when the LM supports it.
                If True, the adapter will automatically configure function calling when input fields contain `dspy.Tool`
                or `list[dspy.Tool]` types. Defaults to False.
            native_response_types: List of output field types that should be handled by native LM features rather than
                adapter parsing. For example, `dspy.Citations` can be populated directly by citation APIs
                (e.g., Anthropic's citation feature). Defaults to `[Citations]`.
            parallel_tool_calls: Whether to request provider-side parallel tool-call generation when native function
                calling is active. If None, the adapter does not set the provider option. Defaults to None.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.parallel_tool_calls = parallel_tool_calls
        self.native_response_types = native_response_types or _DEFAULT_NATIVE_RESPONSE_TYPES

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def _plan(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> _AdapterPlan:
        """Decide what this call sends to the LM and how the reply maps back to `signature`.

        The caller's `lm_kwargs`, `demos`, and `inputs` are not modified.
        """
        lm_kwargs = dict(lm_kwargs)
        render_signature = signature
        tools_field = self._get_tool_call_input_field_name(signature)
        tool_calls_field = self._get_tool_call_output_field_name(signature)

        # Native tool calling sends tools through the provider API instead of the prompt.
        if not self.use_native_function_calling:
            _drop_tool_kwargs(lm_kwargs)
        elif tool_calls_field and tools_field is None:
            raise ValueError(
                f"You provided an output field {tool_calls_field} to receive the tool calls information, "
                "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                "input field with type `list[dspy.Tool]`."
            )
        elif tool_calls_field and lm.supports_function_calling:
            tools = inputs[tools_field]
            tools = tools if isinstance(tools, list) else [tools]
            lm_kwargs["tools"] = [tool.format_as_litellm_function_call() for tool in tools]
            if self.parallel_tool_calls is not None and lm_kwargs.get("parallel_tool_calls") is None:
                lm_kwargs["parallel_tool_calls"] = self.parallel_tool_calls
            render_signature = render_signature.delete(tool_calls_field).delete(tools_field)
        elif tool_calls_field:
            _drop_tool_kwargs(lm_kwargs)
        if lm_kwargs.get("parallel_tool_calls") is not None:
            lm_kwargs.setdefault("tool_choice", "auto")

        # TODO(adapter-types): Built-in native response planning (reasoning, citations) should move out of
        # `Type.adapt_to_native_lm_feature()`, which edits the signature and `lm_kwargs` it is given, into this plan.
        for name, field in render_signature.output_fields.items():
            if self._is_native_response_type(field.annotation):
                render_signature = field.annotation.adapt_to_native_lm_feature(render_signature, name, lm, lm_kwargs)

        # A default CoT/RLM reasoning field that is not served natively is rendered under a fallback name.
        wire_names = {}
        for name in render_signature.output_fields:
            wire_name = Reasoning._fallback_field_name(signature, name)
            if wire_name != name:
                wire_names[name] = wire_name
        if wire_names:
            render_signature = Signature(
                {wire_names.get(key, key): field for key, field in render_signature.fields.items()},
                render_signature.instructions,
            )

            def rename(values):
                return {wire_names.get(key, key): value for key, value in values.items()}

            demos = [rename(demo) for demo in demos]
            inputs = dict(inputs)
            for name, field in render_signature.input_fields.items():
                if field.annotation is History and name in inputs:
                    history = inputs[name]
                    inputs[name] = history.model_copy(update={"messages": [rename(m) for m in history.messages]})

        return _AdapterPlan(
            signature=signature,
            render_signature=render_signature,
            demos=demos,
            inputs=inputs,
            lm_kwargs=lm_kwargs,
            tool_calls_field=tool_calls_field,
            native_fields=tuple(
                name
                for name, field in signature.output_fields.items()
                if self._is_native_response_type(field.annotation)
                # Reasoning that the prompt asks for is read from the text, not from the provider.
                and not (field.annotation is Reasoning and wire_names.get(name, name) in render_signature.output_fields)
            ),
            wire_names=wire_names,
        )

    def _is_native_response_type(self, annotation: Any) -> bool:
        return isinstance(annotation, type) and annotation in self.native_response_types and issubclass(annotation, Type)

    def _parse_outputs(self, plan: _AdapterPlan, outputs: list[dict[str, Any] | str]) -> list[dict[str, Any]]:
        """Parse each LM output with `plan.render_signature` and key the result by `plan.signature`."""
        values = []
        for output in outputs:
            text, tool_calls, logprobs = output, None, None
            if isinstance(output, dict):
                text, tool_calls, logprobs = output.get("text"), output.get("tool_calls"), output.get("logprobs")
            use_tool_calls = bool(tool_calls and plan.tool_calls_field)

            if text and not use_tool_calls:
                value = self.parse(plan.render_signature, text)
            elif use_tool_calls:
                # Provider tool calls are authoritative; the other outputs are best effort.
                try:
                    value = self.parse(plan.render_signature, text) if text and plan.render_signature.output_fields else {}
                except AdapterParseError:
                    value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=plan.signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            for name, wire_name in plan.wire_names.items():
                if wire_name in value:
                    value[name] = value.pop(wire_name)

            # Fields removed from the prompt for native features are absent from the parse.
            value = apply_output_field_defaults(plan.signature, value)
            for name in plan.signature.output_fields:
                value.setdefault(name, None)

            if use_tool_calls:
                tool_calls = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
                value[plan.tool_calls_field] = ToolCalls.from_dict_list(tool_calls)

            # TODO(adapter-types): Once `Type.parse_lm_output(context, output)` is the real normalized hook, this
            # should not call the legacy provider-shaped `parse_lm_response()` directly.
            for name in plan.native_fields:
                parsed_value = plan.signature.output_fields[name].annotation.parse_lm_response(output)
                if parsed_value is not None:
                    value[name] = parsed_value

            if logprobs:
                value["logprobs"] = logprobs

            values.append(value)

        return values

    @with_capability_planning
    def __call__(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute the adapter pipeline: format inputs, call LM, and parse outputs.

        Args:
            lm: The Language Model instance to use for generation. Must be an instance of `dspy.BaseLM`.
            lm_kwargs: Additional keyword arguments to pass to the LM call (e.g., temperature, max_tokens). These are
                passed directly to the LM.
            signature: The DSPy signature associated with this LM call.
            demos: List of few-shot examples to include in the prompt. Each dictionary should contain keys matching the
                signature's input and output field names. Examples are formatted as user/assistant message pairs.
            inputs: The current input values for this call. Keys must match the signature's input field names.

        Returns:
            List of dictionaries representing parsed LM responses. Each dictionary contains keys matching the
            signature's output field names. For multiple generations (n > 1), returns multiple dictionaries.
        """
        plan = self._plan(lm, lm_kwargs, signature, demos, inputs)
        messages = self.format(plan.render_signature, plan.demos, plan.inputs)
        if settings.send_stream and settings.stream_listeners:
            from dspy.streaming.messages import _StreamSignature, sync_send_to_stream

            stream_signature = _StreamSignature(id(settings.caller_predict), plan.render_signature, plan.wire_names)
            sync_send_to_stream(settings.send_stream, stream_signature)
        # TODO(3.5): build Request and parse Response directly; remove this marker.
        with adapter_message_call(lm, messages):
            outputs = lm(messages=messages, **plan.lm_kwargs)
        return self._parse_outputs(plan, outputs)

    @with_capability_planning
    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        plan = self._plan(lm, lm_kwargs, signature, demos, inputs)
        messages = self.format(plan.render_signature, plan.demos, plan.inputs)
        if settings.send_stream and settings.stream_listeners:
            from dspy.streaming.messages import _StreamSignature

            stream_signature = _StreamSignature(id(settings.caller_predict), plan.render_signature, plan.wire_names)
            await settings.send_stream.send(stream_signature)
        # TODO(3.5): share the canonical request/response boundary with __call__.
        with adapter_message_call(lm, messages):
            outputs = await lm.acall(messages=messages, **plan.lm_kwargs)
        return self._parse_outputs(plan, outputs)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the input messages for the LM call.

        This method converts the DSPy structured input along with few-shot examples and conversation history into
        multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
        the formatting of the input messages.

        In general we recommend the messages to have the following structure:
        ```
        [
            {"role": "system", "content": system_message},
            # Begin few-shot examples
            {"role": "user", "content": few_shot_example_1_input},
            {"role": "assistant", "content": few_shot_example_1_output},
            {"role": "user", "content": few_shot_example_2_input},
            {"role": "assistant", "content": few_shot_example_2_output},
            ...
            # End few-shot examples
            # Begin conversation history
            {"role": "user", "content": conversation_history_1_input},
            {"role": "assistant", "content": conversation_history_1_output},
            {"role": "user", "content": conversation_history_2_input},
            {"role": "assistant", "content": conversation_history_2_output},
            ...
            # End conversation history
            {"role": "user", "content": current_input},
        ]

        And system message should contain the field description, field structure, and task description.
        ```


        Args:
            signature: The DSPy signature for which to format the input messages.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        inputs_copy = dict(inputs)

        # If the signature and inputs have conversation history, we need to format the conversation history and
        # remove the history field from the signature.
        history_field_name = self._get_history_field_name(signature)
        if history_field_name:
            # In order to format the conversation history, we need to remove the history field from the signature.
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                inputs_copy,
            )

        messages = []
        system_message = self.format_system_message(signature)
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))
        if history_field_name:
            # Conversation history and current input
            content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
            messages.extend(conversation_history)
            if content:
                messages.append({"role": "user", "content": content})
        else:
            # Only current input
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)
            if content:
                messages.append({"role": "user", "content": content})

        return [_expand_legacy_custom_type_markers_in_chat_message(message) for message in messages]

    def format_system_message(self, signature: type[Signature]) -> str:
        """Format the system message for the LM call.


        Args:
            signature: The DSPy signature for which to format the system message.
        """
        return (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format the field description for the system message.

        This method formats the field description for the system message. It should return a string that contains
        the field description for the input fields and the output fields.

        Args:
            signature: The DSPy signature for which to format the field description.

        Returns:
            A string that contains the field description for the input fields and the output fields.
        """
        raise NotImplementedError

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the field structure for the system message.

        This method formats the field structure for the system message. It should return a string that dictates the
        format the input fields should be provided to the LM, and the format the output fields will be in the response.
        Refer to the ChatAdapter and JsonAdapter for an example.

        Args:
            signature: The DSPy signature for which to format the field structure.
        """
        raise NotImplementedError

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description for the system message.

        This method formats the task description for the system message. In most cases this is just a thin wrapper
        over `signature.instructions`.

        Args:
            signature: The DSPy signature of the DSpy module.

        Returns:
            A string that describes the task.
        """
        raise NotImplementedError

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message content.

        This method formats the user message content, which can be used in formatting few-shot examples, conversation
        history, and the current input.

        Args:
            signature: The DSPy signature for which to format the user message content.
            inputs: The input arguments to the DSPy module.
            prefix: A prefix to the user message content.
            suffix: A suffix to the user message content.

        Returns:
            A string that contains the user message content.
        """
        raise NotImplementedError

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format the assistant message content.

        This method formats the assistant message content, which can be used in formatting few-shot examples,
        conversation history.

        Args:
            signature: The DSPy signature for which to format the assistant message content.
            outputs: The output fields to be formatted.
            missing_field_message: A message to be used when a field is missing.

        Returns:
            A string that contains the assistant message content.
        """
        raise NotImplementedError

    def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format the few-shot examples.

        This method formats the few-shot examples as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the few-shot examples.
            demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
                the signature.

        Returns:
            A list of multiturn messages.
        """
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            # Check if all fields are present and not None
            is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

            # Check if demo has at least one input and one output field
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                # We only keep incomplete demos that have at least one input and one output field
                incomplete_demos.append(demo)

        messages = []

        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this particular example. "
                    ),
                }
            )

        for demo in complete_demos:
            messages.append({"role": "user", "content": self.format_user_message_content(signature, demo)})
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this conversation history message. "
                    ),
                }
            )

        return messages

    def _get_history_field_name(self, signature: type[Signature]) -> str | None:
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    def _get_tool_call_input_field_name(self, signature: type[Signature]) -> str | None:
        for name, field in signature.input_fields.items():
            # Look for annotation `list[dspy.Tool]` or `dspy.Tool`
            origin = get_origin(field.annotation)
            if origin is list and field.annotation.__args__[0] == Tool:
                return name
            if field.annotation == Tool:
                return name
        return None

    def _get_tool_call_output_field_name(self, signature: type[Signature]) -> str | None:
        for name, field in signature.output_fields.items():
            if field.annotation == ToolCalls:
                return name
        return None

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the conversation history.

        This method formats the conversation history and the current input as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the conversation history.
            history_field_name: The name of the history field in the signature.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

        if conversation_history is None:
            return []

        messages = []
        for message in conversation_history:
            tool_call_field_name, tool_calls = _tool_calls_from_message(message)
            tool_call_results = (
                ToolCallResults.model_validate(tool_calls.tool_call_results)
                if tool_calls is not None and tool_calls.tool_call_results is not None
                else None
            )

            # Preserve the format reminder from the original request for prompt-cache reuse.
            # Output-only events must not acquire a synthetic user message containing just the reminder.
            user_content = self.format_user_message_content(
                signature, message, main_request=any(name in message for name in signature.input_fields)
            )
            if user_content:
                messages.append({"role": "user", "content": user_content})

            if self.use_native_function_calling and tool_calls is not None:
                content_signature = signature
                for name, field in signature.output_fields.items():
                    if field.annotation == ToolCalls or message.get(name) is None:
                        content_signature = content_signature.delete(name)

                content = (
                    self.format_assistant_message_content(content_signature, message)
                    if content_signature.output_fields
                    else ""
                )

                if tool_call_results is not None:
                    tool_call_ids = [tool_call.id for tool_call in tool_calls.tool_calls]
                    result_ids = [result.call_id for result in tool_call_results.tool_call_results]
                    if tool_call_ids != result_ids or not all(tool_call_ids):
                        tool_call_results = None

                if content or tool_call_results is not None:
                    assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
                    if tool_call_results is not None:
                        assistant_message["tool_calls"] = [
                            _tool_call_as_openai_message_tool_call(tool_call) for tool_call in tool_calls.tool_calls
                        ]
                    messages.append(assistant_message)

                if tool_call_results is not None:
                    for result in tool_call_results.tool_call_results:
                        content = _tool_result_content(result.value)
                        messages.append(
                            {"role": "tool", "tool_call_id": result.call_id, "name": result.name, "content": content}
                        )
                continue

            assistant_values = message
            if tool_call_field_name is not None and tool_call_results is not None:
                assistant_values = dict(message)
                assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})

            assistant_content = self.format_assistant_message_content(signature, assistant_values)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            if tool_call_results is not None:
                result_input = {"tool_call_results": tool_call_results}
                content = self.format_user_message_content(_TOOL_CALL_RESULTS_SIGNATURE, result_input)
                messages.append({"role": "user", "content": content})

        # Remove the history field from the inputs
        del inputs[history_field_name]

        return messages

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            signature: The DSPy signature for which to parse the LM output.
            completion: The LM output to be parsed.

        Returns:
            A dictionary of the output fields.
        """
        raise NotImplementedError


def _provider_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _provider_tool_call_to_tool_call_dict(tool_call: Any) -> dict[str, Any]:
    function = _provider_value(tool_call, "function", {}) or {}
    # Responses API tool calls carry `arguments` at the top level rather than
    # under `function`.
    arguments = _provider_value(function, "arguments", None) or _provider_value(tool_call, "arguments", {})
    if isinstance(arguments, str):
        parsed_arguments = json_repair.loads(arguments)
    elif isinstance(arguments, dict):
        parsed_arguments = arguments
    else:
        parsed_arguments = {}

    return {
        # Responses API items carry both an item `id` ("fc_...") and the
        # `call_id` that tool results must reference; prefer `call_id`.
        "id": _provider_value(tool_call, "call_id") or _provider_value(tool_call, "id"),
        "name": _provider_value(function, "name") or _provider_value(tool_call, "name"),
        "args": parsed_arguments,
    }


def _tool_calls_from_message(message: dict[str, Any]) -> tuple[str | None, ToolCalls | None]:
    for name, value in message.items():
        if isinstance(value, ToolCalls) or (isinstance(value, dict) and "tool_calls" in value):
            return name, ToolCalls.model_validate(value)
    return None, None


def _tool_result_content(value: Any) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(serialize_for_json(value), ensure_ascii=False)


def _tool_call_as_openai_message_tool_call(tool_call: ToolCalls.ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(serialize_for_json(tool_call.args), ensure_ascii=False),
        },
    }
