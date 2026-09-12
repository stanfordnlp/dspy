import json
import logging
from typing import Any, get_origin

import json_repair

from dspy.adapters._legacy_type_markers import (
    _expand_legacy_custom_type_markers_in_chat_message,
)
from dspy.adapters.types import History, Type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import apply_output_field_defaults, serialize_for_json
from dspy.clients._deprecation import adapter_message_call
from dspy.clients.base_lm import BaseLM
from dspy.clients.capabilities import with_capability_planning
from dspy.experimental import Citations
from dspy.primitives.repl_types import REPLEntry, split_repl_event
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]
_TOOL_CALL_RESULTS_SIGNATURE = Signature({"tool_call_results": (ToolCallResults, InputField())})
_REPL_OUTPUT_SIGNATURE = Signature({"repl_output": (str, InputField())})


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

    def _call_preprocess(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> type[Signature]:
        # TODO(adapters-plan): This remains the pre-normalized planning hook. It
        # mutates `lm_kwargs` and returns only the render signature, which loses
        # information we will need for plan-driven rendering/parsing. The next
        # stacked PR should replace this with an explicit `_AdapterPlan` that
        # records deleted fields, native tools, native output fields, inserted
        # messages/parts, and LM config patches.
        if not self.use_native_function_calling:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                lm_kwargs.pop(key, None)
        else:
            tool_call_input_field_name = self._get_tool_call_input_field_name(signature)
            tool_call_output_field_name = self._get_tool_call_output_field_name(signature)

            if tool_call_output_field_name and tool_call_input_field_name is None:
                raise ValueError(
                    f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                    "input field with type `list[dspy.Tool]`."
                )

            if tool_call_output_field_name and lm.supports_function_calling:
                tools = inputs[tool_call_input_field_name]
                tools = tools if isinstance(tools, list) else [tools]

                lm_tools = [tool.format_as_litellm_function_call() for tool in tools]

                lm_kwargs["tools"] = lm_tools
                if self.parallel_tool_calls is not None and lm_kwargs.get("parallel_tool_calls") is None:
                    lm_kwargs["parallel_tool_calls"] = self.parallel_tool_calls

                signature = signature.delete(tool_call_output_field_name)
                signature = signature.delete(tool_call_input_field_name)

        # TODO(adapters-plan): Built-in/native response planning should move out
        # of `Type.adapt_to_native_lm_feature()` and into adapter-owned planning
        # renderers. Keep this compatibility hook for this boundary-only PR.
        # Handle custom types that use native LM features, e.g., reasoning, citations, etc.
        for name, field in signature.output_fields.items():
            if (
                isinstance(field.annotation, type)
                and field.annotation in self.native_response_types
                and issubclass(field.annotation, Type)
            ):
                signature = field.annotation.adapt_to_native_lm_feature(signature, name, lm, lm_kwargs)

        return signature

    def _call_postprocess(
        self,
        processed_signature: type[Signature],
        original_signature: type[Signature],
        outputs: list[dict[str, Any] | str],
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for output in outputs:
            output_logprobs = None
            tool_calls = None
            text = output

            if isinstance(output, dict):
                text = output.get("text")
                output_logprobs = output.get("logprobs")
                tool_calls = output.get("tool_calls")

            if text and not (tool_calls and tool_call_output_field_name):
                value = self.parse(processed_signature, text)
            elif tool_calls and tool_call_output_field_name:
                try:
                    value = self.parse(processed_signature, text) if text and processed_signature.output_fields else {}
                except AdapterParseError:
                    value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            # Fields removed for native features are absent from the processed parse
            value = apply_output_field_defaults(original_signature, value)
            for field_name in original_signature.output_fields:
                value.setdefault(field_name, None)

            if tool_calls and tool_call_output_field_name:
                tool_calls = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            # TODO(adapter-types): Once `Type.parse_lm_output(context, output)` is
            # the real normalized hook, this should not call the legacy
            # provider-shaped `parse_lm_response()` directly.
            # Parse custom types that does not rely on the `Adapter.parse()` method
            for name, field in original_signature.output_fields.items():
                if (
                    isinstance(field.annotation, type)
                    and field.annotation in self.native_response_types
                    and issubclass(field.annotation, Type)
                ):
                    parsed_value = field.annotation.parse_lm_response(output)
                    if parsed_value is not None:
                        value[name] = parsed_value

            if output_logprobs:
                value["logprobs"] = output_logprobs

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
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        messages = self.format(processed_signature, demos, inputs)
        if lm_kwargs.get("parallel_tool_calls") is not None:
            lm_kwargs.setdefault("tool_choice", "auto")
        # TODO(3.5): build Request and parse Response directly; remove this marker.
        with adapter_message_call(lm, messages):
            outputs = lm(messages=messages, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs, lm, lm_kwargs)

    @with_capability_planning
    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        messages = self.format(processed_signature, demos, inputs)
        if lm_kwargs.get("parallel_tool_calls") is not None:
            lm_kwargs.setdefault("tool_choice", "auto")
        # TODO(3.5): share the canonical request/response boundary with __call__.
        with adapter_message_call(lm, messages):
            outputs = await lm.acall(messages=messages, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs, lm, lm_kwargs)

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

        signature_without_history = None
        conversation_history: list[dict[str, Any]] = []

        # If the signature and inputs have conversation history, we need to format the conversation history and
        # remove the history field from the signature.
        history_field_name = self._get_history_field_name(signature)
        if history_field_name:
            # In order to format the conversation history, we need to remove the history field from the signature.
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history, history_field_name, inputs_copy
            )

        messages = []

        system_message = self.format_system_message(signature)
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))
        if history_field_name and signature_without_history:
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

            user_content = self.format_user_message_content(signature, message)
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

            # RLM history events (see `build_repl_event`) carry the RLM's own input/output fields (e.g. `query`,
            # `answer`) around a REPLEntry. Those fields are not part of the inner action signature, so they are
            # rendered here as plain context rather than through the signature-driven path.
            repl_inputs, repl_entry, repl_outputs = split_repl_event(message)
            if repl_entry is not None:
                known = set(signature.input_fields) | set(signature.output_fields)
                extra_inputs = {k: v for k, v in repl_inputs.items() if k not in known}
                extra_outputs = {k: v for k, v in repl_outputs.items() if k not in known}
                if extra_inputs:
                    messages.append({"role": "user", "content": self._format_extra_fields(extra_inputs)})

                # Format assistant message with code and reasoning (if present)
                # Fish code and (potentially) reasoning out of REPLEntry
                assistant_values: dict[str, Any] = {}
                assistant_values["code"] = repl_entry.code

                # If the LM supports native reasoning, then we don't want to put the reasoning in the assistant message content.
                # We don't have access to the LM here, so we check if the signature's outputs has a dspy.Reasoning field as a proxy.
                # All dspy.Reasoning fields are removed from the signature when the LM supports native reasoning,
                # so if there exists a dspy.Reasoning output field, we know we have a non-reasoning LM
                # and therefore include the repl entry's reasoning content in the assistant message content
                if any(
                    isinstance(field.annotation, type) and issubclass(field.annotation, Reasoning)
                    for field in signature.output_fields.values()
                ):
                    assistant_values["reasoning"] = repl_entry.reasoning
                assistant_content = self.format_assistant_message_content(signature, assistant_values)
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})

                # Format user message with repl output, followed by any RLM output fields recorded on this event
                repl_output_values = {
                    "repl_output": REPLEntry.format_output(repl_entry.output, repl_entry.max_output_chars),
                    **extra_outputs,
                }
                content = self._format_extra_fields(repl_output_values)
                messages.append({"role": "user", "content": content})
                continue

            # An event with none of the signature's fields cannot be rendered as a normal turn. RLM's extract
            # fallback produces such an event: only the final output fields, which the model produced via the
            # extract `Predict` call, so replay them as an assistant turn.
            if not any(name in message for name in (*signature.input_fields, *signature.output_fields)):
                extra_fields = {name: value for name, value in repl_outputs.items() if name != tool_call_field_name}
                if extra_fields:
                    messages.append({"role": "assistant", "content": self._format_extra_output_fields(extra_fields)})
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

    def _format_extra_fields(self, values: dict[str, Any]) -> str:
        """Format fields that are not declared on the signature as if they were `str` input fields."""
        extra_signature = Signature({name: (str, InputField()) for name in values})
        return self.format_user_message_content(extra_signature, values)

    def _format_extra_output_fields(self, values: dict[str, Any]) -> str:
        """Format fields that are not declared on the signature as if they were `str` output fields."""
        extra_signature = Signature({name: (str, OutputField()) for name in values})
        return self.format_assistant_message_content(extra_signature, values)

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
