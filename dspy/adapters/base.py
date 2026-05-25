import json
import logging
from typing import Any, get_origin

from dspy.adapters._legacy_type_markers import (
    _expand_legacy_custom_type_markers_in_chat_message,
    _expand_legacy_custom_type_markers_in_lm_message,
)
from dspy.adapters._signature_field_partition import (
    _NativeResponseField,
    _partition_signature_fields,
    _SignatureFieldPartition,
)
from dspy.adapters.types import History, Type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import lm_response_from_legacy_outputs, to_openai_chat_request
from dspy.core.types import LMMessage, LMOutput, LMRequest, LMResponse
from dspy.experimental import Citations
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]


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
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.native_response_types = native_response_types or _DEFAULT_NATIVE_RESPONSE_TYPES

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def _render_request(
        self,
        partition: _SignatureFieldPartition,
        lm: BaseLM,
        demos: list[dict[str, Any]],
    ) -> LMRequest:
        """Render the remaining signature fields into a normalized LM request."""
        messages = self.format(partition.remaining_signature, demos, partition.remaining_inputs)
        request_kwargs = dict(partition.request_kwargs)
        tools = list(partition.tool_specs)
        if "tools" in request_kwargs:
            tools.extend(request_kwargs.pop("tools") or [])
        return LMRequest.from_call(
            model=lm.model,
            messages=self._coerce_lm_messages(messages),
            tools=tools,
            **request_kwargs,
        )

    def _call_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        """Call current `BaseLM` through the normalized request/response boundary.

        TODO(language-models): When `BaseLM` is replaced by/updated to the
        normalized `BaseLM.forward(request: LMRequest) -> LMResponse` contract,
        remove this compatibility shim and let adapters call the normalized LM
        entry point directly. The OpenAI-shaped compatibility kwargs should live
        only inside concrete LM backends.
        """
        data = self._legacy_call_kwargs(request)
        outputs = lm(messages=data.pop("messages"), **data)
        return self._normalize_legacy_outputs(outputs, request)

    async def _acall_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        """Async variant of `_call_lm`.

        TODO(language-models): Same transitional boundary as `_call_lm()`; this
        should eventually call a normalized async LM method directly.
        """
        data = self._legacy_call_kwargs(request)
        outputs = await lm.acall(messages=data.pop("messages"), **data)
        return self._normalize_legacy_outputs(outputs, request)

    def _legacy_call_kwargs(self, request: LMRequest) -> dict[str, Any]:
        # TODO(language-models): Current `BaseLM` expects OpenAI/LiteLLM-shaped
        # chat kwargs. We intentionally use `dspy.clients.openai_format` here so
        # the conversion code lives in the future LM/client layer, not in
        # adapters. Remove this adapter helper once `BaseLM` accepts `LMRequest`.
        data = to_openai_chat_request(request)
        data.pop("model", None)
        # TODO(language-models): `cache` and `rollout_id` are DSPy BaseLM
        # execution controls, not provider request fields. The future
        # normalized LM base should own them before provider-format conversion.
        if request.config.cache is not None:
            if request.config.cache.enabled is not None:
                data["cache"] = request.config.cache.enabled
            if request.config.cache.rollout_id is not None:
                data["rollout_id"] = request.config.cache.rollout_id
        return data

    def _coerce_lm_messages(self, messages: list[LMMessage | dict[str, Any]]) -> list[LMMessage]:
        """Normalize subclass `format()` output before the LM boundary.

        TODO(adapters-normalized-rendering): Adapter `format()` methods still
        return OpenAI-chat-shaped dictionaries. This coercion is the bridge until
        adapters render `LMMessage` / `LMPart` directly.
        """
        return [
            _expand_legacy_custom_type_markers_in_lm_message(
                message if isinstance(message, LMMessage) else self._chat_dict_to_lm_message(message)
            )
            for message in messages
        ]

    def _chat_dict_to_lm_message(self, message: dict[str, Any]) -> LMMessage:
        try:
            return LMMessage(**message)
        except Exception:
            # TODO(legacy-custom-types): Unknown OpenAI content blocks are
            # temporarily preserved as `legacy_content_block` metadata on an
            # empty text part so `openai_format` can round-trip them back to
            # current BaseLM calls. Replace this with either explicit opaque
            # provider parts or remove it when marker-based custom type
            # serialization is retired.
            message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                sanitized = []
                supported = {"text", "image_url", "input_audio", "file", "document", "video"}
                for block in content:
                    if isinstance(block, dict) and block.get("type") in supported:
                        sanitized.append(block)
                    elif isinstance(block, dict):
                        sanitized.append({"type": "text", "text": "", "metadata": {"legacy_content_block": block}})
                    else:
                        sanitized.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
                message["content"] = sanitized
            return LMMessage(**message)

    def _normalize_legacy_outputs(self, outputs: list[dict[str, Any] | str | None], request: LMRequest) -> LMResponse:
        """Convert current `BaseLM` outputs into a normalized `LMResponse`.

        TODO(language-models): Current `BaseLM` returns `list[str | dict | None]`.
        Future LMs should return `LMResponse` directly, making this method a
        compatibility-only path for old/custom LMs.
        """
        return lm_response_from_legacy_outputs(outputs, request)

    def _parse_response(self, partition: _SignatureFieldPartition, response: LMResponse, lm: BaseLM) -> list[dict[str, Any]]:
        """Parse a normalized LM response into dictionaries for the source signature."""
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(partition.source_signature)

        for output in response.outputs:
            if output.metadata.get("empty_legacy_outputs"):
                continue

            has_text_output = bool(output.text and partition.remaining_signature.output_fields)
            has_tool_output = bool(output.tool_calls and tool_call_output_field_name)

            if has_text_output:
                value = self.parse(partition.remaining_signature, output.text)
                for field_name in partition.source_signature.output_fields:
                    value.setdefault(field_name, None)
            else:
                value = dict.fromkeys(partition.source_signature.output_fields.keys())

            if has_tool_output:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [{"name": call.name, "args": call.args} for call in output.tool_calls]
                )

            has_native_output = self._parse_native_response_fields(value, partition, output, lm)

            if not (has_text_output or has_tool_output or has_native_output):
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=partition.source_signature,
                    lm_response=str(output.to_output_dict()),
                    message="The LM returned an empty or null response.",
                )

            if output.logprobs is not None:
                value["logprobs"] = output.logprobs

            values.append(value)

        return values

    def _parse_native_response_fields(
        self,
        value: dict[str, Any],
        partition: _SignatureFieldPartition,
        output: LMOutput,
        lm: BaseLM,
    ) -> bool:
        parsed_any = False
        for native_field in partition.native_response_fields:
            parsed_value = self._parse_native_response_field(native_field, output, partition, lm)
            if parsed_value is not None:
                value[native_field.name] = parsed_value
                parsed_any = True
        return parsed_any

    def _parse_native_response_field(
        self,
        native_field: _NativeResponseField,
        output: LMOutput,
        partition: _SignatureFieldPartition,
        lm: BaseLM,
    ) -> Type | None:
        # TODO(adapter-types): Replace this provider-shaped compatibility call
        # with `Type.parse_lm_output(context, output)` once the normalized type
        # hook lands. Prefer the exact legacy provider output when present so
        # existing custom types keep seeing the shape they saw before normalized
        # request/response parsing moved into adapters.
        legacy_response = output.provider_output if output.provider_output is not None else output.to_output_dict()
        return native_field.annotation.parse_lm_response(legacy_response)

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
        partition = _partition_signature_fields(self, lm, lm_kwargs, signature, inputs)
        request = self._render_request(partition, lm, demos)
        response = self._call_lm(lm, request)
        return self._parse_response(partition, response, lm)

    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        partition = _partition_signature_fields(self, lm, lm_kwargs, signature, inputs)
        request = self._render_request(partition, lm, demos)
        response = await self._acall_lm(lm, request)
        return self._parse_response(partition, response, lm)

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
            messages.append({"role": "user", "content": content})
        else:
            # Only current input
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)
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

    def _get_history_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    def _get_tool_call_input_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.input_fields.items():
            # Look for annotation `list[dspy.Tool]` or `dspy.Tool`
            origin = get_origin(field.annotation)
            if origin is list and field.annotation.__args__[0] == Tool:
                return name
            if field.annotation == Tool:
                return name
        return None

    def _get_tool_call_output_field_name(self, signature: type[Signature]) -> bool:
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
            A list of multiturn messages.
        """
        conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

        if conversation_history is None:
            return []

        messages = []
        for message in conversation_history:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, message),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(signature, message),
                }
            )

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
