import json
import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, get_args, get_origin

import json_repair
from pydantic.fields import FieldInfo

from dspy.adapters._legacy_type_markers import (
    _expand_legacy_custom_type_markers_in_lm_message,
    _split_legacy_custom_type_text_to_parts,
)
from dspy.adapters.types import History, Type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import format_field_value
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import (
    lm_response_from_legacy_outputs,
    message_to_openai_chat,
    to_openai_chat_request,
)
from dspy.core.types import LMMessage, LMPart, LMRequest, LMResponse, LMTextPart, LMToolSpec
from dspy.experimental import Citations
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]


@dataclass
class _AdapterRequestState:
    source_signature: type[Signature]
    render_signature: type[Signature]
    inputs: dict[str, Any]
    lm_kwargs: dict[str, Any]
    tools: list[LMToolSpec] = dataclass_field(default_factory=list)
    prepared_messages: list[LMMessage] = dataclass_field(default_factory=list)
    hidden_output_fields: tuple[str, ...] = ()


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
        if self.use_native_function_calling:
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

                signature_for_native_function_calling = signature.delete(tool_call_output_field_name)
                signature_for_native_function_calling = signature_for_native_function_calling.delete(
                    tool_call_input_field_name
                )

                return signature_for_native_function_calling

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
        # TODO(adapters-plan): This still parses legacy adapter output objects.
        # PR1 normalizes the LM boundary, then immediately converts back to this
        # shape to avoid changing parser semantics. A later PR should parse
        # `LMResponse` directly and merge text-parsed fields with explicit
        # native fields from `_AdapterPlan`.
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for output in outputs:
            output_logprobs = None
            tool_calls = None
            text = output

            if isinstance(output, dict):
                text = output["text"]
                output_logprobs = output.get("logprobs")
                tool_calls = output.get("tool_calls")

            if text:
                value = self.parse(processed_signature, text)
                for field_name in original_signature.output_fields.keys():
                    if field_name not in value:
                        # We need to set the field not present in the processed signature to None for consistency.
                        value[field_name] = None
            elif tool_calls and tool_call_output_field_name:
                value = {}
                for field_name in original_signature.output_fields.keys():
                    value[field_name] = None
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            if tool_calls and tool_call_output_field_name:
                tool_calls = [
                    {
                        "name": v["function"]["name"],
                        "args": json_repair.loads(v["function"]["arguments"]),
                        **({"id": v["id"]} if v.get("id") is not None else {}),
                    }
                    for v in tool_calls
                ]
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

    def _render_request(self, lm: BaseLM, state: _AdapterRequestState, demos: list[dict[str, Any]]) -> LMRequest:
        messages = self._format_request_with_callbacks(state, demos)
        request_kwargs = self._prepare_request_kwargs(lm, state)
        return LMRequest.from_call(
            model=lm.model,
            messages=self._coerce_lm_messages(messages),
            tools=state.tools,
            **request_kwargs,
        )

    def _prepare_request_kwargs(self, lm: BaseLM, state: _AdapterRequestState) -> dict[str, Any]:
        return dict(state.lm_kwargs)

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

    def _normalize_legacy_outputs(
        self, outputs: list[dict[str, Any] | str | None] | LMResponse, request: LMRequest
    ) -> LMResponse:
        """Convert current `BaseLM` outputs into a normalized `LMResponse`.

        TODO(language-models): Current `BaseLM` returns `list[str | dict | None]`.
        Future LMs should return `LMResponse` directly, making this method a
        compatibility-only path for old/custom LMs.
        """
        if isinstance(outputs, LMResponse):
            return outputs
        return lm_response_from_legacy_outputs(outputs, request)

    def _prepare_request_state(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> _AdapterRequestState:
        """Build the normalized state for one adapter request.

        Preprocessing may rewrite the signature, inputs, and LM kwargs, so this copies caller-owned data before
        applying it and records both the source signature used for final outputs and the render signature used for
        prompt formatting.
        """
        copied_inputs = dict(inputs)
        copied_lm_kwargs = dict(lm_kwargs)
        render_signature = self._call_preprocess(lm, copied_lm_kwargs, signature, copied_inputs)
        tools = self._extract_lm_tools(copied_lm_kwargs)
        prepared_messages: list[LMMessage] = []

        if tools:
            render_signature, tool_result_messages = self._prepare_native_tool_result_inputs(
                render_signature,
                copied_inputs,
            )
            prepared_messages.extend(tool_result_messages)

        hidden_output_fields = tuple(
            field_name for field_name in signature.output_fields if field_name not in render_signature.output_fields
        )
        return _AdapterRequestState(
            source_signature=signature,
            render_signature=render_signature,
            inputs=copied_inputs,
            lm_kwargs=copied_lm_kwargs,
            tools=tools,
            prepared_messages=prepared_messages,
            hidden_output_fields=hidden_output_fields,
        )

    def _extract_lm_tools(self, lm_kwargs: dict[str, Any]) -> list[LMToolSpec]:
        """Remove native LM tool specs from LM kwargs and normalize them for request rendering."""
        raw_tools = lm_kwargs.pop("tools", None)
        if not raw_tools:
            return []
        tools = raw_tools if isinstance(raw_tools, list) else [raw_tools]
        return [self._coerce_lm_tool_spec(tool) for tool in tools]

    @staticmethod
    def _coerce_lm_tool_spec(tool: Any) -> LMToolSpec:
        """Convert supported tool shapes into the internal LMToolSpec representation."""
        if isinstance(tool, LMToolSpec):
            return tool
        if hasattr(tool, "to_lm_tool_spec"):
            return tool.to_lm_tool_spec()
        if isinstance(tool, dict):
            if "function" in tool:
                function = tool["function"]
                provider_data = {key: value for key, value in tool.items() if key not in {"type", "function"}}
                return LMToolSpec(
                    name=function.get("name"),
                    description=function.get("description"),
                    parameters=function.get("parameters", {}),
                    provider_data=provider_data,
                )
            return LMToolSpec(**tool)
        raise TypeError(f"Cannot convert {type(tool)!r} to LMToolSpec.")

    def _prepare_native_tool_result_inputs(
        self,
        render_signature: type[Signature],
        inputs: dict[str, Any],
    ) -> tuple[type[Signature], list[LMMessage]]:
        messages: list[LMMessage] = []
        for field_name, field_info in list(render_signature.input_fields.items()):
            value = inputs.get(field_name)
            tool_call_results = self._coerce_tool_call_results_value(value, field_info)
            if tool_call_results is None:
                if value is None and self._annotation_includes(getattr(field_info, "annotation", None), ToolCallResults):
                    inputs.pop(field_name, None)
                    render_signature = render_signature.delete(field_name)
                continue
            messages.extend(tool_call_results.to_lm_messages())
            inputs.pop(field_name, None)
            render_signature = render_signature.delete(field_name)
        return render_signature, messages

    def _parse_response(self, state: _AdapterRequestState, response: LMResponse) -> list[dict[str, Any]]:
        """Parse normalized LM outputs against the original source signature.

        Text is parsed with the render signature, while native response fields and tool call outputs are restored onto
        the source signature so request-time signature rewrites do not drop fields from the final value.
        """
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(state.source_signature)

        for output in response.outputs:
            if output.metadata.get("empty_legacy_outputs"):
                continue

            value: dict[str, Any] = {}
            parsed_any = False

            if output.text and state.render_signature.output_fields:
                value.update(self.parse(state.render_signature, output.text))
                parsed_any = True

            if output.tool_calls and tool_call_output_field_name:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [
                        {
                            "name": tool_call.name,
                            "args": tool_call.args,
                            **({"id": tool_call.id} if tool_call.id is not None else {}),
                        }
                        for tool_call in output.tool_calls
                    ]
                )
                parsed_any = True

            output_dict = output.to_output_dict()
            legacy_output = output.provider_output if output.provider_output is not None else output_dict
            for name, field_info in state.source_signature.output_fields.items():
                if (
                    isinstance(field_info.annotation, type)
                    and field_info.annotation in self.native_response_types
                    and issubclass(field_info.annotation, Type)
                ):
                    parsed_value = field_info.annotation.parse_lm_response(legacy_output)
                    if parsed_value is not None:
                        value[name] = parsed_value
                        parsed_any = True

            for field_name in state.source_signature.output_fields:
                value.setdefault(field_name, None)

            if not parsed_any:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=state.source_signature,
                    lm_response=str(output_dict),
                    message="The LM returned an empty or null response.",
                )

            if output.logprobs is not None:
                value["logprobs"] = output.logprobs

            values.append(value)

        return values

    def _value_to_lm_parts(self, value: Any, field_info: FieldInfo) -> list[LMPart]:
        rendered = format_field_value(field_info=field_info, value=value)
        return self._coerce_field_payload_to_lm_parts(rendered)

    def _coerce_field_payload_to_lm_parts(self, rendered: Any) -> list[LMPart]:
        if isinstance(rendered, str):
            if "<<CUSTOM-TYPE-START-IDENTIFIER>>" in rendered:
                return _split_legacy_custom_type_text_to_parts(rendered)
            return [LMTextPart(text=rendered)]
        if isinstance(rendered, list):
            return self._chat_dict_to_lm_message({"role": "user", "content": rendered}).parts
        if isinstance(rendered, dict):
            if "type" in rendered:
                return self._chat_dict_to_lm_message({"role": "user", "content": [rendered]}).parts
            return [LMTextPart(text=json.dumps(rendered, ensure_ascii=False))]
        return [LMTextPart(text=str(rendered))]

    def _wrap_input_field_parts(self, field_name: str, parts: list[LMPart]) -> list[LMPart]:
        raise NotImplementedError

    @staticmethod
    def _merge_adjacent_text_parts(parts: list[LMPart]) -> list[LMPart]:
        merged: list[LMPart] = []
        for part in parts:
            if (
                isinstance(part, LMTextPart)
                and merged
                and isinstance(merged[-1], LMTextPart)
                and not part.metadata
                and not merged[-1].metadata
            ):
                merged[-1] = LMTextPart(text=merged[-1].text + part.text)
            else:
                merged.append(part)
        return merged

    @classmethod
    def _coerce_tool_call_results_value(cls, field_value: Any, field_info: Any = None) -> ToolCallResults | None:
        if field_value is None:
            return None
        if isinstance(field_value, ToolCallResults):
            return field_value

        annotation = getattr(field_info, "annotation", None)
        if cls._annotation_includes(annotation, ToolCallResults):
            return ToolCallResults.model_validate(field_value)
        return None

    @classmethod
    def _coerce_tool_calls_value(cls, field_value: Any, field_info: Any = None) -> ToolCalls | None:
        if field_value is None:
            return None
        if isinstance(field_value, ToolCalls):
            return field_value

        annotation = getattr(field_info, "annotation", None)
        if cls._annotation_includes(annotation, ToolCalls):
            return ToolCalls.model_validate(field_value)
        return None

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
        state = self._prepare_request_state(lm, lm_kwargs, signature, inputs)
        request = self._render_request(lm, state, demos)
        response = self._call_lm(lm, request)
        return self._parse_response(state, response)

    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        state = self._prepare_request_state(lm, lm_kwargs, signature, inputs)
        request = self._render_request(lm, state, demos)
        response = await self._acall_lm(lm, request)
        return self._parse_response(state, response)

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
        state = _AdapterRequestState(
            source_signature=signature,
            render_signature=signature,
            inputs=dict(inputs),
            lm_kwargs={},
        )
        messages = self._coerce_lm_messages(self._format_request_messages(state, demos))
        return [message_to_openai_chat(message) for message in messages]

    @with_callbacks
    def _format_request_with_callbacks(
        self, state: _AdapterRequestState, demos: list[dict[str, Any]]
    ) -> list[LMMessage | dict[str, Any]]:
        """Render state-aware messages while preserving the existing adapter format callback."""
        return self._format_request_messages(state, demos)

    def _format_request_messages(
        self,
        state: _AdapterRequestState,
        demos: list[dict[str, Any]],
    ) -> list[LMMessage | dict[str, Any]]:
        """Render LM messages from prepared request state.

        The render signature is used for system instructions and demos. If the signature includes a history field, that
        field is removed only from the per-turn input signature so history is expanded into messages instead of rendered
        again as raw current input.
        """
        render_signature = state.render_signature
        inputs_copy = dict(state.inputs)
        conversation_history: list[LMMessage] = []
        input_signature = render_signature

        history_field_name = self._get_history_field_name(render_signature)
        if history_field_name:
            input_signature = render_signature.delete(history_field_name)
            history_obj = inputs_copy.pop(history_field_name, None)
            if history_obj is not None:
                history_signature = (
                    state.source_signature.delete(history_field_name)
                    if history_field_name in state.source_signature.input_fields
                    else input_signature
                )
                conversation_history = self.format_history(
                    history_obj,
                    history_signature,
                    use_native_tool_calls=bool(state.tools),
                )

        messages: list[LMMessage | dict[str, Any]] = [
            {"role": "system", "content": self.format_system_message(render_signature)}
        ]

        messages.extend(self._format_demos_as_lm_messages(render_signature, demos))
        messages.extend(conversation_history)
        messages.extend(state.prepared_messages)
        messages.extend(self._format_input_messages(input_signature, inputs_copy, main_request=True))
        return messages

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
        return [message_to_openai_chat(message) for message in self._format_demos_as_lm_messages(signature, demos)]

    def _format_demos_as_lm_messages(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[LMMessage]:
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

        messages: list[LMMessage] = []

        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            messages.extend(
                self._format_input_messages(
                    signature,
                    demo,
                    main_request=False,
                    prefix=incomplete_demo_prefix,
                )
            )
            messages.extend(
                self._format_output_messages(
                    signature,
                    demo,
                    missing_field_message="Not supplied for this particular example. ",
                )
            )

        for demo in complete_demos:
            messages.extend(self._format_input_messages(signature, demo, main_request=False))
            messages.extend(
                self._format_output_messages(
                    signature,
                    demo,
                    missing_field_message="Not supplied for this conversation history message. ",
                )
            )

        return messages

    def _format_input_messages(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        *,
        main_request: bool,
        prefix: str = "",
        suffix: str = "",
    ) -> list[LMMessage]:
        regular_inputs = {
            key: value
            for key, value in inputs.items()
            if not (value is None and key in signature.input_fields and signature.input_fields[key].default is None)
        }
        if self._should_render_input_as_parts(signature, regular_inputs):
            parts: list[LMPart] = []
            if prefix:
                parts.append(LMTextPart(text=prefix))
            for key, field_info in signature.input_fields.items():
                if key not in regular_inputs:
                    continue
                if parts:
                    parts.append(LMTextPart(text="\n\n"))
                parts.extend(
                    self._wrap_input_field_parts(key, self._value_to_lm_parts(regular_inputs[key], field_info))
                )
            if main_request:
                output_requirements_fn = getattr(self, "user_message_output_requirements", lambda _signature: None)
                output_requirements = output_requirements_fn(signature)
                if output_requirements is not None:
                    if parts:
                        parts.append(LMTextPart(text="\n\n"))
                    parts.append(LMTextPart(text=output_requirements))
            if suffix:
                if parts:
                    parts.append(LMTextPart(text="\n\n"))
                parts.append(LMTextPart(text=suffix))
            parts = self._merge_adjacent_text_parts(parts)
            return [LMMessage(role="user", parts=parts)] if parts else []

        content = self.format_user_message_content(
            signature,
            regular_inputs,
            prefix=prefix,
            suffix=suffix,
            main_request=main_request,
        )
        return [LMMessage.model_validate({"role": "user", "content": content})] if self._has_content(content) else []

    def _format_output_messages(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        *,
        missing_field_message: str | None,
    ) -> list[LMMessage]:
        content = self.format_assistant_message_content(
            signature,
            outputs,
            missing_field_message=missing_field_message,
        )
        return (
            [LMMessage.model_validate({"role": "assistant", "content": content})] if self._has_content(content) else []
        )

    def _should_render_input_as_parts(self, signature: type[Signature], inputs: dict[str, Any]) -> bool:
        for key, value in inputs.items():
            field_info = signature.input_fields.get(key)
            if field_info is None:
                continue
            if any(
                not isinstance(part, LMTextPart) or part.metadata.get("legacy_content_block")
                for part in self._value_to_lm_parts(value, field_info)
            ):
                return True
        return False

    @staticmethod
    def _has_content(content: Any) -> bool:
        if isinstance(content, str):
            return bool(content.strip())
        return bool(content)

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
            if self._annotation_includes(field.annotation, ToolCalls):
                return name
        return None

    @classmethod
    def _annotation_includes(cls, annotation: Any, target: type) -> bool:
        if annotation is target:
            return True
        return any(cls._annotation_includes(arg, target) for arg in get_args(annotation))

    def force_tool_call_config(self, tool_name: str) -> dict[str, Any]:
        if not self.use_native_function_calling:
            return {}
        return {"tool_choice": {"mode": "required", "allowed": [tool_name]}}

    def format_history(
        self,
        history: History,
        signature: type[Signature],
        *,
        use_native_tool_calls: bool = False,
    ) -> list[LMMessage]:
        history = history if isinstance(history, History) else History.model_validate(history)
        messages: list[LMMessage] = []

        for entry in history.messages:
            if not isinstance(entry, dict):
                continue

            input_values = {key: value for key, value in entry.items() if key in signature.input_fields}
            output_values = {key: value for key, value in entry.items() if key in signature.output_fields}
            known_keys = set(input_values) | set(output_values)
            unknown_values = {key: value for key, value in entry.items() if key not in known_keys}

            native_tool_results: list[LMMessage] = []
            if use_native_tool_calls:
                regular_inputs = {}
                for key, value in input_values.items():
                    tool_results = self._coerce_tool_call_results_value(value, signature.input_fields.get(key))
                    if tool_results is None:
                        regular_inputs[key] = value
                    else:
                        native_tool_results.extend(tool_results.to_lm_messages())
            else:
                regular_inputs = input_values

            if regular_inputs:
                messages.extend(self._format_input_messages(signature, regular_inputs, main_request=False))

            native_tool_call_parts: list[LMPart] = []
            regular_outputs = {}
            for key, value in output_values.items():
                tool_calls = None
                if use_native_tool_calls:
                    tool_calls = self._coerce_tool_calls_value(value, signature.output_fields.get(key))
                if tool_calls is not None:
                    native_tool_call_parts.extend(tool_calls.to_lm_parts())
                else:
                    regular_outputs[key] = value

            assistant_text = self._format_history_assistant_text(signature, regular_outputs, unknown_values)
            if use_native_tool_calls and native_tool_call_parts:
                parts: list[LMPart] = []
                if assistant_text:
                    parts.append(LMTextPart(text=assistant_text))
                parts.extend(native_tool_call_parts)
                messages.append(LMMessage(role="assistant", parts=parts))
            elif assistant_text:
                messages.append(LMMessage.model_validate({"role": "assistant", "content": assistant_text}))

            messages.extend(native_tool_results)

        return messages

    def _format_history_assistant_text(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        unknown_outputs: dict[str, Any],
    ) -> str | None:
        text_signature = signature
        for field_name, field_info in signature.output_fields.items():
            if self._annotation_includes(field_info.annotation, ToolCalls):
                text_signature = text_signature.delete(field_name)

        sections = []
        signature_outputs = {key: value for key, value in outputs.items() if key in text_signature.output_fields}
        if signature_outputs:
            sections.append(
                self.format_assistant_message_content(
                    text_signature,
                    signature_outputs,
                    missing_field_message="Not supplied for this conversation history message. ",
                )
            )

        for key, value in unknown_outputs.items():
            formatted_value = "\n".join(str(item) for item in value) if isinstance(value, list) else str(value)
            sections.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if unknown_outputs and not any(section.endswith("[[ ## completed ## ]]") for section in sections):
            sections.append("[[ ## completed ## ]]")

        if signature_outputs and not unknown_outputs and len(sections) == 1:
            content = sections[0]
        else:
            content = "\n\n".join(section.strip() for section in sections if section).strip()
        return content or None

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

        # Remove the history field from the inputs
        del inputs[history_field_name]
        history = History(messages=conversation_history)
        return [message_to_openai_chat(message) for message in self.format_history(history, signature)]

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
