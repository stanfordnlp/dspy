import logging
from typing import Any, get_origin

from dspy.adapters.types import Audio, File, History, Image, Type
from dspy.adapters.types.base_type import split_message_content_for_custom_types
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import message_to_openai_chat, provider_tool_call_to_part, to_openai_chat_request
from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMPart,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMThinkingPart,
    LMToolSpec,
)
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
        renderers: dict[type, Any] | None = None,
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
        self.renderers = renderers or {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def plan_fields(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Plan which fields are rendered by the prompt adapter and which use native LM features."""
        lm_kwargs = dict(lm_kwargs)
        prompt_signature = signature
        inputs = dict(inputs)
        messages: list[LMMessage] = []
        user_parts: list[LMPart] = []
        tools: list[LMToolSpec] = []

        output_parsers: dict[str, Any] = {}

        for name, field in list(prompt_signature.input_fields.items()):
            if name not in inputs:
                continue
            value = inputs[name]
            custom_edits = self._run_custom_renderers(
                role="input",
                field_name=name,
                field=field,
                value=value,
                signature=prompt_signature,
                lm=lm,
                lm_kwargs=lm_kwargs,
                inputs=inputs,
            )
            if custom_edits is not None:
                prompt_signature = self._apply_renderer_edits(prompt_signature, inputs, custom_edits, messages, user_parts, tools, lm_kwargs, output_parsers)
                continue

        if self.use_native_function_calling:
            tool_call_input_field_name = self._get_tool_call_input_field_name(prompt_signature)
            tool_call_output_field_name = self._get_tool_call_output_field_name(prompt_signature)

            if tool_call_output_field_name and tool_call_input_field_name is None:
                raise ValueError(
                    f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                    "input field with type `list[dspy.Tool]`."
                )

            if tool_call_output_field_name and lm.supports_function_calling:
                tool_values = inputs[tool_call_input_field_name]
                tool_values = tool_values if isinstance(tool_values, list) else [tool_values]
                tools.extend(self._tool_to_lm_tool_spec(tool) for tool in tool_values)
                prompt_signature = prompt_signature.delete(tool_call_output_field_name).delete(tool_call_input_field_name)
                inputs.pop(tool_call_input_field_name, None)

        for name, field in list(prompt_signature.output_fields.items()):
            custom_edits = self._run_custom_renderers(
                role="output",
                field_name=name,
                field=field,
                value=None,
                signature=prompt_signature,
                lm=lm,
                lm_kwargs=lm_kwargs,
                inputs=inputs,
            )
            if custom_edits is not None:
                prompt_signature = self._apply_renderer_edits(prompt_signature, inputs, custom_edits, messages, user_parts, tools, lm_kwargs, output_parsers)
                continue
            if field.annotation == Reasoning and Reasoning in self.native_response_types:
                reasoning_signature = self._plan_native_reasoning(prompt_signature, name, lm, lm_kwargs)
                if reasoning_signature is not prompt_signature:
                    prompt_signature = reasoning_signature
            elif field.annotation in self.native_response_types and field.annotation is Citations:
                if getattr(lm, "model", "").startswith("anthropic/"):
                    prompt_signature = prompt_signature.delete(name)

        return {
            "original_signature": signature,
            "prompt_signature": prompt_signature,
            "inputs": inputs,
            "lm_kwargs": lm_kwargs,
            "messages": messages,
            "user_parts": user_parts,
            "tools": tools,
            "output_parsers": output_parsers,
        }

    def _run_custom_renderers(
        self,
        *,
        role: str,
        field_name: str,
        field: Any,
        value: Any,
        signature: type[Signature],
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        inputs: dict[str, Any],
    ) -> dict[str, Any] | None:
        renderers = self._renderers_for_annotation(field.annotation)
        for renderer in renderers:
            edits = renderer(
                role=role,
                field_name=field_name,
                field=field,
                value=value,
                signature=signature,
                adapter=self,
                lm=lm,
                lm_kwargs=lm_kwargs,
                inputs=inputs,
            )
            if edits is not None:
                return edits
        return None

    def _renderers_for_annotation(self, annotation: Any) -> list[Any]:
        renderers: list[Any] = []
        for marker, marker_renderers in self.renderers.items():
            try:
                matches = annotation == marker or (isinstance(annotation, type) and issubclass(annotation, marker))
            except TypeError:
                matches = False
            if matches:
                if isinstance(marker_renderers, list | tuple):
                    renderers.extend(marker_renderers)
                else:
                    renderers.append(marker_renderers)
        return renderers

    def _apply_renderer_edits(
        self,
        prompt_signature: type[Signature],
        inputs: dict[str, Any],
        edits: dict[str, Any],
        messages: list[LMMessage],
        user_parts: list[LMPart],
        tools: list[LMToolSpec],
        lm_kwargs: dict[str, Any],
        output_parsers: dict[str, Any],
    ) -> type[Signature]:
        for field_name in edits.get("remove_from_prompt", []):
            if field_name in prompt_signature.fields:
                prompt_signature = prompt_signature.delete(field_name)
            inputs.pop(field_name, None)
        messages.extend(edits.get("messages", []))
        user_parts.extend(edits.get("user_parts", []))
        tools.extend(edits.get("tools", []))
        lm_kwargs.update(edits.get("lm_kwargs", {}))
        output_parsers.update(edits.get("output_parsers", {}))
        return prompt_signature

    def render_request(
        self,
        plan: dict[str, Any],
        lm: BaseLM,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> LMRequest:
        """Render the prompt-facing signature into a normalized LM request."""
        messages = self.render_messages(plan["prompt_signature"], demos, plan["inputs"])
        messages = self._apply_planned_messages(messages, plan)
        request_kwargs = dict(plan["lm_kwargs"])
        tools = list(plan.get("tools", []))
        return LMRequest.from_call(model=lm.model, messages=messages, tools=tools, **request_kwargs)

    def call_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        """Call a legacy `BaseLM` through a normalized request/response boundary."""
        data = self._legacy_call_kwargs(request)
        outputs = lm(messages=data.pop("messages"), **data)
        return self.normalize_legacy_outputs(outputs, request)

    async def acall_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
        """Async variant of `call_lm`."""
        data = self._legacy_call_kwargs(request)
        outputs = await lm.acall(messages=data.pop("messages"), **data)
        return self.normalize_legacy_outputs(outputs, request)

    def _legacy_call_kwargs(self, request: LMRequest) -> dict[str, Any]:
        # Legacy `BaseLM` currently accepts OpenAI-chat-shaped `messages` for all model types;
        # `dspy.LM` converts those messages to text-completion or Responses requests internally.
        data = to_openai_chat_request(request)
        data.pop("model", None)
        data["messages"] = split_message_content_for_custom_types(data["messages"])
        if request.config.cache is not None:
            if request.config.cache.enabled is not None:
                data["cache"] = request.config.cache.enabled
            if request.config.cache.rollout_id is not None:
                data["rollout_id"] = request.config.cache.rollout_id
        return data

    def _apply_planned_messages(self, messages: list[LMMessage], plan: dict[str, Any]) -> list[LMMessage]:
        planned_messages = list(plan.get("messages", []))
        user_parts = list(plan.get("user_parts", []))
        if planned_messages:
            insert_at = self._last_user_message_index(messages)
            if insert_at is None:
                insert_at = len(messages)
            messages[insert_at:insert_at] = planned_messages
        if user_parts:
            user_index = self._last_user_message_index(messages)
            if user_index is None:
                messages.append(LMMessage(role="user", parts=user_parts))
            else:
                messages[user_index].parts.extend(user_parts)
        return messages

    def _last_user_message_index(self, messages: list[LMMessage]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None

    def _history_to_lm_messages(self, signature: type[Signature], history: History) -> list[LMMessage]:
        messages: list[LMMessage] = []
        for turn in history.messages:
            messages.append(LMMessage(role="user", parts=[LMTextPart(text=self.format_user_message_content(signature, turn))]))
            messages.append(
                LMMessage(role="assistant", parts=[LMTextPart(text=self.format_assistant_message_content(signature, turn))])
            )
        return messages

    def _image_to_lm_part(self, image: Image) -> LMImagePart:
        source = image.url
        if source.startswith("data:") and "," in source:
            header, data = source.split(",", 1)
            media_type = header.removeprefix("data:").split(";", 1)[0]
            return LMImagePart(data=data, media_type=media_type)
        return LMImagePart(url=source)

    def _audio_to_lm_part(self, audio: Audio) -> LMAudioPart:
        return LMAudioPart(data=audio.data, media_type=f"audio/{audio.audio_format}")

    def _file_to_lm_part(self, file: File) -> LMBinaryPart:
        if file.file_data is not None:
            media_type, data = self._split_data_uri(file.file_data)
            return LMBinaryPart(data=data, media_type=media_type, filename=file.filename)
        if file.file_id is not None:
            return LMBinaryPart(file_id=file.file_id, filename=file.filename)
        raise ValueError("File must have file_data or file_id.")

    def _split_data_uri(self, value: str) -> tuple[str, str]:
        if value.startswith("data:") and "," in value:
            header, data = value.split(",", 1)
            return header.removeprefix("data:").split(";", 1)[0], data
        return "application/octet-stream", value

    def _tool_to_lm_tool_spec(self, tool: Tool) -> LMToolSpec:
        args = tool.args or {}
        return LMToolSpec(
            name=tool.name or "",
            description=tool.desc,
            parameters={"type": "object", "properties": args, "required": list(args.keys())},
        )

    def _plan_native_reasoning(
        self,
        signature: type[Signature],
        field_name: str,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
    ) -> type[Signature]:
        reasoning_effort = lm_kwargs.get("reasoning_effort", lm.kwargs.get("reasoning_effort", "low"))
        if reasoning_effort is None or not lm.supports_reasoning:
            return signature
        if "gpt-5" in lm.model and getattr(lm, "model_type", None) == "chat":
            return signature
        lm_kwargs["reasoning_effort"] = reasoning_effort
        return signature.delete(field_name)

    def normalize_legacy_outputs(self, outputs: list[dict[str, Any] | str], request: LMRequest) -> LMResponse:
        """Convert legacy adapter outputs into a normalized `LMResponse` immediately after the LM call."""
        return LMResponse(model=request.model, outputs=[self._legacy_output_to_lm_output(output) for output in outputs])

    def _legacy_output_to_lm_output(self, output: dict[str, Any] | str) -> LMOutput:
        if isinstance(output, str):
            return LMOutput(parts=[LMTextPart(text=output)])

        parts = []
        text = output.get("text")
        if text:
            parts.append(LMTextPart(text=text))
        reasoning = output.get("reasoning_content")
        if reasoning:
            parts.append(LMThinkingPart(text=str(reasoning)))
        for tool_call in output.get("tool_calls") or []:
            parts.append(provider_tool_call_to_part(tool_call))
        for citation in output.get("citations") or []:
            from dspy.clients.openai_format import citation_to_part

            parts.append(citation_to_part(citation))
        return LMOutput(parts=parts, logprobs=output.get("logprobs"))

    def parse_response(self, plan: dict[str, Any], response: LMResponse, lm: BaseLM) -> list[dict[str, Any]]:
        """Parse a normalized LM response into dictionaries matching the original signature."""
        values = []
        original_signature = plan["original_signature"]
        prompt_signature = plan["prompt_signature"]
        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for output in response.outputs:
            if output.text:
                value = self.parse(prompt_signature, output.text)
                for field_name in original_signature.output_fields.keys():
                    if field_name not in value:
                        value[field_name] = None
            elif output.tool_calls and tool_call_output_field_name:
                value = {field_name: None for field_name in original_signature.output_fields.keys()}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            if output.tool_calls and tool_call_output_field_name:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [{"name": call.name, "args": call.args} for call in output.tool_calls]
                )

            for field_name, parser in plan.get("output_parsers", {}).items():
                parsed_value = parser(
                    field_name=field_name,
                    output=output,
                    response=response,
                    adapter=self,
                    lm=lm,
                    plan=plan,
                )
                if parsed_value is not None:
                    value[field_name] = parsed_value

            # Parse custom types that do not rely on the `Adapter.parse()` text parser.
            output_dict = output.to_output_dict()
            for name, field in original_signature.output_fields.items():
                if (
                    isinstance(field.annotation, type)
                    and field.annotation in self.native_response_types
                    and issubclass(field.annotation, Type)
                ):
                    parsed_value = field.annotation.parse_lm_response(output_dict)
                    if parsed_value is not None:
                        value[name] = parsed_value

            if output.logprobs is not None:
                value["logprobs"] = output.logprobs

            values.append(value)

        return values

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
        plan = self.plan_fields(lm, lm_kwargs, signature, inputs)
        request = self.render_request(plan, lm, demos, inputs)
        response = self.call_lm(lm, request)
        return self.parse_response(plan, response, lm)

    async def acall(
        self,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        plan = self.plan_fields(lm, lm_kwargs, signature, inputs)
        request = self.render_request(plan, lm, demos, inputs)
        response = await self.acall_lm(lm, request)
        return self.parse_response(plan, response, lm)

    def render_messages(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[LMMessage]:
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

        return [message if isinstance(message, LMMessage) else LMMessage(**message) for message in messages]

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return legacy OpenAI-style messages for compatibility.

        Deprecated:
            Since DSPy 3.3. Adapter internals now use `render_messages()` and
            normalized LM message types. This compatibility method will be
            removed in DSPy 3.5.
        """
        messages = [message_to_openai_chat(message) for message in self.render_messages(signature, demos, inputs)]
        return split_message_content_for_custom_types(messages)

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
