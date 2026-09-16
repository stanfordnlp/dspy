import json
import logging
from dataclasses import dataclass
from typing import Any, get_origin

from dspy._vendor.lm15.types import tool_result
from dspy.adapters.types import History, Type
from dspy.adapters.types.base_type import parts_from_text
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.utils import apply_output_field_defaults, serialize_for_json
from dspy.clients.base_lm import BaseLM
from dspy.clients.capabilities import with_capability_planning
from dspy.clients.requests import build_request
from dspy.experimental import Citations
from dspy.lm15 import Message, Response, TextPart, ToolCallPart
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]
_TOOL_CALL_RESULTS_SIGNATURE = Signature({"tool_call_results": (ToolCallResults, InputField())})


@dataclass(frozen=True)
class Prompt:
    """What an adapter renders: system instructions plus the conversation turns.

    Both halves use `dspy.lm15` objects. `messages` are `Message`s in user,
    assistant and tool roles; `system` is the instruction text (or content
    parts) that becomes `Request.system`.
    """

    messages: tuple[Message, ...]
    system: str | tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "messages", tuple(self.messages))


def user_message(text: str) -> Message:
    """A user message whose custom-type markers become lm15 content parts."""
    return Message.user(parts_from_text(text))


def response_text(response: Response) -> str | None:
    """The visible answer text, even when tool calls or media accompany it."""
    texts = [part.text for part in response.message.parts_of(TextPart)]
    return "".join(texts) if texts else None


def prompt_to_openai_messages(prompt: Prompt) -> list[dict[str, Any]]:
    """Write a prompt as OpenAI chat messages, for fine-tuning files and display.

    This is lm15's own Chat Completions writer; DSPy does not keep a second one.
    """
    from dspy.clients.lm15_boundary import request_kwargs
    from dspy.lm15 import Request

    request = Request(model="conversion-only", messages=prompt.messages, system=prompt.system)
    return request_kwargs(request, "chat")["messages"]


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
        """Decide which fields the model handles natively, and adjust the LM options.

        Returns the signature the prompt is rendered from: fields handled by a
        native feature (tool calling, reasoning, ...) are removed from it and
        filled in from the `Response` in `_call_postprocess`.
        """
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

                lm_kwargs["tools"] = [tool.as_function_tool() for tool in tools]
                if self.parallel_tool_calls is not None and lm_kwargs.get("parallel_tool_calls") is None:
                    lm_kwargs["parallel_tool_calls"] = self.parallel_tool_calls
                if lm_kwargs.get("parallel_tool_calls") is not None:
                    lm_kwargs.setdefault("tool_choice", "auto")

                signature = signature.delete(tool_call_output_field_name)
                signature = signature.delete(tool_call_input_field_name)
            elif tool_call_output_field_name:
                for key in ("tools", "tool_choice", "parallel_tool_calls"):
                    lm_kwargs.pop(key, None)

        # Custom types that use native LM features, e.g., reasoning, citations.
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
        responses: list[Response],
    ) -> list[dict[str, Any]]:
        """Parse each `Response` into the output fields of the original signature."""
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for response in responses:
            text = response_text(response)
            tool_calls = response.tool_calls if tool_call_output_field_name else []

            if text and not tool_calls:
                value = self.parse(processed_signature, text)
            elif tool_calls:
                try:
                    value = self.parse(processed_signature, text) if text and processed_signature.output_fields else {}
                except AdapterParseError:
                    value = {}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=text or "",
                    message="The LM returned an empty or null response.",
                )

            # Fields removed for native features are absent from the processed parse
            value = apply_output_field_defaults(original_signature, value)
            for field_name in original_signature.output_fields:
                value.setdefault(field_name, None)

            if tool_calls:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [{"id": call.id, "name": call.name, "args": call.input} for call in tool_calls]
                )

            # Custom types that read their value from the response itself.
            for name, field in original_signature.output_fields.items():
                if (
                    isinstance(field.annotation, type)
                    and field.annotation in self.native_response_types
                    and issubclass(field.annotation, Type)
                ):
                    parsed_value = field.annotation.parse_lm_response(response)
                    if parsed_value is not None:
                        value[name] = parsed_value

            if response.logprobs is not None:
                value["logprobs"] = response.logprobs

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
        prompt = self.format(processed_signature, demos, inputs)
        request = build_request(lm, prompt, lm_kwargs)
        responses = lm.generate(request, **_execution_options(lm_kwargs))
        return self._call_postprocess(processed_signature, signature, responses)

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
        prompt = self.format(processed_signature, demos, inputs)
        request = build_request(lm, prompt, lm_kwargs)
        responses = await lm.agenerate(request, **_execution_options(lm_kwargs))
        return self._call_postprocess(processed_signature, signature, responses)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> Prompt:
        """Render the prompt for the LM call.

        This method converts the DSPy structured input along with few-shot examples and conversation history into
        a system instruction plus multiturn lm15 messages. For custom adapters, this method can be overridden to
        customize the rendering.

        The recommended structure is:
        ```
        Prompt(
            system=system_message,  # field descriptions, field structure and task description
            messages=(
                # Begin few-shot examples
                Message.user(few_shot_example_1_input),
                Message.assistant(few_shot_example_1_output),
                ...
                # End few-shot examples
                # Begin conversation history
                Message.user(conversation_history_1_input),
                Message.assistant(conversation_history_1_output),
                ...
                # End conversation history
                Message.user(current_input),
            ),
        )
        ```

        Args:
            signature: The DSPy signature for which to format the prompt.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            The rendered `Prompt`.
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
        messages.extend(self.format_demos(signature, demos))
        if history_field_name:
            # Conversation history and current input
            content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
            messages.extend(conversation_history)
            if content:
                messages.append(user_message(content))
        else:
            # Only current input
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)
            if content:
                messages.append(user_message(content))

        return Prompt(system=system_message or None, messages=tuple(messages))

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

    def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[Message]:
        """Format the few-shot examples.

        This method formats the few-shot examples as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the few-shot examples.
            demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
                the signature.

        Returns:
            A list of multiturn lm15 messages.
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
            messages.append(user_message(self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix)))
            messages.append(Message.assistant(self.format_assistant_message_content(
                signature, demo, missing_field_message="Not supplied for this particular example. "
            )))

        for demo in complete_demos:
            messages.append(user_message(self.format_user_message_content(signature, demo)))
            messages.append(Message.assistant(self.format_assistant_message_content(
                signature, demo, missing_field_message="Not supplied for this conversation history message. "
            )))

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
    ) -> list[Message]:
        """Format the conversation history.

        This method formats the conversation history and the current input as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the conversation history.
            history_field_name: The name of the history field in the signature.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn lm15 messages.
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
                messages.append(user_message(user_content))

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
                    parts = [TextPart(content)] if content else []
                    if tool_call_results is not None:
                        parts.extend(
                            ToolCallPart(id=tool_call.id, name=tool_call.name, input=serialize_for_json(tool_call.args))
                            for tool_call in tool_calls.tool_calls
                        )
                    messages.append(Message.assistant(parts))

                if tool_call_results is not None:
                    messages.append(Message.tool([
                        tool_result(result.call_id, _tool_result_content(result.value), name=result.name)
                        for result in tool_call_results.tool_call_results
                    ]))
                continue

            assistant_values = message
            if tool_call_field_name is not None and tool_call_results is not None:
                assistant_values = dict(message)
                assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})

            assistant_content = self.format_assistant_message_content(signature, assistant_values)
            if assistant_content:
                messages.append(Message.assistant(assistant_content))
            if tool_call_results is not None:
                result_input = {"tool_call_results": tool_call_results}
                content = self.format_user_message_content(_TOOL_CALL_RESULTS_SIGNATURE, result_input)
                messages.append(user_message(content))

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


def _execution_options(lm_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Execution controls and client settings an adapter forwards to `generate`.

    Generation options are already inside the Request; these are the keys the
    DSPy LM layer (`n`, `cache`, `rollout_id`) and engines (`api_base`, ...) own.
    """
    from dspy.clients.requests import CLIENT_KEYS

    options = {}
    if lm_kwargs.get("n", lm_kwargs.get("num_generations")) is not None:
        options["n"] = lm_kwargs.get("n", lm_kwargs.get("num_generations"))
    for key in ("cache", "rollout_id", *CLIENT_KEYS):
        if lm_kwargs.get(key) is not None:
            options[key] = lm_kwargs[key]
    return options


def _tool_calls_from_message(message: dict[str, Any]) -> tuple[str | None, ToolCalls | None]:
    for name, value in message.items():
        if isinstance(value, ToolCalls) or (isinstance(value, dict) and "tool_calls" in value):
            return name, ToolCalls.model_validate(value)
    return None, None


def _tool_result_content(value: Any) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(serialize_for_json(value), ensure_ascii=False)
