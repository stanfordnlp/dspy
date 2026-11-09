from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pydantic
from pydantic_core import PydanticSerializationError

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.adapters.types.tool import ToolCalls
    from dspy.lm15 import Message
    from dspy.signatures.signature import Signature


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated signature.
    For example, if you have the following signature:

    ```
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Examples:
        ```
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?", history=history)
        ```

    Example of capturing the conversation history:
        ```
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?")
        history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = predict(question="Are you sure?", history=history)
        ```
    """

    messages: list[dict[str, Any]]

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    @pydantic.field_serializer("messages", mode="wrap", when_used="json")
    def _serialize_messages(self, messages: list[dict[str, Any]], handler: pydantic.SerializerFunctionWrapHandler):
        """Keep normal serialization; unsupported values use the adapter's text fallback, without mutating History."""
        from dspy.adapters.utils import serialize_for_json

        try:
            return handler(messages)
        except PydanticSerializationError:
            return handler(serialize_for_json(messages, for_prompt=False))

    def _remap_fields(self, names: dict[str, str]) -> History:
        """Copy recorded field names to match the adapter's adapted signature, without mutating the caller."""
        return self.model_copy(update={
            "messages": [{names.get(key, key): value for key, value in message.items()} for message in self.messages],
        })

    def format(self, adapter: Adapter, signature: type[Signature]) -> list[Message]:
        """Render recorded events using the adapter's field syntax and native-tool policy.

        History owns event dispatch and conversation ordering. REPL events own their action/observation
        representation; the adapter only renders fields and does not inspect their serialized layout.
        """
        from dspy._vendor.lm15.types import tool_result
        from dspy.adapters.base import _tool_result_content, user_message
        from dspy.adapters.types.tool import ToolCallResults, ToolCalls
        from dspy.adapters.utils import serialize_for_json
        from dspy.lm15 import Message, TextPart, ToolCallPart
        from dspy.primitives.repl_types import format_repl_event, is_repl_event
        from dspy.signatures.field import InputField
        from dspy.signatures.signature import Signature

        results_signature = Signature({"tool_call_results": (ToolCallResults, InputField())})
        messages = []
        for message in self.messages:
            # Recognize structured REPL events before tool-shaped application data inside their inputs.
            if is_repl_event(message):
                messages.extend(format_repl_event(adapter, signature, message))
                continue

            tool_call_field_name, tool_calls = _tool_calls_from_message(message)
            tool_call_results = (
                ToolCallResults.model_validate(tool_calls.tool_call_results)
                if tool_calls is not None and tool_calls.tool_call_results is not None
                else None
            )

            # Retain the original reminder for cache-prefix reuse, but not on output-only events.
            user_content = adapter.format_user_message_content(
                signature, message, main_request=any(name in message for name in signature.input_fields)
            )
            if user_content:
                messages.append(user_message(user_content))

            if adapter.use_native_function_calling and tool_calls is not None:
                content_signature = signature
                for name, field in signature.output_fields.items():
                    if field.annotation == ToolCalls or message.get(name) is None:
                        content_signature = content_signature.delete(name)

                content = (
                    adapter.format_assistant_message_content(content_signature, message)
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

            assistant_content = adapter.format_assistant_message_content(signature, assistant_values)
            if assistant_content:
                messages.append(Message.assistant(assistant_content))
            if tool_call_results is not None:
                content = adapter.format_user_message_content(
                    results_signature, {"tool_call_results": tool_call_results}
                )
                messages.append(user_message(content))
        return messages


def _tool_calls_from_message(message: dict[str, Any]) -> tuple[str | None, ToolCalls | None]:
    from dspy.adapters.types.tool import ToolCalls

    for name, value in message.items():
        if isinstance(value, ToolCalls) or (isinstance(value, dict) and "tool_calls" in value):
            return name, ToolCalls.model_validate(value)
    return None, None
