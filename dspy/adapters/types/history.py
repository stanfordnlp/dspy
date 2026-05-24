from typing import Any, Callable

import pydantic
from pydantic import Field

from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.adapters.utils import serialize_for_json


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

    messages: list[dict[str, Any]] = Field(default_factory=list)

    model_config = pydantic.ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def __init__(self, *args: Any, compact_fn: Callable[["History"], None] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_compact_fn", compact_fn)

    @pydantic.model_serializer()
    def serialize_model(self) -> dict[str, Any]:
        return {"messages": [self._serialize_message(message) for message in self.messages]}

    def compact_if_needed(self) -> None:
        compact_fn = getattr(self, "_compact_fn", None)
        if compact_fn is not None:
            compact_fn(self)

    def append(self, message: dict[str, Any]) -> dict[str, Any]:
        message = dict(message)
        self.messages.append(message)
        return message

    @staticmethod
    def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
        serialized = {}
        for key, value in message.items():
            if isinstance(value, ToolCalls):
                serialized[key] = {
                    "tool_calls": [
                        {
                            "name": tool_call.name,
                            "args": serialize_for_json(tool_call.args),
                            **({"id": tool_call.id} if tool_call.id is not None else {}),
                        }
                        for tool_call in value.tool_calls
                    ]
                }
            elif isinstance(value, ToolCallResults):
                serialized[key] = value.format()
            elif hasattr(value, "model_dump"):
                serialized[key] = value.model_dump()
            else:
                serialized[key] = value
        return serialized
