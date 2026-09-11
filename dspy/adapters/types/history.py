from typing import Any

import pydantic

from dspy.core.types import LMCompactionPart


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history contains optional compacted context followed by a
    list of recent messages. Compacted context can be a portable text summary
    or provider-native state in an ``LMCompactionPart``. Each message entity
    should have keys from the associated signature. For example, if you have
    the following signature:

    ```
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()
    ```

    Then the history messages should be dictionaries with keys "question" and
    "answer". A string ``compaction`` is placed before the recent messages as
    ordinary user context. An ``LMCompactionPart`` is replayed in its provider's
    native format. Compacted context is not filtered against signature fields.

    Examples:
        ```
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History(
            compaction="The user is comparing European capitals.",
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
    compaction: str | LMCompactionPart | None = pydantic.Field(default=None, exclude_if=lambda value: value is None)

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )
