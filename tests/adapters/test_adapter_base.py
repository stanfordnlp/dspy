import dspy
from tests.adapters.conftest import format_messages_and_lm_kwargs


def test_adapter_does_not_require_native_tool_call_format_kwarg_when_unused():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class OldStyleFormatAdapter(dspy.Adapter):
        def format(self, signature, demos, inputs):
            return [{"role": "user", "content": inputs["question"]}]

        def parse(self, signature, completion):
            return {"answer": completion}

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        OldStyleFormatAdapter(),
        QA,
        [],
        {"question": "Q"},
    )

    assert messages == [{"role": "user", "content": "Q"}]
    assert lm_kwargs == {}
