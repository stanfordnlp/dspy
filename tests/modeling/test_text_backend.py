import textwrap
from dspy.modeling import TextBackend
from dspy.primitives.example import Example
from dspy.signatures.signature import ensure_signature
from dspy.utils.dummies import DummyResponse


def test_prepare_request():
    backend = TextBackend(model="dummy")
    signature = ensure_signature("input -> output")
    example = Example(input="what day of the week is it?", demos=[])
    output = backend.prepare_request(signature, example, config={})

    assert output.get("messages", None) is not None
    assert output["messages"] == [
        {
            "role": "user",
            "content": textwrap.dedent("""\
            Given the fields `input`, produce the fields `output`.

            ---

            Follow the following format.

            Input: ${input}

            Output: ${output}

            ---

            Input: what day of the week is it?

            Output:"""),
        }
    ]


def test_process_response():
    backend = TextBackend(model="dummy")
    signature = ensure_signature("question -> answer")
    example = Example(question="what day of the week is it?", demos=[])
    input_kwargs = backend.prepare_request(signature, example, config={})

    response = DummyResponse(choices=[{"message": {"content": "Monday"}, "finish_reason": "done"}])

    output = backend.process_response(signature, example, response, input_kwargs)

    assert output.answer == "Monday"
