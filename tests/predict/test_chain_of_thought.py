import textwrap

import dspy
from dspy import ChainOfThought
from dspy.utils import DummyLM


def test_initialization_with_string_signature():
    lm = DummyLM(["find the number after 1", "2"])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThought("question -> answer")
    assert list(predict.extended_signature.output_fields.keys()) == [
        "rationale",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"

    for message in lm.get_convo(-1)[0]:
        print("----")
        print(message["content"])
        print("----")
    assert lm.get_convo(-1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
            """
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                """
            ),
        },
    ]
