import textwrap

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def test_pot_code_generation():
    pot = ProgramOfThought(BasicQA)
    lm = DummyLM(
        [
            "Reason_A",
            "```python\nresult = 1+1\n```",
            "Reason_B",
            "2",
        ]
    )
    dspy.settings.configure(lm=lm)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
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


def test_pot_code_generation_with_error():
    pot = ProgramOfThought(BasicQA)
    lm = DummyLM(
        [
            "Reason_A",
            "```python\nresult = 1+0/0\n```",
            "Reason_B",  # Error: division by zero
            "```python\nresult = 1+1\n```",
            "Reason_C",
            "2",
        ]
    )
    dspy.settings.configure(lm=lm)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"

    # The first code example failed
    for message in lm.get_convo(2)[0]:
        print("----")
        print(message["content"])
        print("----")
    assert lm.get_convo(2)[0] == [
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

    # The second code example succeeded
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
