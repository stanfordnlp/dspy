import textwrap

import dspy
from dspy import ChainOfThoughtWithHint
from dspy.utils import DummyLM


def test_cot_with_no_hint():
    lm = DummyLM(["[[ ## rationale ## ]]\nfind the number after 1\n\n[[ ## answer ## ]]\n2"])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThoughtWithHint("question -> answer")
    # Check output fields have the right order
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"

    assert lm.get_convo(-1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
                Your input fields are:
                1. `question` (str)

                Your output fields are:
                1. `rationale` (str): ${produce the answer}. We ...
                2. `answer` (str)

                All interactions will be structured in the following way, with the appropriate values filled in.

                [[ ## question ## ]]
                {question}

                [[ ## rationale ## ]]
                {rationale}

                [[ ## answer ## ]]
                {answer}

                [[ ## completed ## ]]

                In adhering to this structure, your objective is: 
                        Given the fields `question`, produce the fields `answer`."""
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                [[ ## question ## ]]
                What is 1+1?

                Respond with the corresponding output fields, starting with the field `rationale`, then `answer`, and then ending with the marker for `completed`."""
            ),
        },
    ]


def test_cot_with_hint():
    lm = DummyLM(
        [
            "[[ ## rationale ## ]]\nfind the number after 1\n\n[[ ## hint ## ]]\nIs it helicopter?\n\n[[ ## answer ## ]]\n2"
        ]
    )
    dspy.settings.configure(lm=lm)
    predict = ChainOfThoughtWithHint("question -> answer")
    assert list(predict.extended_signature2.output_fields.keys()) == [
        "rationale",
        "hint",
        "answer",
    ]
    assert predict(question="What is 1+1?", hint="think small").answer == "2"

    assert lm.get_convo(-1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
                Your input fields are:
                1. `question` (str)

                Your output fields are:
                1. `rationale` (str): ${produce the answer}. We ...
                2. `hint` (str)
                3. `answer` (str)

                All interactions will be structured in the following way, with the appropriate values filled in.

                [[ ## question ## ]]
                {question}

                [[ ## rationale ## ]]
                {rationale}

                [[ ## hint ## ]]
                {hint}

                [[ ## answer ## ]]
                {answer}

                [[ ## completed ## ]]

                In adhering to this structure, your objective is: 
                        Given the fields `question`, produce the fields `answer`."""
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                [[ ## question ## ]]
                What is 1+1?

                Respond with the corresponding output fields, starting with the field `rationale`, then `hint`, then `answer`, and then ending with the marker for `completed`."""
            ),
        },
    ]
