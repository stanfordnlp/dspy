import textwrap

import dspy
from dspy import ChainOfThought
from dspy.utils import DummyLM


def test_initialization_with_string_signature():
    lm = DummyLM(["[[ ## reasoning ## ]]\nfind the number after 1\n\n[[ ## answer ## ]]\n2"])
    dspy.settings.configure(lm=lm)
    predict = ChainOfThought("question -> answer")
    assert list(predict.extended_signature.output_fields.keys()) == [
        "reasoning",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"

    for message in lm.get_convo(-1)[0]:
        print("----")
        print(textwrap.indent(message["content"], "                "))
        print("----")
    assert lm.get_convo(-1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
                Your input fields are:
                1. `question` (str)

                Your output fields are:
                1. `reasoning` (str)
                2. `answer` (str)

                All interactions will be structured in the following way, with the appropriate values filled in.

                [[ ## question ## ]]
                {question}

                [[ ## reasoning ## ]]
                {reasoning}

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

                Respond with the corresponding output fields, starting with the field `reasoning`, then `answer`, and then ending with the marker for `completed`."""
            ),
        },
    ]
