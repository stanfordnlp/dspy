import textwrap

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def test_pot_code_generation():
    lm = DummyLM(
        [
            "[[ ## reasoning ## ]]\nReason_A\n\n[[ ## generated_code ## ]]\n```python\nresult = 1+1\n```",
            "[[ ## reasoning ## ]]\nReason_B\n\n[[ ## answer ## ]]\n2",
        ]
    )
    dspy.settings.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"

    assert lm.get_convo(-1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
                Your input fields are:
                1. `question` (str)
                2. `final_generated_code` (str): python code that answers the question
                3. `code_output` (str): output of previously-generated python code

                Your output fields are:
                1. `reasoning` (str)
                2. `answer` (str): often between 1 and 5 words

                All interactions will be structured in the following way, with the appropriate values filled in.

                [[ ## question ## ]]
                {question}

                [[ ## final_generated_code ## ]]
                {final_generated_code}

                [[ ## code_output ## ]]
                {code_output}

                [[ ## reasoning ## ]]
                {reasoning}

                [[ ## answer ## ]]
                {answer}

                [[ ## completed ## ]]

                In adhering to this structure, your objective is: 
                        Given the final code `question`, `final_generated_code`, `code_output`, provide the final `answer`."""
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                [[ ## question ## ]]
                What is 1+1?

                [[ ## final_generated_code ## ]]
                result = 1+1

                [[ ## code_output ## ]]
                2

                Respond with the corresponding output fields, starting with the field `reasoning`, then `answer`, and then ending with the marker for `completed`."""
            ),
        },
    ]


def test_pot_code_generation_with_error():
    lm = DummyLM(
        [
            "[[ ## reasoning ## ]]\nReason_A\n\n[[ ## generated_code ## ]]\n```python\nresult = 1+0/0\n```",
            "[[ ## reasoning ## ]]\nReason_B\n\n[[ ## generated_code ## ]]\n```python\nresult = 1+1\n```",
            "[[ ## reasoning ## ]]\nReason_C\n\n[[ ## answer ## ]]\n2",
        ]
    )
    dspy.settings.configure(lm=lm)

    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"

    # The first code example failed
    assert lm.get_convo(1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
                Your input fields are:
                1. `question` (str)
                2. `previous_code` (str): previously-generated python code that errored
                3. `error` (str): error message from previously-generated python code

                Your output fields are:
                1. `reasoning` (str)
                2. `generated_code` (str): python code that answers the question

                All interactions will be structured in the following way, with the appropriate values filled in.

                [[ ## question ## ]]
                {question}

                [[ ## previous_code ## ]]
                {previous_code}

                [[ ## error ## ]]
                {error}

                [[ ## reasoning ## ]]
                {reasoning}

                [[ ## generated_code ## ]]
                {generated_code}

                [[ ## completed ## ]]

                In adhering to this structure, your objective is: 
                        You are given `question`, `previous_code`, `error` due to an error in previous code.
                        Your task is to correct the error and provide the new `generated_code`."""
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                [[ ## question ## ]]
                What is 1+1?

                [[ ## previous_code ## ]]
                result = 1+0/0

                [[ ## error ## ]]
                division by zero

                Respond with the corresponding output fields, starting with the field `reasoning`, then `generated_code`, and then ending with the marker for `completed`."""
            ),
        },
    ]
    assert lm.get_convo(-1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
                Your input fields are:
                1. `question` (str)
                2. `final_generated_code` (str): python code that answers the question
                3. `code_output` (str): output of previously-generated python code

                Your output fields are:
                1. `reasoning` (str)
                2. `answer` (str): often between 1 and 5 words

                All interactions will be structured in the following way, with the appropriate values filled in.

                [[ ## question ## ]]
                {question}

                [[ ## final_generated_code ## ]]
                {final_generated_code}

                [[ ## code_output ## ]]
                {code_output}

                [[ ## reasoning ## ]]
                {reasoning}

                [[ ## answer ## ]]
                {answer}

                [[ ## completed ## ]]

                In adhering to this structure, your objective is: 
                        Given the final code `question`, `final_generated_code`, `code_output`, provide the final `answer`."""
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                [[ ## question ## ]]
                What is 1+1?

                [[ ## final_generated_code ## ]]
                result = 1+1

                [[ ## code_output ## ]]
                2

                Respond with the corresponding output fields, starting with the field `reasoning`, then `answer`, and then ending with the marker for `completed`."""
            ),
        },
    ]
