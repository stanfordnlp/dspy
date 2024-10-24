import textwrap

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DSPDummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def test_pot_code_generation():
    pot = ProgramOfThought(BasicQA)
    lm = DSPDummyLM(
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
    assert lm.get_convo(index=-1) == textwrap.dedent(
        """\
        Given the final code `question`, `final_generated_code`, `code_output`, provide the final `answer`.

        ---

        Follow the following format.

        Question: ${question}

        Code: python code that answers the question

        Code Output: output of previously-generated python code

        Reasoning: Let's think step by step in order to ${produce the answer}. We ...

        Answer: often between 1 and 5 words

        ---

        Question: What is 1+1?

        Code: result = 1+1

        Code Output: 2

        Reasoning: Let's think step by step in order to Reason_B

        Answer: 2"""
    )


def test_pot_code_generation_with_error():
    pot = ProgramOfThought(BasicQA)
    lm = DSPDummyLM(
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
    assert lm.get_convo(index=2) == textwrap.dedent(
        """\
        You are given `question`, `previous_code`, `error` due to an error in previous code.
        Your task is to correct the error and provide the new `generated_code`.

        ---

        Follow the following format.

        Question: ${question}

        Previous Code: previously-generated python code that errored

        Error: error message from previously-generated python code

        Reasoning: Let's think step by step in order to ${produce the generated_code}. We ...

        Code: python code that answers the question

        ---

        Question: What is 1+1?

        Previous Code: result = 1+0/0

        Error: division by zero

        Reasoning: Let's think step by step in order to Reason_B"""
    )

    # The second code example succeeded
    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the final code `question`, `final_generated_code`, `code_output`, provide the final `answer`.

        ---

        Follow the following format.

        Question: ${question}

        Code: python code that answers the question

        Code Output: output of previously-generated python code

        Reasoning: Let's think step by step in order to ${produce the answer}. We ...

        Answer: often between 1 and 5 words

        ---

        Question: What is 1+1?

        Code: result = 1+1

        Code Output: 2

        Reasoning: Let's think step by step in order to Reason_C

        Answer: 2"""
    )
