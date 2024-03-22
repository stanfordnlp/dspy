import textwrap
import dspy
from dspy import ChainOfThought
from dspy.utils import DummyLM, DummyLanguageModel
from dspy.backends import TemplateBackend


def test_initialization_with_string_signature():
    dspy.settings.configure(experimental=False)

    lm = DummyLM(["find the number after 1", "2"])
    with dspy.settings.context(lm=lm, backend=None):

        predict = ChainOfThought("question -> answer")
        assert list(predict.extended_signature.output_fields.keys()) == [
            "rationale",
            "answer",
        ]
        assert predict(question="What is 1+1?").answer == "2"

        print(lm.get_convo(-1))
        assert lm.get_convo(-1) == textwrap.dedent(
            """\
            Given the fields `question`, produce the fields `answer`.

            ---

            Follow the following format.

            Question: ${question}
            Reasoning: Let's think step by step in order to ${produce the answer}. We ...
            Answer: ${answer}

            ---

            Question: What is 1+1?
            Reasoning: Let's think step by step in order to find the number after 1
            Answer: 2"""
        )


def test_initialization_with_string_signature_experimental():
    lm = DummyLanguageModel(answers=[["find the number after 1\n\nAnswer: 2"]])
    backend = TemplateBackend(lm=lm)
    with dspy.settings.context(backend=backend, lm=None, cache=False):

        predict = ChainOfThought("question -> answer")
        assert list(predict.extended_signature.output_fields.keys()) == [
            "rationale",
            "answer",
        ]
        output = predict(question="What is 1+1?")
        assert output.answer == "2"
        assert output.rationale == "find the number after 1"
