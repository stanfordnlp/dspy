import pydantic
from typing import Annotated

import dspy
from dspy.functional import predictor, cot
from dspy.utils.dummies import DummyLM

def test_simple():

    @predictor
    def hard_question(topic: str) -> str:
        """Think of a hard factual question about a topic."""

    expected = "What is the speed of light?"
    lm = DummyLM([expected])
    dspy.settings.configure(lm=lm)

    question = hard_question(topic="Physics")
    lm.inspect_history(n=2)

    assert question == expected

def test_simple_type():
    class Question(pydantic.BaseModel):
        value: str

    @predictor
    def hard_question(topic: str) -> Question:
        """Think of a hard factual question about a topic."""

    expected = "What is the speed of light?"
    lm = DummyLM([f'{{"value": "{expected}"}}'])
    dspy.settings.configure(lm=lm)

    question = hard_question(topic="Physics")
    lm.inspect_history(n=2)

    assert question.value == expected


def test_simple_class():
    class Answer(pydantic.BaseModel):
        value: float
        certainty: float
        comments: list[str] = pydantic.Field(
            description="At least two comments about the answer"
        )

    class QA(dspy.Module):
        @predictor
        def hard_question(self, topic: str) -> str:
            """Think of a hard factual question about a topic. It should be answerable with a number."""

        @cot
        def answer(self, question: Annotated[str, "Question to answer"]) -> Answer:
            pass

        def forward(self, **kwargs):
            question = self.hard_question(**kwargs)
            return (question, self.answer(question=question))

    expected = Answer(value=3e8, certainty=0.9, comments=["It is the speed of light", "It is a constant"])

    lm = DummyLM([
        "What is the speed of light?",
        "Some bad reasoning, 3e8 m/s.",
        "3e8",  # Bad answer 1
        "Some good reasoning...",
        expected.model_dump_json(),  # Good answer
    ])
    dspy.settings.configure(lm=lm)

    qa = QA()
    question, answer = qa(topic="Physics")
    lm.inspect_history(n=6)

    assert question == "What is the speed of light?"
    assert answer == expected


def test_named_params():
    class QA(dspy.Module):
        @predictor
        def hard_question(self, topic: str) -> str:
            """Think of a hard factual question about a topic. It should be answerable with a number."""

        @cot
        def answer(self, question: str) -> str:
            pass
    
    qa = QA()
    print(dir(qa))
    print(qa.__dict__)
    named_predictors = list(qa.named_predictors())
    assert len(named_predictors) == 2
    names, _ = zip(*qa.named_predictors())
    assert names == ["hard_question", "answer"]