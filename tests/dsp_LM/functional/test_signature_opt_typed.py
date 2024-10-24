import json
from typing import Generic, TypeVar

import pydantic
from pydantic_core import to_jsonable_python

import dspy
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.functional import TypedPredictor
from dspy.teleprompt.signature_opt_typed import make_info, optimize_signature
from dspy.utils import DSPDummyLM

hotpotqa = [
    ex.with_inputs("question")
    for ex in [
        dspy.Example(
            question="At My Window was released by which American singer-songwriter?",
            answer="John Townes Van Zandt",
        ),
        dspy.Example(
            question="which  American actor was Candace Kita  guest starred with ",
            answer="Bill Murray",
        ),
        dspy.Example(
            question="Which of these publications was most recently published, Who Put the Bomp or Self?",
            answer="Self",
        ),
        dspy.Example(
            question="The Victorians - Their Story In Pictures is a documentary series written by an author born in what year?",
            answer="1950",
        ),
        dspy.Example(
            question="Which magazine has published articles by Scott Shaw, Tae Kwon Do Times or Southwest Art?",
            answer="Tae Kwon Do Times",
        ),
        dspy.Example(
            question="In what year was the club founded that played Manchester City in the 1972 FA Charity Shield",
            answer="1874",
        ),
        dspy.Example(
            question="Which is taller, the Empire State Building or the Bank of America Tower?",
            answer="The Empire State Building",
        ),
        dspy.Example(
            question='Which American actress who made their film debut in the 1995 teen drama "Kids" was the co-founder of Voto Latino?',
            answer="Rosario Dawson",
        ),
        dspy.Example(
            question="Tombstone stared an actor born May 17, 1955 known as who?",
            answer="Bill Paxton",
        ),
        dspy.Example(
            question="What is the code name for the German offensive that started this Second World War engagement on the Eastern Front (a few hundred kilometers from Moscow) between Soviet and German forces, which included 102nd Infantry Division?",
            answer="Operation Citadel",
        ),
        dspy.Example(
            question='Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?',
            answer="Kerry Condon",
        ),
        dspy.Example(
            question="Which company distributed this 1977 American animated film produced by Walt Disney Productions for which Sherman Brothers wrote songs?",
            answer="Buena Vista Distribution",
        ),
        dspy.Example(
            question="Samantha Cristoforetti and Mark Shuttleworth are both best known for being first in their field to go where? ",
            answer="space",
        ),
        dspy.Example(
            question="Having the combination of excellent foot speed and bat speed helped Eric Davis, create what kind of outfield for the Los Angeles Dodgers? ",
            answer="Outfield of Dreams",
        ),
        dspy.Example(
            question="Which Pakistani cricket umpire who won 3 consecutive ICC umpire of the year awards in 2009, 2010, and 2011 will be in the ICC World Twenty20?",
            answer="Aleem Sarwar Dar",
        ),
        dspy.Example(
            question="The Organisation that allows a community to influence their operation or use and to enjoy the benefits arisingwas founded in what year?",
            answer="2010",
        ),
        dspy.Example(
            question='"Everything Has Changed" is a song from an album released under which record label ?',
            answer="Big Machine Records",
        ),
        dspy.Example(
            question="Who is older, Aleksandr Danilovich Aleksandrov or Anatoly Fomenko?",
            answer="Aleksandr Danilovich Aleksandrov",
        ),
        dspy.Example(
            question="On the coast of what ocean is the birthplace of Diogal Sakho?",
            answer="Atlantic",
        ),
        dspy.Example(
            question="This American guitarist best known for her work with the Iron Maidens is an ancestor of a composer who was known as what?",
            answer="The Waltz King",
        ),
    ]
]


def test_opt():
    class BasicQA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    qa_model = DSPDummyLM([])
    prompt_model = DSPDummyLM(
        [
            # Seed prompts
            "some thoughts",
            '[{"instructions": "I", "question_desc": "$q", "question_prefix": "Q:", "answer_desc": "$a", "answer_prefix": "A:"}]',
        ]
    )
    dspy.settings.configure(lm=qa_model)

    result = optimize_signature(
        student=TypedPredictor(BasicQA),
        evaluator=Evaluate(devset=hotpotqa, metric=answer_exact_match, num_threads=1),
        initial_prompts=1,
        n_iterations=2,
        verbose=True,
        prompt_model=prompt_model,
        strategy="last",
    )

    # Since we are requesting the last signature, it doesn't matter that our qa_model is
    # bad, and gets 0 score. We should still get the last signature.
    class ExpectedSignature(dspy.Signature):
        "I"

        question: str = dspy.InputField(desc="$q", prefix="Q:")
        answer: str = dspy.OutputField(desc="$a", prefix="A:")

    assert result.program.signature.equals(ExpectedSignature)

    assert result.scores == [0, 0]


def test_opt_composed():
    class MyModule(dspy.Module):
        def __init__(self):
            self.p1 = TypedPredictor("question:str -> considerations:list[str]", max_retries=1)
            self.p2 = TypedPredictor("considerations:list[str] -> answer:str", max_retries=1)

        def forward(self, question):
            considerations = self.p1(question=question).considerations
            return self.p2(considerations=considerations)

    class ExpectedSignature1(dspy.Signature):
        "I1"

        question: str = dspy.InputField(desc="$q", prefix="Q:")
        considerations: list[str] = dspy.OutputField(desc="$c", prefix="C:")

    info1 = make_info(ExpectedSignature1)

    class ExpectedSignature2(dspy.Signature):
        "I2"

        considerations: list[str] = dspy.InputField(desc="$c", prefix="C:")
        answer: str = dspy.OutputField(desc="$a", prefix="A:")

    info2 = make_info(ExpectedSignature2)

    qa_model = DSPDummyLM([])
    prompt_model = DSPDummyLM(
        [
            "some thoughts",
            json.dumps([to_jsonable_python(info1)]),
            "some thoughts",
            json.dumps([to_jsonable_python(info2)]),
        ]
    )
    dspy.settings.configure(lm=qa_model)

    result = optimize_signature(
        student=MyModule(),
        evaluator=lambda x: 0,  # We don't care about the evaluator here
        initial_prompts=1,
        n_iterations=2,
        verbose=True,
        prompt_model=prompt_model,
        strategy="last",
    )

    assert result.program.p1.signature.equals(ExpectedSignature1)
    assert result.program.p2.signature.equals(ExpectedSignature2)
