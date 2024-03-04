import json
import dspy
from dspy.evaluate import Evaluate
from dspy.functional import TypedPredictor
from dspy.teleprompt.signature_opt2 import (
    GenerateInstructionGivenAttempts,
    ScoredSignature,
    make_info,
    optimize_signature,
)
from dspy.utils import DummyLM

from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.functional import TypedPredictor


class BasicQA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


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


def test_signature_info():
    info = make_info(BasicQA)
    SignatureInfo = type(info)

    devset = [
        dspy.Example(
            instructions="Answer the following questions",
            question_desc="Some question to answer",
            question_prefix="Q: ",
            answer_desc="A short answer to the question",
            answer_prefix="A: ",
        ),
    ]

    lm = DummyLM(
        [
            json.dumps(dict(devset[0])),  # Proposed signature
        ]
    )
    dspy.settings.configure(lm=lm)

    generator = TypedPredictor(GenerateInstructionGivenAttempts[SignatureInfo])

    res = generator(attempted_signatures=[ScoredSignature[SignatureInfo](signature=info, score=50)])
    assert res.proposed_signature == SignatureInfo(**devset[0])

    # Test the "to_signature" method

    class OutputSignature(dspy.Signature):
        """Answer the following questions"""

        question: str = dspy.InputField(desc="Some question to answer", prefix="Q: ")
        answer: str = dspy.OutputField(desc="A short answer to the question", prefix="A: ")

    assert res.proposed_signature.to_signature().equals(OutputSignature)


def test_opt():
    qa_model = DummyLM([])
    prompt_model = DummyLM(
        [
            # Seed prompts
            "some thoughts",
            '{"value": [{"instructions": "I", "question_desc": "$q", "question_prefix": "Q:", "answer_desc": "$a", "answer_prefix": "A:"}]}',
        ]
    )
    dspy.settings.configure(lm=qa_model)

    program = optimize_signature(
        student=TypedPredictor(BasicQA),
        evaluator=Evaluate(devset=hotpotqa, metric=answer_exact_match, num_threads=1),
        initial_prompts=1,
        n_iterations=1,
        verbose=True,
        prompt_model=prompt_model,
    )
