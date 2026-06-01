from pathlib import Path

from dotenv import load_dotenv

import dspy
from dspy import FlexGEPA, flex

load_dotenv()
exec_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=500)
reflection_lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=4000)
dspy.configure(lm=exec_lm)

def test_flex() -> None:
    @flex(persist_to=str(Path(__file__).parent / "math_word.py"))
    class MathWord(dspy.Signature):
        """Solve a math word problem."""

        problem: str = dspy.InputField()
        answer: int = dspy.OutputField()

    program = MathWord()
    baseline = program(problem="Alice has 3 apples and gets 2 more. How many does she have?")

    print(f"Baseline answer is: '{baseline.answer}', correct answer is int 5.")

    def ex(p, a):
        return dspy.Example(problem=p, answer=a).with_inputs("problem")

    trainset = [
        ex("Alice has 3 apples and gets 2 more. How many does she have?", 5),
        ex("A train travels 60 miles in 2 hours. How many miles per hour?", 30),
        ex("Bob has 12 candies and gives 4 away. How many remain?", 8),
    ]
    valset = [
        ex("There are 7 birds; 3 fly away. How many remain?", 4),
        ex("A box has 6 pencils per row and 4 rows. How many pencils total?", 24),
    ]

    def metric(example, prediction, trace=None):
        try:
            return 1.0 if int(prediction.answer) == int(example.answer) else 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

    optimized = FlexGEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=15,
        reflection_minibatch_size=2,
        num_threads=1,
    ).compile(program, trainset=trainset, valset=valset)

    pred = optimized(problem="What is 2 plus 2?")
    assert int(pred.answer) == 4

