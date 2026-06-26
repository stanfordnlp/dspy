from pathlib import Path

from dotenv import load_dotenv

import dspy

load_dotenv()
exec_lm = dspy.LM("anthropic/claude-opus-4-7", max_tokens=500)
reflection_lm = dspy.LM("anthropic/claude-opus-4-7", max_tokens=4000)
dspy.configure(lm=exec_lm)


class MathWord(dspy.Signature):
    """Solve a problem that I won't tell you about"""

    idk: str = dspy.InputField()
    answer: int = dspy.OutputField()


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print(f"\n===== {label} =====")
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print("--- module_src (a normal dspy.Module subclass) ---")
    print(program.module_src)


def test_flex() -> None:
    # Reconfigure here (not just at import) so the test is order-independent: other
    # tests in the session reconfigure the global LM.
    dspy.configure(lm=exec_lm)
    program = dspy.VibNe(MathWord, persist_to=str(Path(__file__).parent / "math_flex_gen.py")).save()

    # Fresh baseline: a clean dspy.Module subclass that delegates to one dspy.RLM.
    assert program.module_src.lstrip().startswith("class ")
    assert "dspy.RLM(" in program.module_src
    _showcase(program, "baseline (un-optimized flex)")

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

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        try:
            return 1.0 if int(pred.answer) == int(gold.answer) else 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=15,
        reflection_minibatch_size=2,
        num_threads=1,
    ).compile(program, trainset=trainset, valset=valset)

    # GEPA rewrote the whole class (e.g. into a ChainOfThought + Python coercion).
    _showcase(optimized, "optimized by GEPA")
    print(f"GEPA changed the code: {optimized.module_src != program.module_src}")

    pred = optimized(problem="What is 2 plus 2?")
    assert int(pred.answer) == 4
