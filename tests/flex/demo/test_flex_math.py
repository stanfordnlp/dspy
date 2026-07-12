from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()
exec_lm = dspy.LM("anthropic/claude-opus-4-5", max_tokens=500)
reflection_lm = dspy.LM("anthropic/claude-opus-4-8", max_tokens=4000)
dspy.configure(lm=exec_lm)

LLM_CALL_PENALTY = 0.15


class MathWord(dspy.Signature):
    """Solve a math word problem."""

    problem: str = dspy.InputField()
    answer: int = dspy.OutputField()


def apply_operation(a: float, b: float, operation) -> float:
    """Apply a mathematical operation to two floats and return the result as a float."""
    return operation(a, b)


def test_flex(tmp_path) -> None:
    baseline_path = tmp_path / "flex_mathword.json"
    optimized_path = tmp_path / "flex_mathword_optimized.json"

    program = dspy.Flex(MathWord, tools=[apply_operation])
    program.save(baseline_path)

    reloaded_baseline = dspy.Flex(MathWord, tools=[apply_operation])
    reloaded_baseline.load(baseline_path)

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

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
        try:
            correct = int(pred.answer) == int(gold.answer)
        except (ValueError, TypeError, AttributeError):
            return ScoreWithFeedback(score=0.0, feedback="`answer` was missing or not an int. Return an int.")
        n_calls = len(trace) if trace else 0
        score = max(0.0, (1.0 if correct else 0.0) - LLM_CALL_PENALTY * n_calls)
        fb = (
            f"{'CORRECT' if correct else 'WRONG'}. You used {n_calls} LM call(s) "
            f"(cost {LLM_CALL_PENALTY * n_calls:.2f}). Prefer a plain code solution to using "
            f"an LLM when possible."
        )
        return ScoreWithFeedback(score=score, feedback=fb)

    dspy.configure(lm=exec_lm)
    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=15,
        reflection_minibatch_size=2,
        num_threads=1,
    ).compile(program, trainset=trainset, valset=valset)

    print(f"GEPA changed the code: {optimized.module_src != program.module_src}")

    optimized.save(optimized_path)
    reloaded_optimized = dspy.Flex(MathWord, tools=[apply_operation])
    reloaded_optimized.load(optimized_path)
    assert reloaded_optimized.module_src == optimized.module_src

    # Run an example on the reloaded optimized program.
    pred = reloaded_optimized(problem="What is 2 plus 2?")
    assert int(pred.answer) == 4
