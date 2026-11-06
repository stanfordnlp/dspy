import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import InferRules
from dspy.utils.dummies import DummyLM


def answer_metric(example, prediction, trace=None):
    return example.answer == prediction.answer


trainset = [Example(question=f"train {i}", answer="good").with_inputs("question") for i in range(4)]
valset = [Example(question=f"val {i}", answer="good").with_inputs("question") for i in range(4)]


def test_compile_returns_best_candidate_and_leaves_student_unchanged():
    # One bootstrap call, then per candidate one rules-induction call and one call per valset example.
    # Only the first candidate answers the valset correctly, so it is the one compile() should return.
    answers = [{"answer": "good"}]
    for i in range(3):
        answers.append({"reasoning": "r", "natural_language_rules": f"RULE-{i}"})
        answers += [{"answer": "good" if i == 0 else "bad"}] * len(valset)
    dspy.configure(lm=DummyLM(answers))

    student = Predict("question -> answer")
    original_instructions = student.signature.instructions

    optimizer = InferRules(
        metric=answer_metric, num_candidates=3, num_threads=1, max_bootstrapped_demos=1, max_labeled_demos=0
    )
    best = optimizer.compile(student, trainset=trainset, valset=valset)

    assert "RULE-0" in best.signature.instructions
    assert "RULE-1" not in best.signature.instructions
    assert "RULE-2" not in best.signature.instructions
    assert student.signature.instructions == original_instructions
