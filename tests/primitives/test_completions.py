from dspy.primitives.prediction import Completions


def test_empty_completions_have_zero_length():
    assert len(Completions({})) == 0
