import pytest

from dspy.primitives.prediction import Completions


def test_completions_support_negative_indexing():
    """Completions advertises the sequence protocol, so [-1] must be the last one."""
    completions = Completions([{"answer": "first"}, {"answer": "second"}])

    assert completions[-1].answer == "second"
    assert completions[-2].answer == "first"


def test_completions_still_reject_out_of_range_indexes():
    completions = Completions([{"answer": "first"}, {"answer": "second"}])

    with pytest.raises(IndexError):
        completions[2]

    with pytest.raises(IndexError):
        completions[-3]
