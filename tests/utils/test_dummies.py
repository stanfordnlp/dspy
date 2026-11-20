import random

from dspy.utils import DummyVectorizer


def test_dummy_vectorizer_leaves_the_global_random_stream_alone():
    """Building a test vectorizer must not reseed the process-wide `random` module."""
    state = random.getstate()
    try:
        random.seed(7)
        expected = [random.random() for _ in range(3)]

        random.seed(7)
        DummyVectorizer(max_length=8)

        assert [random.random() for _ in range(3)] == expected
    finally:
        random.setstate(state)


def test_dummy_vectorizer_coefficients_stay_reproducible():
    """The n-gram hash weights are unchanged: the seed moved, the coefficients did not."""
    assert DummyVectorizer(max_length=8).coeffs == DummyVectorizer(max_length=8).coeffs
    assert DummyVectorizer().coeffs == [56224986, 287416173]
