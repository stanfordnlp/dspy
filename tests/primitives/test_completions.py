import pytest

from dspy.primitives.prediction import Completions, Prediction


def test_completions_indexing():
    c = Completions([{"text": "one", "score": 1.0}, {"text": "two", "score": 2.0}, {"text": "three", "score": 3.0}])
    assert len(c) == 3

    # Positive indexing
    assert c[0] == Prediction(text="one", score=1.0)
    assert c[1] == Prediction(text="two", score=2.0)
    assert c[2] == Prediction(text="three", score=3.0)

    # Negative indexing
    assert c[-1] == Prediction(text="three", score=3.0)
    assert c[-2] == Prediction(text="two", score=2.0)
    assert c[-3] == Prediction(text="one", score=1.0)

    # Out of range indexing raises IndexError
    with pytest.raises(IndexError, match="Index out of range"):
        _ = c[3]

    with pytest.raises(IndexError, match="Index out of range"):
        _ = c[-4]


def test_completions_slicing():
    c = Completions(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}],
        signature="test_signature",
    )

    # Slice returns a new Completions instance
    sub = c[1:3]
    assert isinstance(sub, Completions)
    assert len(sub) == 2
    assert sub.signature == "test_signature"
    assert sub[0] == Prediction(a=2, b="y")
    assert sub[-1] == Prediction(a=3, b="z")
    assert sub["a"] == [2, 3]

    # Prefix slice
    prefix = c[:2]
    assert isinstance(prefix, Completions)
    assert len(prefix) == 2
    assert prefix[0] == Prediction(a=1, b="x")
    assert prefix[1] == Prediction(a=2, b="y")

    # Negative slice
    tail = c[-2:]
    assert isinstance(tail, Completions)
    assert len(tail) == 2
    assert tail[0] == Prediction(a=2, b="y")
    assert tail[1] == Prediction(a=3, b="z")

    # Step slice (reversed)
    rev = c[::-1]
    assert isinstance(rev, Completions)
    assert len(rev) == 3
    assert rev[0] == Prediction(a=3, b="z")
    assert rev[1] == Prediction(a=2, b="y")
    assert rev[2] == Prediction(a=1, b="x")


def test_completions_equality():
    c1 = Completions([{"a": 1}, {"a": 2}], signature="sig")
    c2 = Completions([{"a": 1}, {"a": 2}], signature="sig")
    c3 = Completions([{"a": 1}, {"a": 2}], signature="other_sig")
    c4 = Completions([{"a": 1}], signature="sig")

    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert c1 != "not_completions"
    assert c1[0:1] == c4


def test_completions_equality_with_signature():
    import dspy

    sig1 = dspy.Signature("question -> answer")
    sig2 = dspy.Signature("question -> answer")
    sig3 = dspy.Signature("context, question -> answer")

    c1 = Completions([{"answer": "yes"}], signature=sig1)
    c2 = Completions([{"answer": "yes"}], signature=sig2)
    c3 = Completions([{"answer": "yes"}], signature=sig3)

    assert c1 == c2
    assert c1 != c3
    assert c1 != Completions([{"answer": "yes"}], signature=None)


def test_prediction_from_completions_indexing():
    pred = Prediction.from_completions([{"a": 1}, {"a": 2}, {"a": 3}])
    assert pred.completions is not None
    assert pred.completions[-1] == Prediction(a=3)
    assert pred.completions[-2] == Prediction(a=2)
    assert pred.completions[:2] == Completions([{"a": 1}, {"a": 2}])
