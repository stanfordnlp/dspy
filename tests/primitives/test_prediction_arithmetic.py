import pytest

from dspy.primitives.prediction import Prediction


def test_prediction_subtraction():
    p1 = Prediction(score=10.0)
    p2 = Prediction(score=3.0)

    # Prediction - Prediction
    assert p1 - p2 == 7.0
    assert p2 - p1 == -7.0

    # Prediction - scalar
    assert p1 - 2.5 == 7.5
    assert p1 - 4 == 6.0

    # scalar - Prediction (rsub)
    assert 15.0 - p1 == 5.0
    assert 20 - p1 == 10.0


def test_prediction_multiplication():
    p1 = Prediction(score=4.0)
    p2 = Prediction(score=2.5)

    # Prediction * Prediction
    assert p1 * p2 == 10.0

    # Prediction * scalar
    assert p1 * 3 == 12.0
    assert p1 * 1.5 == 6.0

    # scalar * Prediction (rmul)
    assert 2 * p1 == 8.0
    assert 0.5 * p1 == 2.0


def test_prediction_unary_and_abs():
    p_pos = Prediction(score=5.0)
    p_neg = Prediction(score=-5.0)

    assert -p_pos == -5.0
    assert -p_neg == 5.0
    assert +p_pos == 5.0
    assert abs(p_neg) == 5.0
    assert abs(p_pos) == 5.0


def test_prediction_arithmetic_compound_expression():
    p1 = Prediction(score=2.0)
    p2 = Prediction(score=3.0)
    p3 = Prediction(score=1.0)

    result = (p1 + p2) * 2 - p3
    assert result == 9.0


def test_prediction_arithmetic_missing_score_raises_value_error():
    p_no_score = Prediction(answer="test")

    with pytest.raises(ValueError, match="does not have a 'score' field"):
        _ = p_no_score - 1.0

    with pytest.raises(ValueError, match="does not have a 'score' field"):
        _ = 1.0 - p_no_score

    with pytest.raises(ValueError, match="does not have a 'score' field"):
        _ = p_no_score * 2.0

    with pytest.raises(ValueError, match="does not have a 'score' field"):
        _ = 2.0 * p_no_score

    with pytest.raises(ValueError, match="does not have a 'score' field"):
        _ = -p_no_score

    with pytest.raises(ValueError, match="does not have a 'score' field"):
        _ = abs(p_no_score)


def test_prediction_arithmetic_invalid_type_raises_type_error():
    p = Prediction(score=1.0)

    with pytest.raises(TypeError, match="Unsupported type for subtraction"):
        _ = p - "invalid"

    with pytest.raises(TypeError, match="Unsupported type for subtraction"):
        _ = "invalid" - p

    with pytest.raises(TypeError, match="Unsupported type for multiplication"):
        _ = p * "invalid"

    with pytest.raises(TypeError, match="Unsupported type for multiplication"):
        _ = "invalid" * p
