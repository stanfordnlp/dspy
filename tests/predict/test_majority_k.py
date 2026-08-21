import pytest
from unittest.mock import MagicMock, call
from dspy.predict.helpers import majority_k
from dspy.primitives.prediction import Prediction, Completions

def test_majority_k_with_callable(monkeypatch):
    # Test with a callable predictor
    mock_predictor = MagicMock(return_value={"answer": "42"})
    mock_majority = MagicMock(return_value="mock_result")
    monkeypatch.setattr("dspy.predict.helpers.majority", mock_majority)
    
    wrapped = majority_k(mock_predictor, k=3)
    
    # Call the wrapped function
    result = wrapped(question="test", other_param=123)
    
    # Verify the predictor was called 3 times with the same args
    assert mock_predictor.call_count == 3
    mock_predictor.assert_has_calls([call(question="test", other_param=123)] * 3)
    
    # Verify majority was called once with a list of k predictions
    mock_majority.assert_called_once()
    predictions = mock_majority.call_args[0][0]
    assert len(predictions) == 3
    assert all(p == {"answer": "42"} for p in predictions)
    assert result == "mock_result"

def test_majority_k_with_existing_completions(monkeypatch):
    # Test with existing completions (non-callable)
    completions = [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
    mock_majority = MagicMock(return_value="mock_result")
    monkeypatch.setattr("dspy.predict.helpers.majority", mock_majority)
    
    result = majority_k(completions, field="answer")
    
    # Should directly call majority() with the completions
    mock_majority.assert_called_once_with(completions, field="answer")
    assert result == "mock_result"

def test_majority_k_with_kwargs(monkeypatch):
    # Test that kwargs are passed to majority()
    mock_majority = MagicMock(return_value="mock_result")
    monkeypatch.setattr("dspy.predict.helpers.majority", mock_majority)
    
    predictor = lambda x: {"answer": x}
    wrapped = majority_k(predictor, k=2, field="answer", normalize=lambda x: x)
    
    result = wrapped(x="test")
    kwargs = mock_majority.call_args[1]
    
    assert kwargs["field"] == "answer"
    assert callable(kwargs["normalize"])
    assert result == "mock_result"
