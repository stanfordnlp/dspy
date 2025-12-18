import pytest

import dspy


# Fixtures
@pytest.fixture
def mock_lm():
    """Mock LM that returns dspy.Predictable responses"""
    lm = dspy.LM("openai/gpt-4o-mini")
    # Configure DSPy with mock LM
    dspy.settings.configure(lm=lm)
    return lm


@pytest.fixture
def qa_signature():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
    return QA


# Basic functionality tests
def test_stateful_with_dspy_predict(mock_lm, qa_signature):
    """Test Stateful wrapper with basic dspy.Predict"""
    predictor = dspy.Predict(qa_signature)
    stateful = dspy.Stateful(predictor)

    # Verify signature was modified to include history
    for pred in stateful.module.predictors():
        assert "history" in pred.signature.input_fields
        assert pred.signature.input_fields["history"].annotation == dspy.History


def test_stateful_with_chain_of_thought(mock_lm, qa_signature):
    """Test Stateful wrapper with ChainOfThought"""
    cot = dspy.ChainOfThought(qa_signature)
    stateful = dspy.Stateful(cot)

    # Verify signature modification
    for pred in stateful.module.predictors():
        assert "history" in pred.signature.input_fields
        # CoT should still have reasoning field
        assert "reasoning" in pred.signature.output_fields


def test_stateful_with_react(mock_lm):
    """Test Stateful wrapper with ReAct"""
    class Task(dspy.Signature):
        task: str = dspy.InputField()
        result: str = dspy.OutputField()

    react = dspy.ReAct(Task, tools=[print])
    stateful = dspy.Stateful(react)

    # Verify all dspy.Predictors have history
    predictors = list(stateful.module.predictors())
    assert len(predictors) > 0
    for pred in predictors:
        assert "history" in pred.signature.input_fields


# History accumulation tests
def test_history_accumulation(mock_lm, qa_signature):
    """Test that history accumulates over multiple calls"""
    predictors = dspy.Predict(qa_signature)
    stateful = dspy.Stateful(predictors)

    # Initial state - empty history
    assert len(stateful._history.messages) == 0

    # Mock forward to return dspy.Predictable results
    def mock_forward(**kwargs):
        return dspy.Prediction(answer=f"Answer to: {kwargs['question']}")

    stateful.module.forward = mock_forward

    # First call
    stateful(question="What is Python?")
    assert len(stateful._history.messages) == 1
    assert stateful._history.messages[0]["question"] == "What is Python?"
    assert "answer" in stateful._history.messages[0]

    # Second call
    stateful(question="Is it popular?")
    assert len(stateful._history.messages) == 2
    assert stateful._history.messages[1]["question"] == "Is it popular?"

    # Verify history is passed to dspy.Predictor
    original_forward = stateful.module.forward
    def verify_history(**kwargs):
        assert "history" in kwargs
        assert len(kwargs["history"].messages) == 2
        return original_forward(**kwargs)

    stateful.module.forward = verify_history
    stateful(question="Tell me more")


def test_reset_history(mock_lm, qa_signature):
    """Test history reset functionality"""
    predictors = dspy.Predict(qa_signature)
    stateful = dspy.Stateful(predictors)

    # Mock forward
    stateful.module.forward = lambda **kw: dspy.Prediction(answer="test")

    # Add some history
    stateful(question="Q1")
    stateful(question="Q2")
    assert len(stateful._history.messages) == 2

    # Reset
    stateful.reset_history()
    assert len(stateful._history.messages) == 0

    # New history should start fresh
    stateful(question="Q3")
    assert len(stateful._history.messages) == 1


# Custom module tests
class CustomModule(dspy.Module):
    """Custom module for testing"""
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
        self.summarizer = dspy.Predict("text -> summary")

    def forward(self, question):
        answer = self.qa(question=question)
        summary = self.summarizer(text=answer.answer)
        return dspy.Prediction(
            answer=answer.answer,
            summary=summary.summary
        )


def test_stateful_with_custom_module(mock_lm):
    """Test Stateful with custom module containing multiple dspy.Predictors"""
    module = CustomModule()
    stateful = dspy.Stateful(module)

    # Verify all dspy.Predictors got history
    predictors = list(stateful.predictors())
    assert len(predictors) == 2
    for pred in predictors:
        assert "history" in pred.signature.input_fields

    # Mock to test
    def mock_forward(**kwargs):
        return dspy.Prediction(answer="ans", summary="sum")

    stateful.module.forward = mock_forward

    stateful(question="test")
    assert len(stateful._history.messages) == 1
    assert stateful._history.messages[0]["answer"] == "ans"
    assert stateful._history.messages[0]["summary"] == "sum"


# Nested module tests
class NestedModule(dspy.Module):
    """Module with nested modules"""
    def __init__(self):
        super().__init__()
        self.sub_module = CustomModule()
        self.final = dspy.Predict("input -> output")

    def forward(self, question):
        sub_result = self.sub_module(question=question)
        final = self.final(input=sub_result.summary)
        return final


def test_stateful_with_nested_modules(mock_lm):
    """Test Stateful with nested module structures"""
    module = NestedModule()
    stateful = dspy.Stateful(module)

    # All dspy.Predictors should have history, even nested ones
    predictors = list(stateful.module.predictors())
    assert len(predictors) == 3  # 2 from CustomModule + 1 from NestedModule
    for pred in predictors:
        assert "history" in pred.signature.input_fields


# Edge case tests
def test_stateful_preserves_original_module(mock_lm, qa_signature):
    """Test that original module is not modified"""
    original = dspy.Predict(qa_signature)

    # Check original state
    assert "history" not in original.signature.input_fields

    # Create stateful version
    stateful = dspy.Stateful(original)

    # Original should be unchanged
    assert "history" not in original.signature.input_fields

    # Stateful version should have history
    for pred in stateful.module.predictors():
        assert "history" in pred.signature.input_fields


def test_stateful_with_dict_return(mock_lm):
    """Test handling of dict returns instead of dspy.Prediction"""
    predictor = dspy.Predict("question -> answer")
    stateful = dspy.Stateful(predictor)

    # Mock to return dict
    stateful.module.forward = lambda **kw: {"answer": "test", "extra": "data"}

    stateful(question="test")
    assert len(stateful._history.messages) == 1
    assert stateful._history.messages[0]["answer"] == "test"
    assert stateful._history.messages[0]["extra"] == "data"


def test_stateful_with_simple_return(mock_lm):
    """Test handling of simple value returns"""
    predictor = dspy.Predict("question -> answer")
    stateful = dspy.Stateful(predictor)

    # Mock to return simple string
    stateful.module.forward = lambda **kw: "simple answer"

    stateful(question="test")
    assert len(stateful._history.messages) == 1
    assert stateful._history.messages[0]["output"] == "simple answer"


def test_multiple_stateful_instances(mock_lm, qa_signature):
    """Test that multiple Stateful instances maintain separate histories"""
    predictor = dspy.Predict(qa_signature)

    stateful1 = dspy.Stateful(predictor)
    stateful2 = dspy.Stateful(predictor)

    # Mock forwards
    stateful1.module.forward = lambda **kw: dspy.Prediction(answer="s1")
    stateful2.module.forward = lambda **kw: dspy.Prediction(answer="s2")

    # Use first instance
    stateful1(question="Q1")
    assert len(stateful1._history.messages) == 1
    assert len(stateful2._history.messages) == 0

    # Use second instance
    stateful2(question="Q2")
    assert len(stateful1._history.messages) == 1
    assert len(stateful2._history.messages) == 1

    # Histories should be different
    assert stateful1._history.messages[0]["answer"] == "s1"
    assert stateful2._history.messages[0]["answer"] == "s2"
