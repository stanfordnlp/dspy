"""
Tests for native async support (aforward/acall) in DSPy modules.

This module tests the async functionality added to:
- MultiChainComparison
- BestOfN
- ProgramOfThought
- CodeAct
- Refine
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
from dspy.predict.best_of_n import BestOfN
from dspy.predict.multi_chain_comparison import MultiChainComparison
from dspy.predict.predict import Prediction
from dspy.primitives.module import Module


class MockLM:
    """Mock LM for testing."""

    def __init__(self):
        self.kwargs = {"rollout_id": 0}

    def copy(self, **kwargs):
        new_lm = MockLM()
        new_lm.kwargs = {**self.kwargs, **kwargs}
        return new_lm


class MockModule(Module):
    """Mock module for testing async functionality."""

    def __init__(self, return_value=None):
        super().__init__()
        self.return_value = return_value or Prediction(answer="test")
        self.lm = MockLM()

    def forward(self, **kwargs):
        return self.return_value

    async def aforward(self, **kwargs):
        return self.return_value

    def get_lm(self):
        return self.lm

    def set_lm(self, lm):
        self.lm = lm

    def deepcopy(self):
        return MockModule(self.return_value)

    def named_predictors(self):
        return []


@pytest.mark.asyncio
async def test_multi_chain_comparison_aforward():
    """Test MultiChainComparison.aforward() works correctly."""
    # Create a mock Predict that has acall
    mock_predict = MagicMock()
    mock_predict.acall = AsyncMock(return_value=Prediction(rationale="test", answer="42"))

    mcc = MultiChainComparison("question -> answer", M=2)
    mcc.predict = mock_predict

    completions = [
        {"rationale": "thinking step 1", "answer": "41"},
        {"rationale": "thinking step 2", "answer": "42"},
    ]

    result = await mcc.aforward(completions, question="What is the answer?")

    # Verify acall was called
    mock_predict.acall.assert_called_once()
    assert result.answer == "42"


@pytest.mark.asyncio
async def test_best_of_n_aforward():
    """Test BestOfN.aforward() works correctly."""
    mock_module = MockModule(Prediction(answer="correct"))

    def reward_fn(args, pred):
        return 1.0 if pred.answer == "correct" else 0.0

    bon = BestOfN(
        module=mock_module,
        N=3,
        reward_fn=reward_fn,
        threshold=0.5,
    )

    with patch.object(dspy, 'settings') as mock_settings:
        mock_settings.lm = MockLM()
        mock_settings.trace = []
        mock_settings.context = MagicMock()
        mock_settings.context.return_value.__enter__ = MagicMock()
        mock_settings.context.return_value.__exit__ = MagicMock()

        result = await bon.aforward(question="test")

        assert result is not None
        assert result.answer == "correct"


@pytest.mark.asyncio
async def test_module_acall_exists():
    """Test that Module.acall() method exists and is callable."""
    module = MockModule()

    # acall should exist on all Module subclasses
    assert hasattr(module, 'acall')
    assert callable(module.acall)

    # Test it can be awaited
    result = await module.acall(test="input")
    assert result is not None


@pytest.mark.asyncio
async def test_predict_acall_exists():
    """Test that Predict.acall() method exists."""
    from dspy.predict.predict import Predict

    predict = Predict("question -> answer")

    # acall should exist
    assert hasattr(predict, 'acall')
    assert callable(predict.acall)


@pytest.mark.asyncio
async def test_chain_of_thought_aforward_exists():
    """Test that ChainOfThought.aforward() method exists."""
    from dspy.predict.chain_of_thought import ChainOfThought

    cot = ChainOfThought("question -> answer")

    # aforward should exist
    assert hasattr(cot, 'aforward')
    assert callable(cot.aforward)


@pytest.mark.asyncio
async def test_react_aforward_exists():
    """Test that ReAct.aforward() method exists."""
    from dspy.predict.react import ReAct

    def dummy_tool():
        return "result"

    react = ReAct("question -> answer", tools=[dummy_tool])

    # aforward should exist
    assert hasattr(react, 'aforward')
    assert callable(react.aforward)


@pytest.mark.asyncio
async def test_program_of_thought_aforward_exists():
    """Test that ProgramOfThought.aforward() method exists."""
    from dspy.predict.program_of_thought import ProgramOfThought

    # Note: ProgramOfThought requires deno to be installed
    # This test only checks the method exists, not full functionality
    assert hasattr(ProgramOfThought, 'aforward')


@pytest.mark.asyncio
async def test_code_act_aforward_exists():
    """Test that CodeAct.aforward() method exists."""
    from dspy.predict.code_act import CodeAct

    # This test only checks the method exists
    assert hasattr(CodeAct, 'aforward')


@pytest.mark.asyncio
async def test_refine_aforward_exists():
    """Test that Refine.aforward() method exists."""
    from dspy.predict.refine import Refine

    # This test only checks the method exists
    assert hasattr(Refine, 'aforward')


@pytest.mark.asyncio
async def test_best_of_n_aforward_exists():
    """Test that BestOfN.aforward() method exists."""
    from dspy.predict.best_of_n import BestOfN

    assert hasattr(BestOfN, 'aforward')
