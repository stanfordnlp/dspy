import pytest

import dspy
from dspy.teleprompt import Ensemble


class MockProgram(dspy.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, *args, **kwargs):
        return self.output


# Simple reduction function to test with
def mock_reduce_fn(outputs):
    return sum(outputs) / len(outputs)


def test_ensemble_without_reduction():
    """Test that Ensemble correctly combines outputs without applying a reduce_fn."""
    programs = [MockProgram(i) for i in range(5)]
    ensemble = Ensemble()
    ensembled_program = ensemble.compile(programs)

    outputs = ensembled_program()
    assert len(outputs) == 5, "Ensemble did not combine the correct number of outputs"


def test_ensemble_with_reduction():
    """Test that Ensemble correctly applies a reduce_fn to combine outputs."""
    programs = [MockProgram(i) for i in range(5)]
    ensemble = Ensemble(reduce_fn=mock_reduce_fn)
    ensembled_program = ensemble.compile(programs)

    output = ensembled_program()
    expected_output = sum(range(5)) / 5
    assert output == expected_output, "Ensemble did not correctly apply the reduce_fn"


def test_ensemble_with_size_limitation():
    """Test that specifying a size limits the number of programs used in the ensemble."""
    programs = [MockProgram(i) for i in range(10)]
    ensemble_size = 3
    ensemble = Ensemble(size=ensemble_size)
    ensembled_program = ensemble.compile(programs)

    outputs = ensembled_program()
    assert len(outputs) == ensemble_size, "Ensemble did not respect the specified size limitation"


def test_ensemble_deterministic_behavior():
    """Verify that the Ensemble class raises an assertion for deterministic behavior."""
    with pytest.raises(
        AssertionError,
        match="TODO: Implement example hashing for deterministic ensemble.",
    ):
        Ensemble(deterministic=True)
