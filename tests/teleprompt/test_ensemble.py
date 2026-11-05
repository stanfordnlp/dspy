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


def test_ensemble_with_zero_size_uses_all_programs():
    """Test that a size of 0 keeps the existing semantics of using every program."""
    programs = [MockProgram(i) for i in range(5)]
    ensemble = Ensemble(size=0)
    ensembled_program = ensemble.compile(programs)

    outputs = ensembled_program()
    assert len(outputs) == 5, "Ensemble with size=0 did not use every program"


@pytest.mark.parametrize("size", [-1, -5])
def test_ensemble_rejects_negative_size_at_compile_time(size):
    """Test that a negative size is rejected by compile() instead of by random.sample()."""
    programs = [MockProgram(i) for i in range(5)]
    with pytest.raises(ValueError, match="Ensemble size must be non-negative"):
        Ensemble(size=size).compile(programs)


def test_ensemble_rejects_size_larger_than_program_count_at_compile_time():
    """Test that a size larger than the program count is rejected by compile()."""
    programs = [MockProgram(i) for i in range(2)]
    with pytest.raises(ValueError, match="exceeds the number of programs"):
        Ensemble(size=3).compile(programs)


def test_ensemble_deterministic_behavior():
    """Verify that the Ensemble class raises an assertion for deterministic behavior."""
    with pytest.raises(
        AssertionError,
        match="TODO: Implement example hashing for deterministic ensemble.",
    ):
        Ensemble(deterministic=True)
