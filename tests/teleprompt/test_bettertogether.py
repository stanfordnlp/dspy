"""BetterTogether optimizer tests.

Most of the code in this test file was LLM-generated but has been verified
to correctly test the BetterTogether optimizer functionality.
"""
from unittest.mock import Mock, patch

import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BetterTogether, BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.dummies import DummyLM


# Define a simple metric function for testing
def simple_metric(example, prediction, trace=None):
    return 1.0 if example.output == prediction.output else 0.0


examples = [
    Example(input="What is the oldest known human-made monument?", output="Göbekli Tepe in southeastern Turkiye, dating back to around 9600 BCE").with_inputs("input"),
    Example(input="Why can't fish fall in love?", output="Because love is in the air").with_inputs("input"),
    Example(input="What would bring world peace?", output="8 billion people meeting for a tea party in my backyard").with_inputs("input"),
]
trainset = examples[:2]
valset = [examples[2]]


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


# ============================================================================
# Reusable Mock Optimizers
# ============================================================================

class SimpleOptimizer(Teleprompter):
    """A simple optimizer that returns the student unchanged."""
    def compile(self, student, **kwargs):
        return student


class MarkedOptimizer(Teleprompter):
    """An optimizer that marks the program with a specific identifier."""
    def __init__(self, marker):
        self.marker = marker

    def compile(self, student, **kwargs):
        prog = SimpleModule("input -> output")
        prog.marker = self.marker
        return prog


class CapturingOptimizer(Teleprompter):
    """An optimizer that captures the kwargs it receives."""
    def __init__(self):
        self.received_kwargs = {}

    def compile(self, student, trainset=None, valset=None, teacher=None,
                num_trials=None, max_bootstrapped_demos=None, **kwargs):
        self.received_kwargs = {
            "trainset": trainset,
            "valset": valset,
            "teacher": teacher,
            "num_trials": num_trials,
            "max_bootstrapped_demos": max_bootstrapped_demos,
            **kwargs
        }
        return student


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def student_with_lm():
    """Create a student module with a DummyLM."""
    student = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)
    return student


@pytest.fixture
def mock_bt_dependencies():
    """Mock the common BetterTogether dependencies."""
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval, \
         patch("dspy.teleprompt.bettertogether.launch_lms") as mock_launch, \
         patch("dspy.teleprompt.bettertogether.kill_lms") as mock_kill:
        mock_eval.return_value = Mock(score=0.8)
        yield mock_eval, mock_launch, mock_kill


# ============================================================================
# Tests
# ============================================================================

def test_bettertogether_import():
    """Sanity check: Test that BetterTogether can be imported."""
    assert BetterTogether is not None, "Failed to import BetterTogether"


def test_bettertogether_initialization_default():
    """Test BetterTogether initialization with default optimizers."""
    optimizer = BetterTogether(metric=simple_metric)

    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert "p" in optimizer.optimizers, "Default 'p' optimizer not created"
    assert "w" in optimizer.optimizers, "Default 'w' optimizer not created"
    assert isinstance(optimizer.optimizers["p"], BootstrapFewShotWithRandomSearch), \
        "Default 'p' should be BootstrapFewShotWithRandomSearch"
    assert isinstance(optimizer.optimizers["w"], BootstrapFinetune), \
        "Default 'w' should be BootstrapFinetune"


def test_bettertogether_initialization_custom():
    """Test BetterTogether initialization with custom optimizers."""
    custom_p = BootstrapFewShotWithRandomSearch(metric=simple_metric)
    custom_w = BootstrapFinetune(metric=simple_metric)

    optimizer = BetterTogether(
        metric=simple_metric,
        p=custom_p,
        w=custom_w
    )

    assert optimizer.optimizers["p"] is custom_p, "Custom 'p' optimizer not set"
    assert optimizer.optimizers["w"] is custom_w, "Custom 'w' optimizer not set"


def test_bettertogether_initialization_invalid_optimizer():
    """Test that BetterTogether rejects non-Teleprompter optimizers."""
    try:
        optimizer = BetterTogether(
            metric=simple_metric,
            p="not_a_teleprompter"  # Invalid type
        )
        assert False, "Should have raised TypeError for invalid optimizer"
    except TypeError as e:
        assert "must be a Teleprompter" in str(e)


def test_strategy_validation():
    """Test strategy validation: valid, invalid, and empty strategies."""
    optimizer = BetterTogether(metric=simple_metric)

    # Valid strategies should parse without errors
    valid_strategies = ["p", "w", "p -> w", "w -> p", "p -> w -> p"]
    for strategy in valid_strategies:
        parsed = optimizer._prepare_strategy(strategy)
        assert parsed is not None, f"Failed to parse valid strategy: {strategy}"

    # Invalid strategies should raise ValueError
    with pytest.raises(ValueError, match="invalid optimizer keys"):
        optimizer._prepare_strategy("p -> x -> w")

    with pytest.raises(ValueError, match="cannot be empty"):
        optimizer._prepare_strategy("")


def test_compile_basic():
    """Test basic compilation with mocked optimizers."""
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule("input -> output")

    lm = DummyLM([{"output": "blue"}, {"output": "4"}])
    student.set_lm(lm)

    # Create a mock Teleprompter that returns the student
    class MockTeleprompter(Teleprompter):
        def __init__(self):
            self.compile_called = False

        def compile(self, student, **kwargs):
            self.compile_called = True
            return student

    mock_p = MockTeleprompter()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)

    # Mock evaluation to avoid actually running the metric
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.return_value = Mock(score=0.8)

        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                compiled = optimizer.compile(
                    student,
                    trainset=trainset,
                    valset=valset,
                    strategy="p"
                )

    assert compiled is not None, "Compilation returned None"
    assert hasattr(compiled, "candidate_programs"), "Missing candidate_programs attribute"
    assert hasattr(compiled, "flag_compilation_error_occurred"), "Missing flag_compilation_error_occurred attribute"
    assert mock_p.compile_called, "Mock optimizer compile was not called"


def test_trainset_validation():
    """Test that empty trainset is rejected."""
    optimizer = BetterTogether(metric=simple_metric)
    student = SimpleModule("input -> output")

    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    try:
        optimizer.compile(student, trainset=[], valset=valset)
        assert False, "Should have raised ValueError for empty trainset"
    except ValueError as e:
        assert "cannot be empty" in str(e).lower()


def test_valset_ratio_validation():
    """Test that invalid valset_ratio is rejected."""
    optimizer = BetterTogether(metric=simple_metric)
    student = SimpleModule("input -> output")

    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    # Test valset_ratio >= 1
    try:
        optimizer.compile(student, trainset=trainset, valset_ratio=1.0)
        assert False, "Should have raised ValueError for valset_ratio >= 1"
    except ValueError as e:
        assert "must be in range [0, 1)" in str(e)

    # Test valset_ratio < 0
    try:
        optimizer.compile(student, trainset=trainset, valset_ratio=-0.1)
        assert False, "Should have raised ValueError for valset_ratio < 0"
    except ValueError as e:
        assert "must be in range [0, 1)" in str(e)


def test_optimizer_compile_args_validation():
    """Test that optimizer_compile_args is validated correctly."""
    optimizer = BetterTogether(metric=simple_metric)

    # Test invalid optimizer key
    try:
        optimizer._prepare_optimizer_compile_args(
            {"invalid_key": {"num_trials": 10}},
            teacher=None
        )
        assert False, "Should have raised ValueError for invalid optimizer key"
    except ValueError as e:
        assert "invalid optimizer key" in str(e).lower()


def test_student_in_optimizer_compile_args():
    """Test that 'student' in optimizer_compile_args is rejected."""
    optimizer = BetterTogether(metric=simple_metric)

    try:
        optimizer._validate_compile_args(
            optimizer.optimizers["p"],
            "p",
            {"student": SimpleModule("input -> output")}
        )
        assert False, "Should have raised ValueError for 'student' in compile_args"
    except ValueError as e:
        assert "student" in str(e).lower()
        assert "not allowed" in str(e).lower()


def test_compile_args_passed_to_optimizer(student_with_lm, mock_bt_dependencies):
    """Test that optimizer_compile_args are correctly passed to optimizers."""
    mock_eval, _, _ = mock_bt_dependencies
    mock_eval.return_value = Mock(score=0.9)

    mock_p = CapturingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)

    # Define custom compile args for optimizer 'p'
    custom_args = {"num_trials": 20, "max_bootstrapped_demos": 8}

    optimizer.compile(
        student_with_lm,
        trainset=trainset,
        valset=valset,
        strategy="p",
        optimizer_compile_args={"p": custom_args}
    )

    # Verify the custom args were passed to the optimizer
    assert mock_p.received_kwargs is not None, "Optimizer compile was not called"
    assert "num_trials" in mock_p.received_kwargs, "num_trials not passed to optimizer"
    assert mock_p.received_kwargs["num_trials"] == 20, "num_trials value incorrect"
    assert "max_bootstrapped_demos" in mock_p.received_kwargs, "max_bootstrapped_demos not passed"
    assert mock_p.received_kwargs["max_bootstrapped_demos"] == 8, "max_bootstrapped_demos value incorrect"


def test_compile_args_multi_optimizer_strategy():
    """Test that different optimizers in a strategy receive their respective compile_args."""
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    # Create mock Teleprompters that capture their compile kwargs
    class PromptOptimizer(Teleprompter):
        def __init__(self):
            self.received_kwargs = {}

        def compile(self, student, trainset=None, num_trials=None, **kwargs):
            self.received_kwargs = {
                "trainset": trainset,
                "num_trials": num_trials,
                **kwargs
            }
            return student

    class WeightOptimizer(Teleprompter):
        def __init__(self):
            self.received_kwargs = {}

        def compile(self, student, trainset=None, num_batches=None, **kwargs):
            self.received_kwargs = {
                "trainset": trainset,
                "num_batches": num_batches,
                **kwargs
            }
            return student

    mock_p = PromptOptimizer()
    mock_w = WeightOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)

    # Define different compile args for each optimizer
    compile_args = {
        "p": {"num_trials": 10},
        "w": {"num_batches": 5}
    }

    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.return_value = Mock(score=0.85)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    optimizer.compile(
                        student,
                        trainset=trainset,
                        valset=valset,
                        strategy="p -> w",
                        optimizer_compile_args=compile_args
                    )

    # Verify each optimizer received its specific args
    assert mock_p.received_kwargs is not None, "Optimizer 'p' compile was not called"
    assert "num_trials" in mock_p.received_kwargs, "num_trials not passed to optimizer 'p'"
    assert mock_p.received_kwargs["num_trials"] == 10, "num_trials value incorrect for 'p'"
    assert mock_p.received_kwargs.get("num_batches") is None, "Optimizer 'p' should not receive 'w' args"

    assert mock_w.received_kwargs is not None, "Optimizer 'w' compile was not called"
    assert "num_batches" in mock_w.received_kwargs, "num_batches not passed to optimizer 'w'"
    assert mock_w.received_kwargs["num_batches"] == 5, "num_batches value incorrect for 'w'"
    assert mock_w.received_kwargs.get("num_trials") is None, "Optimizer 'w' should not receive 'p' args"


def test_compile_args_override_global_params():
    """Test that optimizer_compile_args override global trainset/valset/teacher parameters."""
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    # Create a mock Teleprompter that captures compile kwargs
    class CapturingTeleprompter(Teleprompter):
        def __init__(self):
            self.received_kwargs = {}

        def compile(self, student, trainset=None, valset=None, teacher=None, **kwargs):
            self.received_kwargs = {
                "trainset": trainset,
                "valset": valset,
                "teacher": teacher,
                **kwargs
            }
            return student

    mock_p = CapturingTeleprompter()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)

    # Create override values
    override_trainset = [examples[2]]  # Different from global trainset
    override_valset = [examples[0]]    # Different from global valset
    override_teacher = SimpleModule("input -> output")

    # Pass global values to compile, but override them in optimizer_compile_args
    compile_args = {
        "p": {
            "trainset": override_trainset,
            "valset": override_valset,
            "teacher": override_teacher,
        }
    }

    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.return_value = Mock(score=0.9)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                optimizer.compile(
                    student,
                    trainset=trainset,  # Global trainset (examples[:2])
                    valset=valset,      # Global valset (examples[2])
                    teacher=None,       # Global teacher (None)
                    strategy="p",
                    optimizer_compile_args=compile_args
                )

    # Verify the optimizer received the override values, not the global ones
    assert mock_p.received_kwargs["trainset"] == override_trainset, \
        "Optimizer should receive override trainset from compile_args"
    assert mock_p.received_kwargs["valset"] == override_valset, \
        "Optimizer should receive override valset from compile_args"
    assert mock_p.received_kwargs["teacher"] is override_teacher, \
        "Optimizer should receive override teacher from compile_args"

    # Verify they're different from the global values
    assert mock_p.received_kwargs["trainset"] != trainset, \
        "Override trainset should differ from global trainset"
    assert mock_p.received_kwargs["valset"] != valset, \
        "Override valset should differ from global valset"


def test_trainset_shuffling_between_steps():
    """Test that trainset is shuffled between steps when shuffle_trainset_between_steps=True."""
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    # Create mock optimizers that capture the trainset they receive
    trainsets_received = []

    class TrainsetCapturingOptimizer(Teleprompter):
        def compile(self, student, trainset=None, **kwargs):
            trainsets_received.append(trainset)
            return student

    mock_p = TrainsetCapturingOptimizer()
    mock_w = TrainsetCapturingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)

    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    optimizer.compile(
                        student,
                        trainset=trainset,
                        valset=valset,
                        strategy="p -> w",
                        shuffle_trainset_between_steps=True
                    )

    # Verify trainset was shuffled between steps
    assert len(trainsets_received) == 2, "Should have received trainset twice (for p and w)"
    trainset_p = trainsets_received[0]
    trainset_w = trainsets_received[1]

    # Both should have same examples but potentially different order
    assert len(trainset_p) == len(trainset_w), "Trainsets should have same length"
    # With shuffling enabled and only 2 examples, there's a 50% chance they're in different order
    # We can't reliably test order difference with small dataset, but we can verify they contain same examples
    assert set(id(ex) for ex in trainset_p) == set(id(ex) for ex in trainset_w), \
        "Trainsets should contain the same example objects"


def test_strategy_execution_order():
    """Test that strategy steps are executed in order and programs are passed correctly."""
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    # Track execution order and what program each optimizer receives
    execution_log = []

    class LoggingOptimizer(Teleprompter):
        def __init__(self, name):
            self.name = name

        def compile(self, student, **kwargs):
            # Create a new student with a marker to track the optimization path
            optimized = SimpleModule("input -> output")
            if not hasattr(student, "optimization_path"):
                optimized.optimization_path = [self.name]
            else:
                optimized.optimization_path = student.optimization_path + [self.name]
            execution_log.append((self.name, optimized.optimization_path.copy()))
            return optimized

    mock_p = LoggingOptimizer("p")
    mock_w = LoggingOptimizer("w")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)

    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.return_value = Mock(score=0.85)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    result = optimizer.compile(
                        student,
                        trainset=trainset,
                        valset=valset,
                        strategy="p -> w -> p"
                    )

    # Verify execution order
    assert len(execution_log) == 3, "Should have executed 3 optimization steps"
    assert execution_log[0] == ("p", ["p"]), "First step should be 'p'"
    assert execution_log[1] == ("w", ["p", "w"]), "Second step should be 'w' receiving output from 'p'"
    assert execution_log[2] == ("p", ["p", "w", "p"]), "Third step should be 'p' receiving output from 'w'"


def test_lm_lifecycle_management():
    """Test that launch_lms and kill_lms are called appropriately between steps."""
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    class SimpleOptimizer(Teleprompter):
        def compile(self, student, **kwargs):
            return student

    mock_p = SimpleOptimizer()
    mock_w = SimpleOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)

    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with patch("dspy.teleprompt.bettertogether.launch_lms") as mock_launch:
            with patch("dspy.teleprompt.bettertogether.kill_lms") as mock_kill:
                with patch.object(optimizer, "_models_changed", return_value=True):
                    optimizer.compile(
                        student,
                        trainset=trainset,
                        valset=valset,
                        strategy="p -> w"
                    )

    # Verify launch and kill were called
    # When models change (which we mocked to return True), launch should be called
    assert mock_launch.called, "launch_lms should be called when models change"
    assert mock_kill.called, "kill_lms should be called when models change"


def test_error_handling_returns_best_program():
    """Test that if a step fails, the best program found so far is still returned."""
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)

    # Create optimizers where the second one will fail
    class SuccessfulOptimizer(Teleprompter):
        def compile(self, student, **kwargs):
            optimized = SimpleModule("input -> output")
            optimized.step_name = "p_success"
            return optimized

    class FailingOptimizer(Teleprompter):
        def compile(self, student, **kwargs):
            raise RuntimeError("Intentional failure for testing")

    mock_p = SuccessfulOptimizer()
    mock_w = FailingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)

    # First call succeeds with score 0.7, second call (to failing optimizer) fails
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.side_effect = [
            Mock(score=0.5),  # Baseline
            Mock(score=0.7),  # After p (success)
        ]
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    result = optimizer.compile(
                        student,
                        trainset=trainset,
                        valset=valset,
                        strategy="p -> w"
                    )

    # Verify a program was returned despite the failure
    assert result is not None, "Should return a program even if a step fails"
    assert hasattr(result, "flag_compilation_error_occurred"), "Should have error flag"
    assert result.flag_compilation_error_occurred is True, "Error flag should be True"
    assert hasattr(result, "candidate_programs"), "Should have candidate_programs"
    assert len(result.candidate_programs) > 0, "Should have at least one candidate program"


@pytest.mark.parametrize("test_valset,expected_marker,test_description", [
    (valset, "p_optimized", "With valset: returns best score (p), not latest (w)"),
    (None, "w_optimized", "Without valset: returns latest program (w)"),
])
def test_program_selection(student_with_lm, test_valset, expected_marker, test_description):
    """Test program selection logic with and without validation set."""
    mock_p = MarkedOptimizer("p_optimized")
    mock_w = MarkedOptimizer("w_optimized")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)

    # Set up scores: baseline=0.5, p=0.9 (best), w=0.7
    # When test_valset is provided, best score wins; when None, latest wins
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        if test_valset is not None:
            mock_eval.side_effect = [
                Mock(score=0.5),  # Baseline
                Mock(score=0.9),  # After p (best score)
                Mock(score=0.7),  # After w (lower than p)
            ]
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    result = optimizer.compile(
                        student_with_lm,
                        trainset=trainset,
                        valset=test_valset,
                        strategy="p -> w"
                    )

    # Verify the correct program was returned based on valset presence
    assert hasattr(result, "marker"), "Result should have marker"
    assert result.marker == expected_marker, test_description


def test_candidate_programs_structure(student_with_lm):
    """Test that candidate_programs has the correct structure and content."""
    mock_p = MarkedOptimizer("p")
    mock_w = MarkedOptimizer("w")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)

    # Set up scores: baseline=0.5, p=0.8, w=0.9 (best)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program") as mock_eval:
        mock_eval.side_effect = [
            Mock(score=0.5),  # Baseline
            Mock(score=0.8),  # After p
            Mock(score=0.9),  # After w (best)
        ]
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    result = optimizer.compile(
                        student_with_lm,
                        trainset=trainset,
                        valset=valset,
                        strategy="p -> w"
                    )

    # Verify candidate_programs exists and has correct structure
    assert hasattr(result, "candidate_programs"), "Result should have candidate_programs attribute"
    candidates = result.candidate_programs

    # Should have 3 candidates: baseline, p, w
    assert len(candidates) == 3, f"Should have 3 candidates, got {len(candidates)}"

    # Each candidate should have the required keys
    for i, candidate in enumerate(candidates):
        assert "score" in candidate, f"Candidate {i} missing 'score' key"
        assert "program" in candidate, f"Candidate {i} missing 'program' key"
        assert "strategy" in candidate, f"Candidate {i} missing 'strategy' key"
        assert isinstance(candidate["score"], (int, float)), f"Candidate {i} score should be numeric"
        assert isinstance(candidate["program"], dspy.Module), f"Candidate {i} program should be a Module"
        assert isinstance(candidate["strategy"], (str, type(None))), f"Candidate {i} strategy should be str or None"

    # Candidates should be sorted by score (best first)
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True), "Candidates should be sorted by score (descending)"

    # Verify the best candidate is first
    assert candidates[0]["score"] == 0.9, "Best candidate should have score 0.9"
    assert candidates[0]["program"].marker == "w", "Best candidate should be from optimizer 'w'"

    # Verify baseline candidate
    baseline = [c for c in candidates if c["strategy"] is None or c["strategy"] == ""]
    assert len(baseline) == 1, "Should have exactly one baseline candidate"
    assert baseline[0]["score"] == 0.5, "Baseline should have score 0.5"


def test_empty_valset_handling(student_with_lm):
    """Test behavior when valset is an empty list vs None."""
    # Test with empty list []
    mock_p = MarkedOptimizer("optimized")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)

    with patch("dspy.teleprompt.bettertogether.launch_lms"):
        with patch("dspy.teleprompt.bettertogether.kill_lms"):
            with patch.object(optimizer, "_models_changed", return_value=False):
                result = optimizer.compile(
                    student_with_lm,
                    trainset=trainset,
                    valset=[],  # Empty list (not None)
                    strategy="p"
                )

    # With empty valset, should return latest program (same behavior as valset=None)
    assert hasattr(result, "marker"), "Result should have marker"
    assert result.marker == "optimized", "Should return the latest program when valset is empty list"
    assert hasattr(result, "candidate_programs"), "Should have candidate_programs"

    # Test with None - create fresh student and optimizer
    student2 = SimpleModule("input -> output")
    lm = DummyLM([{"output": "test"}])
    student2.set_lm(lm)

    mock_p2 = MarkedOptimizer("optimized")
    optimizer2 = BetterTogether(metric=simple_metric, p=mock_p2)

    with patch("dspy.teleprompt.bettertogether.launch_lms"):
        with patch("dspy.teleprompt.bettertogether.kill_lms"):
            with patch.object(optimizer2, "_models_changed", return_value=False):
                result2 = optimizer2.compile(
                    student2,
                    trainset=trainset,
                    valset=None,  # Explicit None
                    strategy="p"
                )

    # Both should behave the same way
    assert hasattr(result2, "marker"), "Result2 should have marker"
    assert result2.marker == "optimized", "Should return the latest program when valset is None"
