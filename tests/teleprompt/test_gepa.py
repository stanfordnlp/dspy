import sys
from unittest.mock import patch, MagicMock
import pytest

from dspy.teleprompt.gepa.gepa import GEPA


def test_import_dependencies_with_sys_modules_mock():
    """Test that GEPA can be imported without gepa package installed by mocking sys.modules."""
    
    # Mock sys.modules to simulate gepa not being installed
    with patch.dict(sys.modules, {'gepa': None}):
        # This should not raise an ImportError
        from dspy.teleprompt.gepa.gepa import GEPA
        assert GEPA is not None


def test_import_dependencies_with_import_patch():
    """Test that GEPA can be imported without gepa package installed by patching __import__."""
    
    def mock_import(name, *args, **kwargs):
        if name == 'gepa':
            raise ImportError("No module named 'gepa'")
        return __import__(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        # This should not raise an ImportError
        from dspy.teleprompt.gepa.gepa import GEPA
        assert GEPA is not None


def test_gepa_compile_requires_gepa_package():
    """Test that GEPA.compile() requires the gepa package to be installed."""
    
    # Mock the metric function
    def mock_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0
    
    # Create GEPA instance
    gepa = GEPA(
        metric=mock_metric,
        auto="light",
        reflection_lm=MagicMock()
    )
    
    # Mock the student module
    student = MagicMock()
    student.predictors.return_value = [MagicMock()]
    student.named_predictors.return_value = [("test_pred", MagicMock())]
    
    # Mock trainset
    trainset = [MagicMock()]
    
    # Test that compile() raises ImportError when gepa is not available
    with patch.dict(sys.modules, {'gepa': None}):
        with pytest.raises(ImportError, match="No module named 'gepa'"):
            gepa.compile(student, trainset=trainset)


def test_gepa_compile_with_gepa_available():
    """Test that GEPA.compile() works when gepa package is available."""
    
    # Mock the metric function
    def mock_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0
    
    # Create GEPA instance
    gepa = GEPA(
        metric=mock_metric,
        auto="light",
        reflection_lm=MagicMock()
    )
    
    # Mock the student module
    student = MagicMock()
    student.predictors.return_value = [MagicMock()]
    student.named_predictors.return_value = [("test_pred", MagicMock())]
    
    # Mock trainset
    trainset = [MagicMock()]
    
    # Mock the gepa package and its functions
    mock_gepa = MagicMock()
    mock_optimize = MagicMock()
    mock_gepa_result = MagicMock()
    mock_gepa_result.candidates = [{"test_pred": "test instructions"}]
    mock_gepa_result.parents = [[]]
    mock_gepa_result.val_aggregate_scores = [1.0]
    mock_gepa_result.best_outputs_valset = None
    mock_gepa_result.val_subscores = [[1.0]]
    mock_gepa_result.per_val_instance_best_candidates = [set([0])]
    mock_gepa_result.discovery_eval_counts = [1]
    mock_gepa_result.total_metric_calls = 1
    mock_gepa_result.num_full_val_evals = 1
    mock_gepa_result.run_dir = None
    mock_gepa_result.seed = 0
    mock_optimize.return_value = mock_gepa_result
    
    with patch.dict(sys.modules, {'gepa': mock_gepa}):
        with patch('dspy.teleprompt.gepa.gepa.optimize', mock_optimize):
            # Mock the adapter and its methods
            with patch('dspy.teleprompt.gepa.gepa.DspyAdapter') as mock_adapter_class:
                mock_adapter = MagicMock()
                mock_adapter_class.return_value = mock_adapter
                mock_adapter.build_program.return_value = MagicMock()
                
                # This should not raise an ImportError
                result = gepa.compile(student, trainset=trainset)
                assert result is not None 