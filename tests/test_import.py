import builtins
import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clean_dspy_modules():
    """Remove dspy and all its submodules from sys.modules to ensure a clean import."""
    to_remove = [mod for mod in list(sys.modules) if mod == "dspy" or mod.startswith("dspy.")]
    for mod in to_remove:
        del sys.modules[mod]

def test_import_dependencies():
    """Test that DSPy can be imported without importing unnecessary packages."""
    unnecessary_packages = {
        "gepa", "datesets", "optuna", "pandas"
    }

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Mock import function that tracks imports and blocks unwanted ones."""
        if name in unnecessary_packages:
            raise ImportError(f"Package '{name}' is not available")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        pass
