from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "humaneval_dspy_harness_bootstrap_v0.py"
)
MOCK_HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "mocks"
    / "humaneval_dspy_harness_bootstrap_v0_mock.py"
)


def _load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def humaneval_harness() -> ModuleType:
    return _load_script_module(
        "humaneval_dspy_harness_bootstrap_v0",
        HARNESS_PATH,
    )


@pytest.fixture(scope="session")
def humaneval_mock_harness() -> ModuleType:
    sys.path.insert(0, str(HARNESS_PATH.parent))
    try:
        return _load_script_module(
            "humaneval_dspy_harness_bootstrap_v0_mock",
            MOCK_HARNESS_PATH,
        )
    finally:
        sys.path.remove(str(HARNESS_PATH.parent))
