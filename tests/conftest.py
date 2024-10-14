import pytest

import dspy
from dsp.utils.settings import DEFAULT_CONFIG


@pytest.fixture(autouse=True)
def clear_settings():
    """Ensures that the settings are cleared after each test."""

    yield

    dspy.settings.configure(**DEFAULT_CONFIG, inherit_config=False)
