import pytest
import copy


@pytest.fixture(autouse=True)
def clear_settings():
    """Ensures that the settings are cleared after each test."""

    yield

    import dspy
    from dspy.dsp.utils.settings import DEFAULT_CONFIG

    dspy.settings.configure(**copy.deepcopy(DEFAULT_CONFIG), inherit_config=False)


@pytest.fixture
def anyio_backend():
    return "asyncio"
