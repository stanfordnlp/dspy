import pytest
import copy


SKIP_DEFAULT_FLAGS = ["reliability"]

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

# Taken from: https://gist.github.com/justinmklam/b2aca28cb3a6896678e2e2927c6b6a38
def pytest_addoption(parser):
    for flag in SKIP_DEFAULT_FLAGS:
        parser.addoption(
            "--{}".format(flag),
            action="store_true",
            default=False,
            help="run {} tests".format(flag),
        )

def pytest_configure(config):
    for flag in SKIP_DEFAULT_FLAGS:
        config.addinivalue_line("markers", flag)

def pytest_collection_modifyitems(config, items):
    for flag in SKIP_DEFAULT_FLAGS:
        if config.getoption("--{}".format(flag)):
            return

        skip_mark = pytest.mark.skip(reason="need --{} option to run".format(flag))
        for item in items:
            if flag in item.keywords:
                item.add_marker(skip_mark)