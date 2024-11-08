import os

import pytest

import dspy
from tests.conftest import clear_settings
from tests.quality.utils import _parse_quality_conf_yaml

# List of models for which to run quality tests
MODEL_LIST = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-o1-preview",
    "gpt-o1-mini",
    "claude-3.5-sonnet",
    "claude-3.5-haiku",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "llama-3.1-405b-instruct",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct",
]


# Hook to parameterize tests with each model in the quality tests YAML configuration
def pytest_generate_tests(metafunc):
    known_failing_models = getattr(metafunc.function, "_known_failing_models", [])

    if "configure_model" in metafunc.fixturenames:
        params = [(model, model in known_failing_models) for model in MODEL_LIST]
        ids = [f"{model}" for model, _ in params]  # Custom IDs for display
        metafunc.parametrize("configure_model", params, indirect=True, ids=ids)


@pytest.fixture(autouse=True)
def configure_model(request):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(module_dir, "quality_conf.yaml")
    quality_conf = _parse_quality_conf_yaml(conf_path)

    if quality_conf.adapter.lower() == "chat":
        adapter = dspy.ChatAdapter()
    elif quality_conf.adapter.lower() == "json":
        adapter = dspy.JSONAdapter()
    else:
        raise ValueError(f"Unknown adapter specification '{adapter}' in quality_conf.yaml")

    model_name, should_ignore_failure = request.param
    model_params = quality_conf.models.get(model_name)
    if model_params:
        lm = dspy.LM(**model_params)
        dspy.configure(lm=lm, adapter=adapter)
    else:
        pytest.skip(
            f"Skipping test because no quality testing YAML configuration was found" f" for model {model_name}."
        )

    # Store `should_ignore_failure` flag on the request node for use in post-test handling
    request.node.should_ignore_failure = should_ignore_failure
    request.node.model_name = model_name


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to conditionally ignore failures in a given test case for known failing models.
    """
    outcome = yield
    rep = outcome.get_result()

    should_ignore_failure = getattr(item, "should_ignore_failure", False)
    model_name = getattr(item, "model_name", "<unknown>")

    if should_ignore_failure and rep.failed:
        rep.outcome = "passed"
        rep.wasxfail = f"Ignoring failure for known failing model '{model_name}'"
