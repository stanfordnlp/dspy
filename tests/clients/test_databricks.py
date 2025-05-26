"""Test the Databricks finetuning and deployment.

This test requires valid Databricks credentials, so it is skipped on github actions. Right now it is only used for
manual testing.
"""

import pytest

import dspy
from dspy.clients.databricks import (
    DatabricksProvider,
    TrainingJobDatabricks,
    _create_directory_in_databricks_unity_catalog,
)

try:
    from databricks.sdk import WorkspaceClient

    WorkspaceClient()
except (ImportError, Exception):
    # Skip the test if the Databricks SDK is not configured or credentials are not available.
    pytestmark = pytest.mark.skip(reason="Databricks SDK not configured or credentials not available")


def test_create_directory_in_databricks_unity_catalog():
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    with pytest.raises(
        ValueError,
        match=(
            "Databricks Unity Catalog path must be in the format '/Volumes/<catalog>/<schema>/<volume>/...', "
            "but received: /badstring/whatever"
        ),
    ):
        _create_directory_in_databricks_unity_catalog(w, "/badstring/whatever")

    _create_directory_in_databricks_unity_catalog(w, "/Volumes/main/chenmoney/testing/dspy_testing")
    # Check that the directory was created successfully, otherwise `get_directory_metadata` will raise an exception.
    w.files.get_directory_metadata("/Volumes/main/chenmoney/testing/dspy_testing")


def test_create_finetuning_job():
    fake_training_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "Berlin!"},
            ]
        },
    ]
    dspy.settings.experimental = True

    job = TrainingJobDatabricks()

    DatabricksProvider.finetune(
        job=job,
        model="meta-llama/Llama-3.2-1B",
        train_data=fake_training_data,
        data_format="chat",
        train_kwargs={
            "train_data_path": "/Volumes/main/chenmoney/testing/dspy_testing",
            "register_to": "main.chenmoney.finetuned_model",
            "task_type": "CHAT_COMPLETION",
            "skip_deploy": True,
        },
    )
    assert job.finetuning_run.status.display_name is not None


def test_deploy_finetuned_model():
    dspy.settings.experimental = True
    model_to_deploy = "main.chenmoney.finetuned_model"

    DatabricksProvider.deploy_finetuned_model(
        model=model_to_deploy,
        data_format="chat",
    )

    lm = dspy.LM(model="databricks/main_chenmoney_finetuned_model")
    lm("what is 2 + 2?")
