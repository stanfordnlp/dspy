import json
import logging
import os
import time
import uuid
from typing import Any

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, get_finetune_directory, validate_data_format

logger = logging.getLogger(__name__)


class TrainingJobTogether(TrainingJob):
    """A claim ticket for a Together AI fine-tuning job.

    Fine-tuning runs for minutes/hours, so instead of blocking we hold the
    job's id and expose `status()` to ask Together how it's going.
    """

    def __init__(self, job_id: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = job_id  # Together's fine-tune job id, set once we start the job

    def status(self):
        if self.job_id is None:
            return None  # no job started yet, so there's nothing to check
        # Lazy import: `together` is an optional dependency, only needed by people
        # who actually use Together fine-tuning (see databricks.py for the same pattern).
        try:
            from together import Together
        except ImportError:
            raise ImportError(
                "To use Together fine-tuning, please install the together package via `pip install together`."
            )
        client = Together()  # reads TOGETHER_API_KEY from the environment
        return client.fine_tuning.retrieve(id=self.job_id).status


class TogetherProvider(Provider):
    # Tells DSPy: yes, this provider can fine-tune, and here's the ticket type it hands out.
    finetunable = True
    TrainingJob = TrainingJobTogether

    @staticmethod
    def is_provider_model(model: str) -> bool:
        # The "bouncer": Together models are named with a "together_ai/" prefix.
        return model.startswith("together_ai/")

    @staticmethod
    def finetune(
        job: "TrainingJobTogether",
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None = "chat",
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        train_kwargs = train_kwargs or {}

        # --- Step 1: normalize the format label, then validate the data matches it ---
        # First the "spelling check": reject unknown format words early.
        if isinstance(train_data_format, str):
            if train_data_format == "chat":
                train_data_format = TrainDataFormat.CHAT
            elif train_data_format == "completion":
                train_data_format = TrainDataFormat.COMPLETION
            else:
                raise ValueError(
                    f"String `train_data_format` must be 'chat' or 'completion', but received: {train_data_format}."
                )
        # Then "open the box": check each training example actually matches the format.
        if train_data_format is not None:
            validate_data_format(train_data, train_data_format)

        # Lazy import (same optional-dependency pattern as status()).
        try:
            from together import Together
        except ImportError:
            raise ImportError(
                "To use Together fine-tuning, please install the together package via `pip install together`."
            )
        client = Together()

        # --- Step 2: save data to a local .jsonl, upload it, then clean up the local copy ---
        logger.info("Uploading training data to Together...")
        file_path = _save_data_to_jsonl(train_data)
        try:
            train_file = client.files.upload(file=file_path, purpose="fine-tune")
        finally:
            os.remove(file_path)  # the local file was only needed for the upload

        # --- Step 3: create the fine-tune job; store its id on the claim ticket ---
        logger.info("Starting Together fine-tuning job...")
        ft = client.fine_tuning.create(training_file=train_file.id, model=model, **train_kwargs)
        job.job_id = ft.id

        # --- Step 4: poll until the job completes or fails ---
        while True:
            status = job.status()
            if status == "completed":
                logger.info("Together fine-tuning job completed successfully!")
                break
            elif status in ("error", "cancelled"):
                raise ValueError(f"Together fine-tuning job {job.job_id} ended with status: {status}.")
            time.sleep(60)

        # --- Step 5: return the fine-tuned model's name, DSPy-prefixed ---
        output_name = client.fine_tuning.retrieve(id=job.job_id).x_model_output_name
        if not output_name:
            raise ValueError(f"Together fine-tuning job {job.job_id} completed but returned no model name.")
        return f"together_ai/{output_name}"


def _save_data_to_jsonl(train_data: list[dict[str, Any]]) -> str:
    """Write the training examples to a local JSONL file and return its path."""
    file_path = os.path.join(get_finetune_directory(), f"togetherai_{uuid.uuid4()}.jsonl")
    with open(file_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    return os.path.abspath(file_path)
