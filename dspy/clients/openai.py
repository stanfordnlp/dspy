import re
import time
from datetime import datetime
from typing import Any

import openai

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus, save_data

_OPENAI_MODELS = [
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "tts-1",
    "tts-1-1106",
    "chatgpt-4o-latest",
    "dall-e-2",
    "whisper-1",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "babbage-002",
    "davinci-002",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "dall-e-3",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "o1-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "o1-mini-2024-09-12",
    "gpt-4o-audio-preview",
    "tts-1-hd",
    "tts-1-hd-1106",
    "o1-preview-2024-09-12",
    "o1-mini",
    "gpt-4-1106-preview",
    "text-embedding-ada-002",
    "gpt-3.5-turbo-16k",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview",
    "gpt-3.5-turbo-1106",
    "gpt-4-0613",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4",
    "gpt-3.5-turbo-instruct-0914",
]


class TrainingJobOpenAI(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_file_id = None
        self.provider_job_id = None

    def cancel(self):
        # Cancel the provider job
        if OpenAIProvider.does_job_exist(self.provider_job_id):
            status = self.status()
            if OpenAIProvider.is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            openai.fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None

        # Delete the provider file
        if self.provider_file_id is not None:
            if OpenAIProvider.does_file_exist(self.provider_file_id):
                openai.files.delete(self.provider_file_id)
            self.provider_file_id = None

        # Call the super's cancel method after the custom cancellation logic
        super().cancel()

    def status(self) -> TrainingStatus:
        status = OpenAIProvider.get_training_status(self.provider_job_id)
        return status


class OpenAIProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobOpenAI

    @staticmethod
    def is_provider_model(model: str) -> bool:
        model = OpenAIProvider._remove_provider_prefix(model)

        # Check if the model is a base OpenAI model
        # TODO(enhance) The following list can be replaced with
        # openai.models.list(), but doing so might require a key. Is there a
        # way to get the list of models without a key?
        if model in _OPENAI_MODELS:
            return True

        # Check if the model is a fine-tuned OpenAI model. Fine-tuned OpenAI
        # models have the prefix "ft:<BASE_MODEL_NAME>:", followed by a string
        # specifying the fine-tuned model. The following RegEx pattern is used
        # to match the base model name.
        # TODO(enhance): This part can be updated to match the actual fine-tuned
        # model names by making a call to the OpenAI API to be more exact, but
        # this might require an API key with the right permissions.
        match = re.match(r"ft:([^:]+):", model)
        if match and match.group(1) in _OPENAI_MODELS:
            return True

        return False

    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        provider_prefix = "openai/"
        return model.replace(provider_prefix, "")

    @staticmethod
    def finetune(
        job: TrainingJobOpenAI,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        model = OpenAIProvider._remove_provider_prefix(model)

        print("[OpenAI Provider] Validating the data format")
        OpenAIProvider.validate_data_format(train_data_format)

        print("[OpenAI Provider] Saving the data to a file")
        data_path = save_data(train_data)
        print(f"[OpenAI Provider] Data saved to {data_path}")

        print("[OpenAI Provider] Uploading the data to the provider")
        provider_file_id = OpenAIProvider.upload_data(data_path)
        job.provider_file_id = provider_file_id

        print("[OpenAI Provider] Starting remote training")
        provider_job_id = OpenAIProvider._start_remote_training(
            train_file_id=job.provider_file_id,
            model=model,
            train_kwargs=train_kwargs,
        )
        job.provider_job_id = provider_job_id
        print(f"[OpenAI Provider] Job started with the OpenAI Job ID {provider_job_id}")

        print("[OpenAI Provider] Waiting for training to complete")
        # TODO(feature): Could we stream OAI logs?
        OpenAIProvider.wait_for_job(job)

        print("[OpenAI Provider] Attempting to retrieve the trained model")
        model = OpenAIProvider.get_trained_model(job)
        print(f"[OpenAI Provider] Model retrieved: {model}")

        return model

    @staticmethod
    def does_job_exist(job_id: str) -> bool:
        try:
            # TODO(nit): This call may fail for other reasons. We should check
            # the error message to ensure that the job does not exist.
            openai.fine_tuning.jobs.retrieve(job_id)
            return True
        except Exception:
            return False

    @staticmethod
    def does_file_exist(file_id: str) -> bool:
        try:
            # TODO(nit): This call may fail for other reasons. We should check
            # the error message to ensure that the file does not exist.
            openai.files.retrieve(file_id)
            return True
        except Exception:
            return False

    @staticmethod
    def is_terminal_training_status(status: TrainingStatus) -> bool:
        return status in [
            TrainingStatus.succeeded,
            TrainingStatus.failed,
            TrainingStatus.cancelled,
        ]

    @staticmethod
    def get_training_status(job_id: str) -> TrainingStatus:
        provider_status_to_training_status = {
            "validating_files": TrainingStatus.pending,
            "queued": TrainingStatus.pending,
            "running": TrainingStatus.running,
            "succeeded": TrainingStatus.succeeded,
            "failed": TrainingStatus.failed,
            "cancelled": TrainingStatus.cancelled,
        }

        # Check if there is an active job
        if job_id is None:
            print("There is no active job.")
            return TrainingStatus.not_started

        err_msg = f"Job with ID {job_id} does not exist."
        assert OpenAIProvider.does_job_exist(job_id), err_msg

        # Retrieve the provider's job and report the status
        provider_job = openai.fine_tuning.jobs.retrieve(job_id)
        provider_status = provider_job.status
        status = provider_status_to_training_status[provider_status]

        return status

    @staticmethod
    def validate_data_format(data_format: TrainDataFormat):
        supported_data_formats = [
            TrainDataFormat.CHAT,
            TrainDataFormat.COMPLETION,
        ]
        if data_format not in supported_data_formats:
            err_msg = f"OpenAI does not support the data format {data_format}."
            raise ValueError(err_msg)

    @staticmethod
    def upload_data(data_path: str) -> str:
        # Upload the data to the provider
        provider_file = openai.files.create(
            file=open(data_path, "rb"),
            purpose="fine-tune",
        )
        return provider_file.id

    @staticmethod
    def _start_remote_training(train_file_id: str, model: str, train_kwargs: dict[str, Any] | None = None) -> str:
        train_kwargs = train_kwargs or {}
        provider_job = openai.fine_tuning.jobs.create(
            model=model,
            training_file=train_file_id,
            hyperparameters=train_kwargs,
        )
        return provider_job.id

    @staticmethod
    def wait_for_job(
        job: TrainingJobOpenAI,
        poll_frequency: int = 20,
    ):
        # Poll for the job until it is done
        done = False
        cur_event_id = None
        reported_estimated_time = False
        while not done:
            # Report estimated time if not already reported
            if not reported_estimated_time:
                remote_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
                timestamp = remote_job.estimated_finish
                if timestamp:
                    estimated_finish_dt = datetime.fromtimestamp(timestamp)
                    delta_dt = estimated_finish_dt - datetime.now()
                    print(f"[OpenAI Provider] The OpenAI estimated time remaining is: {delta_dt}")
                    reported_estimated_time = True

            # Get new events
            page = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job.provider_job_id, limit=1)
            new_event = page.data[0] if page.data else None
            if new_event and new_event.id != cur_event_id:
                dt = datetime.fromtimestamp(new_event.created_at)
                print(f"[OpenAI Provider] {dt} {new_event.message}")
                cur_event_id = new_event.id

            # Sleep and update the flag
            time.sleep(poll_frequency)
            done = OpenAIProvider.is_terminal_training_status(job.status())

    @staticmethod
    def get_trained_model(job):
        status = job.status()
        if status != TrainingStatus.succeeded:
            err_msg = f"Job status is {status}."
            err_msg += f" Must be {TrainingStatus.succeeded} to retrieve model."
            raise Exception(err_msg)

        provider_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
        finetuned_model = provider_job.fine_tuned_model
        return finetuned_model
