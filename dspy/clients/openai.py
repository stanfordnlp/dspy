import re
import time
from typing import Any, Dict, List, Optional

import openai

from dspy.clients.provider import TrainingJob, Provider
from dspy.clients.utils_finetune import DataFormat, TrainingStatus, save_data
from dspy.utils.logging import logger


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
        # Filter the provider_prefix, if exists
        provider_prefix = "openai/"
        if model.startswith(provider_prefix):
            model = model[len(provider_prefix) :]

        client = openai.OpenAI()
        valid_model_names = [model.id for model in client.models.list().data]
        # Check if the model is a base OpenAI model
        if model in valid_model_names:
            return True

        # Check if the model is a fine-tuned OpneAI model. Fine-tuned OpenAI
        # models have the prefix "ft:<BASE_MODEL_NAME>:", followed by a string
        # specifying the fine-tuned model. The following RegEx pattern is used
        # to match the base model name.
        # TODO(enhance): This part can be updated to match the actual fine-tuned
        # model names by making a call to the OpenAI API to be more exact, but
        # this might require an API key with the right permissions.
        match = re.match(r"ft:([^:]+):", model)
        if match and match.group(1) in valid_model_names:
            return True

        return False

    @staticmethod
    def finetune(
        job: TrainingJobOpenAI,
        model: str,
        train_data: List[Dict[str, Any]],
        train_kwargs: Optional[Dict[str, Any]] = None,
        data_format: Optional[DataFormat] = None,
    ) -> str:
        logger.info("[Finetune] Validating the data format")
        OpenAIProvider.validate_data_format(data_format)
        logger.info("[Finetune] Done!")

        logger.info("[Finetune] Saving the data to a file")
        data_path = save_data(train_data)
        logger.info("[Finetune] Done!")

        logger.info("[Finetune] Uploading the data to the provider")
        provider_file_id = OpenAIProvider.upload_data(data_path)
        job.provider_file_id = provider_file_id
        logger.info("[Finetune] Done!")

        logger.info("[Finetune] Start remote training")
        provider_job_id = OpenAIProvider.start_remote_training(
            train_file_id=job.provider_file_id,
            model=model,
            train_kwargs=train_kwargs,
        )
        job.provider_job_id = provider_job_id
        logger.info("[Finetune] Done!")

        logger.info("[Finetune] Wait for training to complete")
        # TODO(feature): Could we stream OAI logs?
        OpenAIProvider.wait_for_job(job)
        logger.info("[Finetune] Done!")

        logger.info("[Finetune] Get trained model if the run was a success")
        model = OpenAIProvider.get_trained_model(job)
        logger.info("[Finetune] Done!")

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
            logger.info("There is no active job.")
            return TrainingStatus.not_started

        err_msg = f"Job with ID {job_id} does not exist."
        assert OpenAIProvider.does_job_exist(job_id), err_msg

        # Retrieve the provider's job and report the status
        provider_job = openai.fine_tuning.jobs.retrieve(job_id)
        provider_status = provider_job.status
        status = provider_status_to_training_status[provider_status]

        return status

    @staticmethod
    def validate_data_format(data_format: DataFormat):
        supported_data_formats = [
            DataFormat.chat,
            DataFormat.completion,
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
    def start_remote_training(
        train_file_id: str,
        model: id,
        train_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
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
        done = False
        while not done:
            done = OpenAIProvider.is_terminal_training_status(job.status())
            time.sleep(poll_frequency)


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
