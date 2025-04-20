import time
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import openai

from dspy.clients.provider import TrainingJob, Provider #, RLBatchJob
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus, save_data

if TYPE_CHECKING:
    from dspy.clients.lm import LM

class SFTTrainingJobArbor(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_file_id = None
        self.provider_job_id = None

    def cancel(self):
        if ArborProvider.does_job_exist(self.provider_job_id):
            status = self.status()
            if ArborProvider.is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            openai.fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None

        if self.provider_file_id is not None:
            if ArborProvider.does_file_exist(self.provider_file_id):
                openai.files.delete(self.provider_file_id)
            self.provider_file_id = None

        super().cancel()

    def status(self) -> TrainingStatus:
        status = ArborProvider.get_training_status(self.provider_job_id)
        return status

# class RLBatchJobArbor(RLBatchJob):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.provider_job_id = None

#     def cancel(self):
#         if ArborProvider.does_job_exist(self.provider_job_id):
#             status = self.status()
#             if ArborProvider.is_terminal_training_status(status):
#                 err_msg = "Jobs that are complete cannot be canceled."
#                 err_msg += f" Job with ID {self.provider_job_id} is done."
#                 raise Exception(err_msg)
#             openai.fine_tuning.jobs.cancel(self.provider_job_id)
#             self.provider_job_id = None

#         # Assuming no file needed for RL batch training (the batches are small enough)
#         super().cancel()

#     def status(self) -> TrainingStatus:
#         status = ArborProvider.get_training_status(self.provider_job_id)
#         return status

class ArborProvider(Provider):

    def __init__(self, api_base: str):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = SFTTrainingJobArbor
        # self.RLBatchJob = RLBatchJobArbor
        self.api_base = api_base

    @staticmethod
    def launch(lm: "LM", launch_kwargs: Optional[Dict[str, Any]] = None):
        model = lm.model
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("arbor:"):
            model = model[6:]
        if model.startswith("huggingface/"):
            model = model[len("huggingface/"):]

        launch_kwargs = launch_kwargs or lm.launch_kwargs

        # Make request to launch endpoint
        response = requests.post(
            f"{launch_kwargs['api_base']}chat/launch",
            json={"model": model, "launch_kwargs": launch_kwargs}
        )

        if response.status_code != 200:
            raise Exception(f"Failed to launch model. Status code: {response.status_code}, Response: {response.text}")

        print(f"Inference server for model {model} launched successfully")
        lm.kwargs["api_base"] = f"{launch_kwargs['api_base']}"
        lm.kwargs["api_key"] = "arbor"

    @staticmethod
    def kill(lm: "LM", launch_kwargs: Optional[Dict[str, Any]] = None):
        launch_kwargs = launch_kwargs or lm.launch_kwargs
        response = requests.post(
            f"{launch_kwargs['api_base']}chat/kill",
        )

        if response.status_code != 200:
            raise Exception(f"Failed to kill model. Status code: {response.status_code}, Response: {response.text}")

        print(f"Inference killed successfully")

    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        provider_prefix = "openai/arbor:"
        return model.replace(provider_prefix, "")

    @staticmethod
    def finetune(
            job: SFTTrainingJobArbor,
            model: str,
            train_data: List[Dict[str, Any]],
            train_data_format: Optional[TrainDataFormat],
            train_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("arbor:"):
            model = model[6:]


        print("[Arbor Provider] Validating the data format")
        ArborProvider.validate_data_format(train_data_format, type='sft')

        print("[Arbor Provider] Saving the data to a file")
        data_path = save_data(train_data)
        print(f"[Arbor Provider] Data saved to {data_path}")

        print("[Arbor Provider] Uploading the data to the provider")
        provider_file_id = ArborProvider.upload_data(data_path)
        job.provider_file_id = provider_file_id

        print("[Arbor Provider] Starting remote training")
        provider_job_id = ArborProvider._start_remote_training(
            train_file_id=job.provider_file_id,
            model=model,
            train_kwargs=train_kwargs,
        )
        job.provider_job_id = provider_job_id
        print(f"[Arbor Provider] Job started with the Arbor Job ID {provider_job_id}")

        print("[Arbor Provider] Waiting for training to complete")
        ArborProvider.wait_for_job(job, training_kwargs)

        print("[Arbor Provider] Attempting to retrieve the trained model")
        model = ArborProvider.get_trained_model(job)
        print(f"[Arbor Provider] Model retrieved: {model}")

        return f"openai/arbor:{model}"

    # @staticmethod
    # def rl_train_batch(
    #         job: RLBatchJobArbor,
    #         model: str,
    #         batch_data: List[Dict[str, Any]],
    #         batch_data_format: Optional[TrainDataFormat],
    #         train_kwargs: Optional[Dict[str, Any]] = None
    # ) -> str:
    #     model = ArborProvider._remove_provider_prefix(model)

    #     print("[Arbor Provider] Validating the data format")
    #     ArborProvider.validate_data_format(batch_data_format, type='rl')

    #     print("[Arbor Provider] Starting remote training")
    #     provider_job_id = ArborProvider._start_remote_rl_train_batch(
    #         batch_data=batch_data,
    #         model=model,
    #         train_kwargs=train_kwargs,
    #     )
    #     job.provider_job_id = provider_job_id
    #     print(f"[Arbor Provider] Job started with the Arbor Job ID {provider_job_id}")

    #     print("[Arbor Provider] Waiting for training to complete")
    #     # TODO: This takes 1 second but that might still be too long
    #     ArborProvider.wait_for_job(job, poll_frequency=1)

    #     model = ArborProvider.get_trained_model(job)
    #     print(f"[Arbor Provider] Model retrieved: {model}")

    #     return f"openai/arbor:{model}"


    @staticmethod
    def does_job_exist(job_id: str, training_kwargs: Dict[str, Any]) -> bool:
        try:
            original_base_url = openai.base_url
            openai.base_url = training_kwargs['api_base']
            openai.fine_tuning.jobs.retrieve(job_id)
            openai.base_url = original_base_url
            return True
        except Exception:
            return False

    @staticmethod
    def does_file_exist(file_id: str, training_kwargs: Dict[str, Any]) -> bool:
        try:
            original_base_url = openai.base_url
            openai.base_url = training_kwargs['api_base']
            openai.files.retrieve(file_id)
            openai.base_url = original_base_url
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
    def get_training_status(job_id: str, training_kwargs: Dict[str, Any]) -> TrainingStatus:
        provider_status_to_training_status = {
            "validating_files": TrainingStatus.pending,
            "queued": TrainingStatus.pending,
            "running": TrainingStatus.running,
            "succeeded": TrainingStatus.succeeded,
            "failed": TrainingStatus.failed,
            "cancelled": TrainingStatus.cancelled,
            "pending": TrainingStatus.pending,
            "pending_pause": TrainingStatus.pending,
            "pending_resume": TrainingStatus.pending,
            "paused": TrainingStatus.pending,
            "pending_cancel": TrainingStatus.pending,
        }

        if job_id is None:
            print("There is no active job.")
            return TrainingStatus.not_started

        err_msg = f"Job with ID {job_id} does not exist."
        assert ArborProvider.does_job_exist(job_id, training_kwargs), err_msg

        original_base_url = openai.base_url
        openai.base_url = training_kwargs['api_base']
        provider_job = openai.fine_tuning.jobs.retrieve(job_id)
        openai.base_url = original_base_url

        provider_status = provider_job.status
        status = provider_status_to_training_status[provider_status]

        return status

    @staticmethod
    def validate_data_format(data_format: TrainDataFormat, type: str):
        supported_data_formats = {
            'sft': [
                TrainDataFormat.CHAT,
                TrainDataFormat.COMPLETION,
            ],
            'rl': [
                TrainDataFormat.CHAT,
                TrainDataFormat.COMPLETION,
            ]
        }[type]

        if data_format not in supported_data_formats:
            err_msg = f"Arbor does not support the data format {data_format}."
            raise ValueError(err_msg)

    @staticmethod
    def upload_data(data_path: str, training_kwargs: Dict[str, Any]) -> str:
        # Upload the data to the provider
        original_base_url = openai.base_url
        openai.base_url = training_kwargs['api_base']
        provider_file = openai.files.create(
            file=open(data_path, "rb"),
            purpose="fine-tune",
        )
        openai.base_url = original_base_url

        return provider_file.id

    @staticmethod
    def _start_remote_training(
            train_file_id: str,
            model: str,
            train_kwargs: Dict[str, Any]
    ) -> str:
        train_kwargs = train_kwargs or {}
        original_base_url = openai.base_url
        openai.base_url = train_kwargs['api_base']
        provider_job = openai.fine_tuning.jobs.create(
            model=model,
            training_file=train_file_id,
            hyperparameters=train_kwargs,
        )
        openai.base_url = original_base_url
        return provider_job.id

    @staticmethod
    def wait_for_job(
            job: TrainingJob,
            training_kwargs: Dict[str, Any],
            poll_frequency: int = 20,
    ):
        done = False
        cur_event_id = None
        reported_estimated_time = False
        while not done:
            # Report estimated time if not already reported
            if not reported_estimated_time:

                original_base_url = openai.base_url
                openai.base_url = training_kwargs['api_base']
                remote_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
                openai.base_url = original_base_url

                timestamp = remote_job.estimated_finish
                if timestamp:
                    estimated_finish_dt = datetime.fromtimestamp(timestamp)
                    delta_dt = estimated_finish_dt - datetime.now()
                    print(f"[Arbor Provider] The Arbor estimated time remaining is: {delta_dt}")
                    reported_estimated_time = True

            # Get new events
            original_base_url = openai.base_url
            openai.base_url = training_kwargs['api_base']
            page = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job.provider_job_id, limit=1)
            openai.base_url = original_base_url

            new_event = page.data[0] if page.data else None
            if new_event and new_event.id != cur_event_id:
                dt = datetime.fromtimestamp(new_event.created_at)
                print(f"[Arbor Provider] {dt} {new_event.message}")
                cur_event_id = new_event.id

            # Sleep and update the flag
            time.sleep(poll_frequency)
            done = ArborProvider.is_terminal_training_status(job.status())

    @staticmethod
    def get_trained_model(job, training_kwargs: Dict[str, Any]):
        status = job.status()
        if status != TrainingStatus.succeeded:
            err_msg = f"Job status is {status}."
            err_msg += f" Must be {TrainingStatus.succeeded} to retrieve model."
            raise Exception(err_msg)

        original_base_url = openai.base_url
        openai.base_url = training_kwargs['api_base']
        provider_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
        openai.base_url = original_base_url

        finetuned_model = provider_job.fine_tuned_model
        return finetuned_model
