import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import openai
import requests

import dspy
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import GRPOGroup, MultiGPUConfig, TrainDataFormat, TrainingStatus, save_data
from dspy.teleprompt.grpo.grpo_config import GRPOConfig

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class ArborTrainingJob(TrainingJob):
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


class ArborReinforceJob(ReinforceJob):
    def __init__(self, lm: "LM", train_kwargs: GRPOConfig, gpu_config: MultiGPUConfig = MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1)):
        # The teleprompter must ensure that this is set
        if not isinstance(train_kwargs, GRPOConfig):
            raise TypeError(f"Expected train_kwargs to be of type GRPOConfig, but got {type(train_kwargs)}")

        self.lm = lm
        self.train_kwargs: GRPOConfig = train_kwargs
        self.provider_job_id = None
        self.checkpoints = {}
        self.last_checkpoint = None
        self.gpu_config = gpu_config

    def initialize(self):
        # TODO(GRPO Team): Set provider job ID
        api_base = self.lm.kwargs["api_base"]

        finetune_model = ArborProvider._remove_provider_prefix(self.lm.model)
        # Only multi-GPU is supported for now
        gpu_config_type = "multi"
        
        # Create data payload from GRPOConfig
        data = self.train_kwargs.to_dict()
        data["model"] = finetune_model
        data["gpu_config"] = {
            "type": gpu_config_type,
            gpu_config_type: self.gpu_config,
        }
        url = urljoin(api_base, "fine_tuning/grpo/initialize")
        headers = {"Content-Type": "application/json"}
        response = requests.post(url=url, headers=headers, json=data)
        assert response.status_code == 200, f"Failed to initialize GRPO: {response}"
        response = response.json()
        self.lm.model = ArborProvider._add_provider_prefix(response["current_model"])
        self.provider_job_id = response.get("job_id")

    def _run_grpo_step_one_group(
        self, train_group: GRPOGroup, train_data_format: TrainDataFormat | str | None = None
    ):
        # TODO: Check that the data follows the intended format
        api_base = self.lm.kwargs["api_base"]
        # api_key = self.lm.kwargs["api_key"]

        finetune_model = ArborProvider._remove_provider_prefix(self.lm.model)
        data = {"job_id": self.provider_job_id, "model": finetune_model, "batch": train_group}
        url = urljoin(api_base, f"fine_tuning/grpo/{self.provider_job_id}/step")
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, f"Failed to run a GRPO step: {response.text}"
        response = response.json()
        assert "current_model" in response, f"Response does not contain the next model ID to be used: {response}"
        current_model = response["current_model"]
        self.lm.model = ArborProvider._add_provider_prefix(current_model)

    def step(self, train_data: list[GRPOGroup], train_data_format: TrainDataFormat | str | None):
        # Note: TrainDataFormat specifies the format for the inner most dict.
        # Because we run GRPO at the group level, train_data will be a list of
        # groups, where each group is a list of GRPOChatData. Our teleprompters
        # ensure that we pass the right data format.
        # We can consider making this distinction clearer, e.g., by having two
        # different step methods or changing our smallets data format to be the
        # GRPO group.
        # TODO: Support step on the server side
        assert (
            train_data_format == TrainDataFormat.GRPO_CHAT
        ), f"GRPO only supports the GRPO_CHAT data format. Got {train_data_format} instead."
        for group in train_data:
            self._run_grpo_step_one_group(group, train_data_format)

    def save_checkpoint(self, checkpoint_name: str, score: float | None = None):
        api_base = self.lm.kwargs["api_base"]
        url = urljoin(api_base, f"fine_tuning/grpo/{self.provider_job_id}/checkpoint")
        headers = {"Content-Type": "application/json"}
        body = {"job_id": self.provider_job_id, "checkpoint_name": checkpoint_name}
        response = requests.post(url, headers=headers, json=body)
        assert response.status_code == 200, f"Failed to save checkpoint: {response.text}"
        response = response.json()

        last_checkpoint = response["last_checkpoint"]
        checkpoints = response["checkpoints"]
        checkpoint_model_path = checkpoints[last_checkpoint]
        self.checkpoints[last_checkpoint] = {
            "model_path": checkpoint_model_path,
            "score": score,
        }
        self.last_checkpoint = last_checkpoint

    def terminate(self):
        api_base = self.lm.kwargs["api_base"]

        url = urljoin(api_base, f"fine_tuning/grpo/{self.provider_job_id}/terminate")
        headers = {"Content-Type": "application/json"}
        body = {"job_id": self.provider_job_id}
        response = requests.post(url, headers=headers, json=body)
        assert response.status_code == 200, f"Failed to terminate GRPO: {response.text}"

        response = response.json()
        current_model = response["current_model"]
        self.lm.model = ArborProvider._add_provider_prefix(current_model)

    def cancel(self):
        if self.provider_job_id:
            api_base = self.lm.kwargs["api_base"]
            url = urljoin(api_base, f"fine_tuning/grpo/{self.provider_job_id}/cancel")
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers)
            if response.status_code == 200:
                self.provider_job_id = None
            else:
                raise Exception(f"Failed to cancel GRPO job: {response.text}")

    def status(self) -> TrainingStatus:
        status = ArborProvider.get_training_status(self.provider_job_id)
        return status


class ArborProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.reinforceable = True
        self.TrainingJob = ArborTrainingJob
        self.ReinforceJob = ArborReinforceJob

    @staticmethod
    def launch(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        model = ArborProvider._remove_provider_prefix(lm.model)

        api_base = lm.kwargs["api_base"]

        launch_kwargs = launch_kwargs or lm.launch_kwargs

        # Make request to launch endpoint
        response = requests.post(urljoin(api_base, "chat/launch"), json={"model": model, "launch_kwargs": launch_kwargs})

        if response.status_code != 200:
            raise Exception(f"Failed to launch model. Status code: {response.status_code}, Response: {response.text}")

        print(f"Inference server for model {model} launched successfully")

    @staticmethod
    def kill(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        api_base = lm.kwargs["api_base"]

        response = requests.post(
            urljoin(api_base, "chat/kill"),
        )

        if response.status_code != 200:
            raise Exception(f"Failed to kill model. Status code: {response.status_code}, Response: {response.text}")

        print("Inference killed successfully")

    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("arbor:"):
            model = model[6:]
        return model

    @staticmethod
    def _add_provider_prefix(model: str) -> str:
        if not model.startswith("openai/arbor:"):
            model = "openai/arbor:" + model
        return model

    @staticmethod
    def _get_arbor_base_api():
        # TODO: We will delete this method once we start passing the LM object
        # to finetune.
        import dspy.settings as settings

        if not hasattr(settings, "arbor_api_base"):
            raise ValueError(
                "Arbor API base not set. Please set the `dspy.settings.arbor_api_base` to the URL for the Arbor server (e.g. 'http://localhost:8000/v1/')."
            )
        return dspy.settings.arbor_api_base

    @staticmethod
    def finetune(
        job: ArborTrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        # TODO: We want to re-factor finetune so that it takes in an LM.
        # Until then, we use the following to get the api information. The
        # following is a dummy call to ensure that dspy.settings.arbor_base_api
        # is set.
        ArborProvider._get_arbor_base_api()

        model = ArborProvider._remove_provider_prefix(model)

        print("[Arbor Provider] Validating the data format")
        ArborProvider.validate_data_format(train_data_format)

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
        ArborProvider.wait_for_job(job, train_kwargs)

        print("[Arbor Provider] Attempting to retrieve the trained model")
        model = ArborProvider.get_trained_model(job)
        print(f"[Arbor Provider] Model retrieved: {model}")

        return ArborProvider._add_provider_prefix(model)

    @staticmethod
    def does_job_exist(job_id: str, training_kwargs: dict[str, Any]) -> bool:
        try:
            original_base_url = openai.base_url
            openai.base_url = ArborProvider._get_arbor_base_api()
            openai.fine_tuning.jobs.retrieve(job_id)
            openai.base_url = original_base_url
            return True
        except Exception:
            return False

    @staticmethod
    def does_file_exist(file_id: str, training_kwargs: dict[str, Any]) -> bool:
        try:
            original_base_url = openai.base_url
            openai.base_url = ArborProvider._get_arbor_base_api()
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
    def get_training_status(job_id: str, training_kwargs: dict[str, Any]) -> TrainingStatus:
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
        openai.base_url = ArborProvider._get_arbor_base_api()
        provider_job = openai.fine_tuning.jobs.retrieve(job_id)
        openai.base_url = original_base_url

        provider_status = provider_job.status
        status = provider_status_to_training_status[provider_status]

        return status

    @staticmethod
    def validate_data_format(data_format: TrainDataFormat):
        supported_data_formats = [
            TrainDataFormat.CHAT,
            TrainDataFormat.COMPLETION,
            TrainDataFormat.GRPO_CHAT,
        ]

        if data_format not in supported_data_formats:
            err_msg = f"Arbor does not support the data format {data_format}."
            raise ValueError(err_msg)

    @staticmethod
    def upload_data(data_path: str, training_kwargs: dict[str, Any]) -> str:
        original_base_url = openai.base_url
        openai.base_url = ArborProvider._get_arbor_base_api()
        provider_file = openai.files.create(
            file=open(data_path, "rb"),
            purpose="fine-tune",
        )
        openai.base_url = original_base_url

        return provider_file.id

    @staticmethod
    def _start_remote_training(train_file_id: str, model: str, train_kwargs: dict[str, Any]) -> str:
        train_kwargs = train_kwargs or {}
        original_base_url = openai.base_url
        openai.base_url = ArborProvider._get_arbor_base_api()
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
        training_kwargs: dict[str, Any],
        poll_frequency: int = 20,
    ):
        done = False
        cur_event_id = None
        reported_estimated_time = False
        while not done:
            # Report estimated time if not already reported
            if not reported_estimated_time:
                original_base_url = openai.base_url
                openai.base_url = ArborProvider._get_arbor_base_api()
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
            openai.base_url = ArborProvider._get_arbor_base_api()
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
    def get_trained_model(job, training_kwargs: dict[str, Any]):
        status = job.status()
        if status != TrainingStatus.succeeded:
            err_msg = f"Job status is {status}."
            err_msg += f" Must be {TrainingStatus.succeeded} to retrieve model."
            raise Exception(err_msg)

        original_base_url = openai.base_url
        openai.base_url = ArborProvider._get_arbor_base_api()
        provider_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
        openai.base_url = original_base_url

        finetuned_model = provider_job.fine_tuned_model
        return finetuned_model
