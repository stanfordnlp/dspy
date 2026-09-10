import json
import logging
import math
import time
from datetime import datetime
from typing import Any

import openai
import pydantic

from dspy.clients.openai_format import response_format_to_responses
from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus, save_data

logger = logging.getLogger(__name__)


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

    def batch(self, lm, requests, *, cancel_event, timeout, poll_interval):
        """Run chat batches using the public OpenAI Files and Batches APIs.

        Separate credentials, headers, endpoints and models never share a job.
        No automatic submission retries: a lost create response may represent
        an accepted, billable job. Completed files are deleted best-effort;
        uncertain remote state is retained and logged for manual recovery.
        """
        from dspy.utils.exceptions import LMUnsupportedFeatureError

        groups = []
        for index, request in enumerate(requests):
            body = dict(request)
            model = body["model"]
            if lm.model_type != "chat" or not self.is_provider_model(model) or body.get("stream"):
                raise LMUnsupportedFeatureError("OpenAI batches require non-streaming OpenAI chat models", model=model)
            body["model"] = self._remove_provider_prefix(model)
            body.pop("rollout_id", None)
            response_format = body.get("response_format")
            if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
                schema_format = response_format_to_responses(response_format)
                schema_format.pop("type")
                body["response_format"] = {"type": "json_schema", "json_schema": {**schema_format, "strict": True}}
            options = {}
            for name in ("api_key", "organization", "project"):
                if body.get(name) is not None:
                    options[name] = body.pop(name)
                else:
                    body.pop(name, None)
            base_url = body.pop("base_url", None)
            api_base = body.pop("api_base", None)
            if base_url and api_base and base_url != api_base:
                raise ValueError("Conflicting base_url and api_base for OpenAI batch")
            if base_url or api_base:
                options["base_url"] = base_url or api_base
            headers = body.pop("headers", None)
            if headers is not None:
                options["default_headers"] = headers
            # Bound each HTTP operation, including cancellation and cleanup.
            request_timeout = body.pop("timeout", 60)
            if (
                not isinstance(request_timeout, (int, float))
                or not math.isfinite(request_timeout)
                or request_timeout <= 0
            ):
                raise ValueError("OpenAI batch request timeout must be a finite positive number of seconds")
            options["timeout"] = min(request_timeout, timeout, 60)
            options["max_retries"] = 0
            for group_options, group_model, entries in groups:
                if options == group_options and body["model"] == group_model:
                    entries.append((index, body))
                    break
            else:
                groups.append((options, body["model"], [(index, body)]))

        outcomes = [None] * len(requests)
        for options, _, entries in groups:
            try:
                with openai.OpenAI(**options) as client:
                    results = self._run_batch(client, entries, cancel_event, timeout, poll_interval)
                for index, result in results.items():
                    outcomes[index] = result
            except Exception as error:
                for index, _ in entries:
                    outcomes[index] = lm._wrap_litellm_exception(error)
        return outcomes

    @staticmethod
    def _run_batch(client, entries, cancel_event, timeout, poll_interval):
        from dspy.utils.exceptions import LMProviderError, LMTimeoutError

        endpoint = "/v1/chat/completions"
        payload = "\n".join(
            json.dumps({"custom_id": str(index), "method": "POST", "url": endpoint, "body": body})
            for index, body in entries
        ).encode()
        if len(entries) > 50000 or len(payload) > 200_000_000:
            raise ValueError("OpenAI batch exceeds 50,000 requests or 200 MB; reduce num_threads or input size")
        file_ids = set()
        job = None
        creating = False
        terminal = {"completed", "failed", "expired", "cancelled"}
        deadline = time.monotonic() + timeout

        def check_cancelled():
            if cancel_event.is_set():
                raise LMProviderError("Provider batch execution cancelled")
            if time.monotonic() >= deadline:
                raise LMTimeoutError("Timed out waiting for OpenAI batch completion")

        try:
            check_cancelled()
            uploaded = client.files.create(file=("dspy-batch.jsonl", payload, "application/jsonl"), purpose="batch")
            file_ids.add(uploaded.id)
            check_cancelled()
            creating = True
            job = client.batches.create(input_file_id=uploaded.id, endpoint=endpoint, completion_window="24h")
            creating = False
            logger.info("OpenAI batch submitted: %s", job.id)
            while job.status not in terminal:
                check_cancelled()
                if job.status not in {"validating", "in_progress", "finalizing", "cancelling"}:
                    raise LMProviderError(f"Unknown OpenAI batch status {job.status!r} for {job.id}")
                cancel_event.wait(min(poll_interval, max(0, deadline - time.monotonic())))
                check_cancelled()
                job = client.batches.retrieve(job.id)

            file_ids.update(file_id for file_id in (job.output_file_id, job.error_file_id) if file_id)
            if job.status == "failed":
                raise LMProviderError(f"OpenAI batch {job.id} failed: {job.errors}")
            rows = []
            for file_id in (job.output_file_id, job.error_file_id):
                if file_id:
                    check_cancelled()
                    rows.extend(json.loads(line) for line in client.files.content(file_id).text.splitlines() if line)
            return OpenAIProvider._parse_batch_results(entries, rows, job.id, job.status)
        finally:
            if job is not None and job.status not in terminal:
                try:
                    client.batches.cancel(job.id)
                except Exception:
                    logger.warning("Could not cancel OpenAI batch %s; check its remote status", job.id)
                logger.warning("OpenAI batch %s may still be running; retained files %s", job.id, sorted(file_ids))
            elif creating:
                logger.warning(
                    "OpenAI batch submission outcome unknown; check batches for input files %s", sorted(file_ids)
                )
            else:
                for file_id in file_ids:
                    try:
                        client.files.delete(file_id)
                    except Exception:
                        logger.warning("Could not delete OpenAI batch file %s", file_id)

    @staticmethod
    def _parse_batch_results(entries, rows, job_id, status):
        from dspy.clients._litellm import get_litellm
        from dspy.clients.lm import _lm_error_class_from_status
        from dspy.utils.exceptions import LMProviderError

        expected = {str(index): (index, body) for index, body in entries}
        results = {}
        for row in rows:
            custom_id = row.get("custom_id")
            if custom_id not in expected or custom_id in results:
                raise LMProviderError(f"Unknown or duplicate custom_id in OpenAI batch {job_id}")
            _, body = expected[custom_id]
            response = row.get("response") or {}
            error = row.get("error") or (response.get("body") or {}).get("error")
            code = response.get("status_code")
            if error or code != 200:
                error = error or {}
                results[custom_id] = _lm_error_class_from_status(code)(
                    error.get("message", "OpenAI batch request failed"),
                    model=body["model"],
                    provider="openai",
                    status=code,
                    provider_code=error.get("code"),
                    request_id=response.get("request_id"),
                )
            else:
                response_body = response.get("body")
                if not isinstance(response_body, dict) or not response_body.get("choices"):
                    raise LMProviderError(f"Malformed response in OpenAI batch {job_id}")
                results[custom_id] = get_litellm(feature="OpenAI batch responses").ModelResponse(**response_body)
        return {
            index: results.get(
                custom_id, LMProviderError(f"Missing result {custom_id} in OpenAI batch {job_id} ({status})")
            )
            for custom_id, (index, _) in expected.items()
        }

    @staticmethod
    def is_provider_model(model: str) -> bool:
        if model.startswith("openai/") or model.startswith("ft:"):
            # Although it looks strange, `ft:` is a unique identifier for openai finetuned models in litellm context:
            # https://github.com/BerriAI/litellm/blob/cd893134b7974d9f21477049a373b469fff747a5/litellm/utils.py#L4495
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
