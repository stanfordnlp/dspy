import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import openai

from dspy.clients.finetune import (
    FinetuneJob,
    TrainingMethod,
    TrainingStatus,
    save_data,
    validate_finetune_data,
)
from dspy.utils.logging import logger

# Provider name
PROVIDER_OPENAI = "openai"


def is_openai_model(model: str) -> bool:
    """Check if the model is an OpenAI model."""
    # Filter the provider_prefix, if exists
    provider_prefix = f"{PROVIDER_OPENAI}/"
    if model.startswith(provider_prefix):
        model = model[len(provider_prefix) :]

    client = openai.OpenAI()
    valid_model_names = [model.id for model in client.models.list().data]
    # Check if the model is a base OpenAI model
    if model in valid_model_names:
        return True

    # Check if the model is a fine-tuned OpneAI model. Fine-tuned OpenAI models
    # have the prefix "ft:<BASE_MODEL_NAME>:", followed by a string specifying
    # the fine-tuned model. The following RegEx pattern is used to match the
    # base model name.
    # TODO: This part can be updated to match the actual fine-tuned model names
    # by making a call to the OpenAI API to be more exact, but this might
    # require an API key with the right permissions.
    match = re.match(r"ft:([^:]+):", model)
    if match and match.group(1) in valid_model_names:
        return True

    return False


class FinetuneJobOpenAI(FinetuneJob):
    def __init__(self, *args, **kwargs):
        self.provider_file_id = None  # TODO: Can we get this using the job_id?
        self.provider_job_id = None
        super().__init__(*args, **kwargs)

    def cancel(self):
        # Cancel the provider job
        if _does_job_exist(self.provider_job_id):
            status = _get_training_status(self.provider_job_id)
            if _is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            openai.fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None

        # Delete the provider file
        # TODO: Should there be a separate clean method?
        if self.provider_file_id is not None:
            if _does_file_exist(self.provider_file_id):
                openai.files.delete(self.provider_file_id)
            self.provider_file_id = None

        # Call the super's cancel method after the custom cancellation logic
        super().cancel()

    def status(self) -> TrainingStatus:
        status = _get_training_status(self.provider_job_id)
        return status


def finetune_openai(
    job: FinetuneJobOpenAI,
    model: str,
    train_data: List[Dict[str, Any]],
    train_kwargs: Optional[Dict[str, Any]] = None,
    train_method: TrainingMethod = TrainingMethod.SFT,
) -> str:
    train_kwargs = train_kwargs or {}
    train_method = TrainingMethod.SFT  # Note: This could be an argument; ignoring method

    # Validate train data and method
    logger.info("[Finetune] Validating the formatting of the data")
    _validate_data(train_data, train_method)
    logger.info("[Finetune] Done!")

    # Convert to the OpenAI format
    logger.info("[Finetune] Converting the data to the OpenAI format")
    # TODO: Should we use the system prompt?
    train_data = _convert_data(train_data)
    logger.info("[Finetune] Done!")

    # Save to a file
    logger.info("[Finetune] Saving the data to a file")
    data_path = save_data(train_data, provider_name=PROVIDER_OPENAI)
    logger.info("[Finetune] Done!")

    # Upload the data to the cloud
    logger.info("[Finetune] Uploading the data to the provider")
    provider_file_id = _upload_data(data_path)
    job.provider_file_id = provider_file_id
    logger.info("[Finetune] Done!")

    logger.info("[Finetune] Start remote training")
    # We utilize model and train_kwargs here
    provider_job_id = _start_remote_training(
        train_file_id=job.provider_file_id,
        model=model,
        train_kwargs=train_kwargs,
    )
    job.provider_job_id = provider_job_id
    # job.provider_job_id = "ftjob-ZdEL1mUDk0dwdDuZJQOng8Vv"
    logger.info("[Finetune] Done!")

    logger.info("[Finetune] Wait for training to complete")
    # TODO: Would it be possible to stream the logs?
    _wait_for_job(job)
    logger.info("[Finetune] Done!")

    logger.info("[Finetune] Get trained model if the run was a success")
    model = _get_trained_model(job)
    logger.info("[Finetune] Done!")

    return model


_SUPPORTED_TRAINING_METHODS = [
    TrainingMethod.SFT,
]


def _get_training_status(job_id: str) -> Union[TrainingStatus, Exception]:
    # TODO: Should this type be shared across all fine-tune clients?
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
    assert _does_job_exist(job_id), err_msg

    # Retrieve the provider's job and report the status
    provider_job = openai.fine_tuning.jobs.retrieve(job_id)
    provider_status = provider_job.status
    status = provider_status_to_training_status[provider_status]

    return status


def _does_job_exist(job_id: str) -> bool:
    try:
        # TODO: Error handling is vague
        openai.fine_tuning.jobs.retrieve(job_id)
        return True
    except Exception:
        return False


def _does_file_exist(file_id: str) -> bool:
    try:
        # TODO: Error handling is vague
        openai.files.retrieve(file_id)
        return True
    except Exception:
        return False


def _is_terminal_training_status(status: TrainingStatus) -> bool:
    return status in [
        TrainingStatus.succeeded,
        TrainingStatus.failed,
        TrainingStatus.cancelled,
    ]


def _validate_data(data: Dict[str, str], train_method: TrainingMethod) -> Optional[Exception]:
    # Check if this train method is supported
    if train_method not in _SUPPORTED_TRAINING_METHODS:
        err_msg = f"OpenAI does not support the training method {train_method}."
        raise ValueError(err_msg)

    validate_finetune_data(data, train_method)


def _convert_data(
    data: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> Union[List[Dict[str, Any]], Exception]:
    # Item-wise conversion function
    def _row_converter(d):
        messages = [{"role": "user", "content": d["prompt"]}, {"role": "assistant", "content": d["completion"]}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        messages_dict = {"messages": messages}
        return messages_dict

    # Convert the data to the OpenAI format; validate the converted data
    converted_data = list(map(_row_converter, data))
    openai_data_validation(converted_data)
    return converted_data


def _upload_data(data_path: str) -> str:
    # Upload the data to the provider
    provider_file = openai.files.create(
        file=open(data_path, "rb"),
        purpose="fine-tune",
    )
    return provider_file.id


def _start_remote_training(train_file_id: str, model: id, train_kwargs: Optional[Dict[str, Any]] = None) -> str:
    train_kwargs = train_kwargs or {}
    provider_job = openai.fine_tuning.jobs.create(
        model=model,
        training_file=train_file_id,
        hyperparameters=train_kwargs,
    )
    return provider_job.id


def _wait_for_job(
    job: FinetuneJobOpenAI,
    poll_frequency: int = 60,
):
    while not _is_terminal_training_status(job.status()):
        time.sleep(poll_frequency)


def _get_trained_model(job):
    status = job.status()
    if status != TrainingStatus.succeeded:
        err_msg = f"Job status is {status}."
        err_msg += f" Must be {TrainingStatus.succeeded} to retrieve the model."
        logger.error(err_msg)
        raise Exception(err_msg)

    provider_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
    finetuned_model = provider_job.fine_tuned_model
    return finetuned_model


# Adapted from https://cookbook.openai.com/examples/chat_finetuning_data_prep
def openai_data_validation(dataset: List[dict[str, Any]]):
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    # Raise an error if there are any format errors
    if format_errors:
        err_msg = "Found errors in the dataset format using the OpenAI API."
        err_msg += " Here are the number of datapoints for each error type:"
        for k, v in format_errors.items():
            err_msg += "\n    {k}: {v}"
        raise ValueError(err_msg)


def check_message_lengths(dataset: List[dict[str, Any]]) -> list[int]:
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    n_too_long = sum([length > 16385 for length in convo_lens])

    if n_too_long > 0:
        logger.info(
            f"There are {n_too_long} examples that may be over the 16,385 token limit, they will be truncated during fine-tuning."
        )

    if n_missing_system > 0:
        logger.info(f"There are {n_missing_system} examples that are missing a system message.")

    if n_missing_user > 0:
        logger.info(f"There are {n_missing_user} examples that are missing a user message.")

    return convo_lens


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens
