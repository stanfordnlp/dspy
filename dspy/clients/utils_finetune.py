"""Utilities for fine-tuning data preparation, validation, and persistence.

This module provides helpers used by the DSPy fine-tuning pipeline to infer
data formats, save training data to disk, and validate that datasets conform
to expected schemas (chat or completion format).
"""

import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import orjson

import dspy
from dspy.utils.caching import DSPY_CACHEDIR

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter


class TrainingStatus(str, Enum):
    """Status of a fine-tuning training job."""

    not_started = "not_started"
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainDataFormat(str, Enum):
    """Supported training data formats for fine-tuning."""

    CHAT = "chat"
    COMPLETION = "completion"
    GRPO_CHAT = "grpo_chat"


class Message(TypedDict):
    """A single chat message with a role and content."""

    role: Literal["user"] | Literal["assistant"] | Literal["system"]
    content: str


class MessageAssistant(TypedDict):
    """A chat message restricted to the assistant role."""

    role: Literal["assistant"]
    content: str


class GRPOChatData(TypedDict):
    """A single GRPO training example with messages, completion, and reward."""

    messages: list[Message]
    completion: MessageAssistant
    reward: float


class GRPOGroup(TypedDict):
    """A batch group of GRPO training examples."""

    batch_id: int | None
    group: list[GRPOChatData]

class GRPOStatus(TypedDict):
    """Status of a GRPO fine-tuning job, including checkpoint information."""

    job_id: str
    status: str | None = None
    current_model: str
    checkpoints: dict[str, str]
    last_checkpoint: str | None = None
    pending_batch_ids: list[int] = []


def infer_data_format(adapter: "Adapter") -> str:
    """Infer the training data format from the given adapter type.

    Args:
        adapter: A DSPy adapter instance (e.g. ``ChatAdapter``).

    Returns:
        The corresponding ``TrainDataFormat`` value.

    Raises:
        ValueError: If the adapter type is not recognized.
    """
    if isinstance(adapter, dspy.ChatAdapter):
        return TrainDataFormat.CHAT
    raise ValueError(f"Could not infer the data format for: {adapter}")


def get_finetune_directory() -> str:
    """Return the directory path used for storing fine-tuning artifacts.

    Uses the ``DSPY_FINETUNEDIR`` environment variable if set, otherwise
    defaults to ``<DSPY_CACHEDIR>/finetune``. Creates the directory if it
    does not exist.

    Returns:
        Absolute path to the fine-tuning directory.
    """
    default_finetunedir = os.path.join(DSPY_CACHEDIR, "finetune")
    finetune_dir = os.environ.get("DSPY_FINETUNEDIR") or default_finetunedir
    finetune_dir = os.path.abspath(finetune_dir)
    os.makedirs(finetune_dir, exist_ok=True)
    return finetune_dir


def write_lines(file_path: str, data: list[Any]) -> None:
    """Write a list of JSON-serializable objects as newline-delimited JSON.

    Args:
        file_path: Destination file path.
        data: Items to serialize; each is written as a single JSON line.
    """
    with open(file_path, "wb") as f:
        for item in data:
            f.write(orjson.dumps(item) + b"\n")


def save_data(
    data: list[dict[str, Any]],
) -> str:
    """Persist training data to a JSONL file named by its content hash.

    The file is saved in the fine-tuning directory returned by
    :func:`get_finetune_directory`.

    Args:
        data: Training examples to save.

    Returns:
        Absolute path to the written JSONL file.
    """
    from dspy.utils.hasher import Hasher

    # Assign a unique name to the file based on the data hash
    hash = Hasher.hash(data)
    file_name = f"{hash}.jsonl"

    finetune_dir = get_finetune_directory()
    file_path = os.path.join(finetune_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "wb") as f:
        for item in data:
            f.write(orjson.dumps(item) + b"\n")
    return file_path


def validate_data_format(
    data: list[dict[str, Any]],
    data_format: TrainDataFormat,
) -> None:
    """Validate that every item in *data* conforms to *data_format*.

    Checks each dictionary in the list against the expected schema for the
    given format. If any errors are found, they are logged to a file and a
    ``ValueError`` is raised.

    Args:
        data: List of training examples to validate.
        data_format: The expected format (``CHAT`` or ``COMPLETION``).

    Raises:
        AssertionError: If *data_format* is not supported.
        ValueError: If *data* is not a list or contains invalid entries.
    """
    find_err_funcs = {
        TrainDataFormat.CHAT: find_data_error_chat,
        TrainDataFormat.COMPLETION: find_data_errors_completion,
    }
    err = f"Data format {data_format} is not supported."
    assert data_format in find_err_funcs, err
    find_err_func = find_err_funcs[data_format]

    if not isinstance(data, list):
        err = f"Data is not a list. Found data type: {type(data)}"
        raise ValueError(err)

    data_dict_errors = []
    for ind, data_dict in enumerate(data):
        err = f"Not a dictionary -- found data type: {type(data_dict)}"
        if isinstance(data_dict, dict):
            err = find_err_func(data_dict)
        if err:
            err_dict = {"index": ind, "error": err}
            data_dict_errors.append(err_dict)

    if data_dict_errors:
        finetune_dir = get_finetune_directory()
        log_path = os.path.join(finetune_dir, "data_format_errors.log")
        log_path = os.path.abspath(log_path)
        write_lines(log_path, data_dict_errors)

        err = f"Data format errors found.  For more details, see the log file: {log_path}"
        raise ValueError(err)


def find_data_errors_completion(data_dict: dict[str, str]) -> str | None:
    """Check a single completion-format example for schema errors.

    Args:
        data_dict: A dictionary expected to have ``prompt`` and ``completion`` keys.

    Returns:
        An error message string if validation fails, or ``None`` if valid.
    """
    keys = ["prompt", "completion"]

    assert isinstance(data_dict, dict)
    expected_keys = sorted(keys)
    found_keys = sorted(data_dict.keys())
    if set(expected_keys) != set(found_keys):
        return f"Expected Keys: {expected_keys}; Found Keys: {found_keys}"

    for key in keys:
        if not isinstance(data_dict[key], str):
            return f"Expected `{key}` to be of type `str`. Found: {type(data_dict[key])}"


# Following functions are modified from the OpenAI cookbook:
# https://cookbook.openai.com/examples/chat_finetuning_data_prep
def find_data_error_chat(messages: dict[str, Any]) -> str | None:
    """Check a single chat-format example for schema errors.

    Validates that the dictionary has a ``messages`` key containing a list
    of well-formed message dictionaries.

    Args:
        messages: A dictionary expected to have a ``messages`` key.

    Returns:
        An error message string if validation fails, or ``None`` if valid.
    """
    assert isinstance(messages, dict)

    expected_keys = ["messages"]
    found_keys = sorted(messages.keys())
    if set(expected_keys) != set(found_keys):
        return f"Expected Keys: {expected_keys}; Found Keys: {found_keys}"

    if not isinstance(messages["messages"], list):
        return f"The value of the `messages` key should be a list instance. Found: {type(messages['messages'])}"

    for ind, message in enumerate(messages["messages"]):
        err = find_data_error_chat_message(message)
        if err:
            return f"Error in message at index {ind}: {err}"


def find_data_error_chat_message(message: dict[str, Any]) -> str | None:
    """Check a single chat message dictionary for schema errors.

    Args:
        message: A dictionary expected to have ``role`` and ``content`` keys.

    Returns:
        An error message string if validation fails, or ``None`` if valid.
    """
    assert isinstance(message, dict)

    message_keys = sorted(["role", "content"])
    found_keys = sorted(message.keys())
    if set(message_keys) != set(found_keys):
        return f"Expected Keys: {message_keys}; Found Keys: {found_keys}"

    expected_roles = sorted(["assistant", "system", "user"])
    found_role = message["role"]
    if found_role not in expected_roles:
        return f"Expected Roles: {expected_roles}; Found Role: {found_role}"

    if not isinstance(message["content"], str):
        return f"Expected Content Type: `str`; Found Content Type: {type(message['content'])}"
