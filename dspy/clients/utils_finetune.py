import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import orjson

import dspy
from dspy.utils.caching import DSPY_CACHEDIR

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter


class TrainingStatus(str, Enum):
    """Enum representing the status of a fine-tuning training job.

    Attributes:
        not_started: The training job has not started yet.
        pending: The training job is pending.
        running: The training job is currently running.
        succeeded: The training job has completed successfully.
        failed: The training job has failed.
        cancelled: The training job has been cancelled.
    """

    not_started = "not_started"
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainDataFormat(str, Enum):
    """Enum representing supported training data formats.

    Attributes:
        CHAT: Chat-style format with a list of messages.
        COMPLETION: Completion-style format with prompt and completion fields.
        GRPO_CHAT: GRPO chat format with messages, completion, and reward.
    """

    CHAT = "chat"
    COMPLETION = "completion"
    GRPO_CHAT = "grpo_chat"


class Message(TypedDict):
    """A typed dictionary representing a single chat message.

    Attributes:
        role: The role of the message sender. One of "user", "assistant", or "system".
        content: The text content of the message.
    """

    role: Literal["user"] | Literal["assistant"] | Literal["system"]
    content: str


class MessageAssistant(TypedDict):
    """A typed dictionary representing an assistant message.

    Attributes:
        role: Always "assistant".
        content: The text content of the assistant's message.
    """

    role: Literal["assistant"]
    content: str


class GRPOChatData(TypedDict):
    """A typed dictionary representing a single GRPO training example.

    Attributes:
        messages: A list of chat messages forming the conversation context.
        completion: The assistant's response to the conversation.
        reward: The reward signal associated with this completion.
    """

    messages: list[Message]
    completion: MessageAssistant
    reward: float


class GRPOGroup(TypedDict):
    """A typed dictionary representing a group of GRPO training examples.

    Attributes:
        batch_id: An optional identifier for the batch this group belongs to.
        group: A list of GRPO training examples in this group.
    """

    batch_id: int | None
    group: list[GRPOChatData]


class GRPOStatus(TypedDict):
    """A typed dictionary representing the status of a GRPO fine-tuning job.

    Attributes:
        job_id: The unique identifier for the fine-tuning job.
        status: The current status of the job, if available.
        current_model: The model currently being fine-tuned.
        checkpoints: A mapping of checkpoint names to their storage paths.
        last_checkpoint: The path to the most recent checkpoint, if any.
        pending_batch_ids: A list of batch IDs that are yet to be processed.
    """

    job_id: str
    status: str | None
    current_model: str
    checkpoints: dict[str, str]
    last_checkpoint: str | None
    pending_batch_ids: list[int]


def infer_data_format(adapter: "Adapter") -> str:
    """Infer the training data format from the given adapter.

    Args:
        adapter: The adapter instance used to determine the data format.

    Returns:
        A `TrainDataFormat` value representing the inferred data format.

    Raises:
        ValueError: If the data format cannot be inferred from the adapter.

    Example:
        >>> import dspy
        >>> adapter = dspy.ChatAdapter()
        >>> infer_data_format(adapter)
        <TrainDataFormat.CHAT: 'chat'>
    """
    if isinstance(adapter, dspy.ChatAdapter):
        return TrainDataFormat.CHAT
    raise ValueError(f"Could not infer the data format for: {adapter}")


def get_finetune_directory() -> str:
    """Return the directory path used for storing fine-tuning data and logs.

    The directory is determined by the ``DSPY_FINETUNEDIR`` environment variable.
    If not set, defaults to a ``finetune`` subdirectory within the DSPy cache directory.
    The directory is created if it does not already exist.

    Returns:
        The absolute path to the fine-tuning directory.

    Example:
        >>> directory = get_finetune_directory()
        >>> import os
        >>> os.path.isdir(directory)
        True
    """
    default_finetunedir = os.path.join(DSPY_CACHEDIR, "finetune")
    finetune_dir = os.environ.get("DSPY_FINETUNEDIR") or default_finetunedir
    finetune_dir = os.path.abspath(finetune_dir)
    os.makedirs(finetune_dir, exist_ok=True)
    return finetune_dir


def write_lines(file_path: str, data: list[Any]) -> None:
    """Write a list of objects to a file as newline-delimited JSON (JSONL).

    Each item in ``data`` is serialized to JSON and written as a separate line.

    Args:
        file_path: The path to the output file.
        data: A list of JSON-serializable objects to write.

    Example:
        >>> write_lines("/tmp/output.jsonl", [{"key": "value"}, {"key": "value2"}])
    """
    with open(file_path, "wb") as f:
        for item in data:
            f.write(orjson.dumps(item) + b"\n")


def save_data(
    data: list[dict[str, Any]],
) -> str:
    """Save a list of data dictionaries to a JSONL file in the fine-tuning directory.

    The file is named using a hash of the data content, ensuring that identical
    datasets map to the same file.

    Args:
        data: A list of dictionaries to serialize and save.

    Returns:
        The absolute path to the saved JSONL file.

    Example:
        >>> path = save_data([{"prompt": "Hello", "completion": "World"}])
        >>> import os
        >>> os.path.exists(path)
        True
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
    """Validate that a dataset conforms to the expected training data format.

    Checks each item in ``data`` against the rules defined for the given
    ``data_format``. If errors are found, they are logged to a file in the
    fine-tuning directory and a ``ValueError`` is raised.

    Args:
        data: A list of data dictionaries to validate.
        data_format: The expected format of the training data.

    Raises:
        AssertionError: If ``data_format`` is not a supported format.
        ValueError: If ``data`` is not a list, or if any data items contain
            format errors. The error message includes the path to a log file
            with detailed error information.

    Example:
        >>> validate_data_format(
        ...     [{"messages": [{"role": "user", "content": "Hi"}]}],
        ...     TrainDataFormat.CHAT,
        ... )
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
    """Check a completion-format data dictionary for format errors.

    Validates that ``data_dict`` contains exactly the keys ``"prompt"`` and
    ``"completion"``, both with string values.

    Args:
        data_dict: A dictionary representing a single completion-format training example.

    Returns:
        An error message string if a format error is found, or ``None`` if the
        data is valid.

    Example:
        >>> find_data_errors_completion({"prompt": "Hello", "completion": "World"})
        >>> find_data_errors_completion({"prompt": "Hello"})
        "Expected Keys: ['completion', 'prompt']; Found Keys: ['prompt']"
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
    """Check a chat-format data dictionary for format errors.

    Validates that ``messages`` contains a ``"messages"`` key with a list value,
    and that each message in the list passes validation.

    Args:
        messages: A dictionary representing a single chat-format training example.
            Expected to have a ``"messages"`` key containing a list of message dicts.

    Returns:
        An error message string if a format error is found, or ``None`` if the
        data is valid.

    Example:
        >>> find_data_error_chat({"messages": [{"role": "user", "content": "Hi"}]})
        >>> find_data_error_chat({"wrong_key": []})
        "Expected Keys: ['messages']; Found Keys: ['wrong_key']"
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
    """Check a single chat message dictionary for format errors.

    Validates that ``message`` contains exactly the keys ``"role"`` and
    ``"content"``, that the role is one of ``"assistant"``, ``"system"``, or
    ``"user"``, and that the content is a string.

    Args:
        message: A dictionary representing a single chat message.

    Returns:
        An error message string if a format error is found, or ``None`` if the
        message is valid.

    Example:
        >>> find_data_error_chat_message({"role": "user", "content": "Hello"})
        >>> find_data_error_chat_message({"role": "unknown", "content": "Hi"})
        "Expected Roles: ['assistant', 'system', 'user']; Found Role: unknown"
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
