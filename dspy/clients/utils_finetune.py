import os
from enum import Enum
from typing import Any, Literal, TypedDict

import ujson

import dspy
from dspy.adapters.base import Adapter
from dspy.utils.caching import DSPY_CACHEDIR


class TrainingStatus(str, Enum):
    not_started = "not_started"
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainDataFormat(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    GRPO_CHAT = "grpo_chat"


class Message(TypedDict):
    role: Literal["user"] | Literal["assistant"] | Literal["system"]
    content: str


class MessageAssistant(TypedDict):
    role: Literal["assistant"]
    content: str


class GRPOChatData(TypedDict):
    messages: list[Message]
    completion: MessageAssistant
    reward: float


GRPOGroup = list[GRPOChatData]


def infer_data_format(adapter: Adapter) -> str:
    if isinstance(adapter, dspy.ChatAdapter):
        return TrainDataFormat.CHAT
    raise ValueError(f"Could not infer the data format for: {adapter}")


def get_finetune_directory() -> str:
    default_finetunedir = os.path.join(DSPY_CACHEDIR, "finetune")
    finetune_dir = os.environ.get("DSPY_FINETUNEDIR") or default_finetunedir
    finetune_dir = os.path.abspath(finetune_dir)
    os.makedirs(finetune_dir, exist_ok=True)
    return finetune_dir


def write_lines(file_path, data):
    with open(file_path, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")


def save_data(
    data: list[dict[str, Any]],
) -> str:
    from datasets.fingerprint import Hasher

    # Assign a unique name to the file based on the data hash
    hash = Hasher.hash(data)
    file_name = f"{hash}.jsonl"

    finetune_dir = get_finetune_directory()
    file_path = os.path.join(finetune_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")
    return file_path


def validate_data_format(
    data: list[dict[str, Any]],
    data_format: TrainDataFormat,
):
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
