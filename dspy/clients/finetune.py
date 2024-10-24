import os
from abc import abstractmethod
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dspy.utils.logging import logger

import ujson
from datasets.fingerprint import Hasher


def get_finetune_directory() -> str:
    """Get the directory to save the fine-tuned models."""
    # TODO: Move to a centralized location with all the other env variables
    dspy_cachedir = os.environ.get("DSPY_CACHEDIR")
    dspy_cachedir = dspy_cachedir or os.path.join(Path.home(), ".dspy_cache")
    finetune_dir = os.path.join(dspy_cachedir, "finetune")
    finetune_dir = os.path.abspath(finetune_dir)
    return finetune_dir


FINETUNE_DIRECTORY = get_finetune_directory()


class TrainingMethod(str, Enum):
    """Enum class for training methods.

    When comparing enums, Python checks for object IDs, which means that the
    enums can't be compared directly. Subclassing the Enum class along with the
    str class allows for direct comparison of the enums.
    """

    SFT = "SFT"
    Preference = "Preference"


class TrainingStatus(str, Enum):
    """Enum class for remote training status."""

    not_started = "not_started"
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


"""Dictionary mapping training methods to the data keys they require."""
TRAINING_METHOD_TO_DATA_KEYS = {
    TrainingMethod.SFT: ["prompt", "completion"],
    TrainingMethod.Preference: ["prompt", "chosen", "rejected"],
}


class FinetuneJob(Future):
    def __init__(
        self,
        model: str,
        train_data: List[Dict[str, Any]],
        train_kwargs: Optional[Dict[str, Any]] = None,
        train_method: TrainingMethod = TrainingMethod.SFT,
        provider: str = "openai",
    ):
        self.model = model
        self.train_data = train_data
        self.train_kwargs: Dict[str, Any] = train_kwargs or {}
        self.train_method = train_method
        self.provider = provider
        super().__init__()

    def get_kwargs(self):
        return dict(
            model=self.model,
            train_data=self.train_data,
            train_kwargs=self.train_kwargs,
            train_method=self.train_method,
            provider=self.provider,
        )

    def __repr__(self):
        return str(self)

    # Subclasses should override the cancel method to cancel the finetune job;
    # then call the super's cancel method so that the future can be cancelled.
    def cancel(self):
        """Cancel the finetune job."""
        super().cancel()

    @abstractmethod
    def status(self):
        """Get the status of the finetune job."""
        raise NotImplementedError("Method `status` is not implemented.")


def validate_finetune_data(data: List[Dict[str, Any]], train_method: TrainingMethod):
    """Validate the finetune data based on the training method."""
    # Get the required data keys for the training method
    required_keys = TRAINING_METHOD_TO_DATA_KEYS[train_method]

    # Check if the training data has the required keys
    for ind, data_dict in enumerate(data):
        if not all([key in data_dict for key in required_keys]):
            raise ValueError(
                f"The datapoint at index {ind} is missing the keys required for {train_method} training. Expected: "
                f"{required_keys}, Found: {data_dict.keys()}"
            )


def save_data(
    data: List[Dict[str, Any]],
    provider_name: Optional[str] = None,
) -> str:
    """Save the fine-tuning data to a file."""
    logger.info("[Finetune] Converting data to JSONL format...")
    # Construct the file name based on the data hash
    hash = Hasher.hash(data)
    file_name = f"{hash}.jsonl"
    file_name = f"{provider_name}_{file_name}" if provider_name else file_name

    # Find the directory to save the fine-tuning data
    finetune_parent_dir = get_finetune_directory()
    os.makedirs(finetune_parent_dir, exist_ok=True)

    # Save the data to a file
    file_path = os.path.join(finetune_parent_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")
    return file_path
