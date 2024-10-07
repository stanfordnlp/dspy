from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import ujson

from dspy import logger


# TODO: Move to a centralized location with all the other environment variables
# Set the directory to save the fine-tuned models
def get_finetune_directory() -> str:
    """Get the directory to save the fine-tuned models."""
    alternative_path = os.path.join(Path.home(), '.dspy_finetune')
    # TODO: Should the parent directory be different than those used for
    # inference
    return os.environ.get('DSPY_FINETUNEDIR') or alternative_path


FINETUNE_DIRECTORY = get_finetune_directory()


class TrainingMethod(str, Enum):
    """Enum class for training methods.
    
    When comparing enums, Python checks for object IDs, which means that the
    enums can't be compared directly. Subclassing the Enum class along with the
    str class allows for direct comparison of the enums.
    """
    SFT = "SFT"
    Preference = "Preference"


"""Dictionary mapping training methods to the data keys they require."""
TRAINING_METHOD_TO_DATA_KEYS = {
    TrainingMethod.SFT: ["prompt", "completion"],
    TrainingMethod.Preference: ["prompt", "chosen", "rejected"],
}


class FinetuneJob(Future):

    def __init__(self,
        model: str,
        message_completion_pairs: List[Dict[str, str]],
        train_kwargs: Optional[Dict[str, Any]]=None,
    ):
        self.model = model
        self.message_completion_pairs = message_completion_pairs
        self.train_kwargs: Dict[str, Any] = train_kwargs or {}
        super().__init__()
    
    def get_kwargs(self):
        return dict(
            model=self.model,
            message_completion_pairs=self.message_completion_pairs,
            train_kwargs=self.train_kwargs,
        )

    def __str__(self):
        return f"FinetuningJob({self.to_dict()})"

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


def validate_training_data(
        data: List[Dict[str, Any]],
        training_method: TrainingMethod
    ) -> Optional[AssertionError]:
    """Validate the training data based on the training method."""
    # Get the required data keys for the training method
    required_keys = TRAINING_METHOD_TO_DATA_KEYS[training_method]

    # Check if the training data has the required keys
    for ind, data_dict in enumerate(data):
        err_msg = f"The datapoint at index {ind} is missing the keys required for {training_method} training."
        err_msg = f"\n    Expected: {required_keys}"
        err_msg = f"\n    Found: {data_dict.keys()}"
        assert all([key in data_dict for key in required_keys]), err_msg


def save_data(data: Dict[str, str]) -> str:
    # TODO: Assign different names based on the data hash
    finetune_parent_dir = get_finetune_directory()
    data_path = os.path.join(finetune_parent_dir, "openai", "train.json")
    with open(data_path, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")
    return data_path
