from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
from typing import List, Dict, Any, Optional


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


class FinetuneJob(Future):

    def __str__(self):
        return f"FinetuningJob(model={self.model}, config={self.config})"

    def __repr__(self):
        return str(self)

    # Subclasses should implement the following methods
    def cancel(self):
        """Cancel the finetuning job."""
        raise NotImplementedError("Method `get_status` is not implemented.")

    def get_status(self):
        """Cancel the finetuning job."""
        raise NotImplementedError("Method `get_status` is not implemented.")
