import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import orjson

if TYPE_CHECKING:
    from dspy.primitives.module import Module

logger = logging.getLogger(__name__)


def get_dependency_versions():
    import dspy

    cloudpickle_version = ".".join(cloudpickle.__version__.split(".")[:2])

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": dspy.__version__,
        "cloudpickle": cloudpickle_version,
    }


def load(path: str, allow_pickle: bool = False) -> "Module":
    """Load saved DSPy model.

    This method is used to load a saved DSPy model with `save_program=True`, i.e., the model is saved with cloudpickle.

    Args:
        path (str): Path to the saved model.
        allow_pickle (bool): Whether to allow loading the model with pickle. This is dangerous and should only be used if you are sure you trust the source of the model.

    Returns:
        The loaded model, a `dspy.Module` instance.
    """
    if not allow_pickle:
        raise ValueError("Loading with pickle is not allowed. Please set `allow_pickle=True` if you are sure you trust the source of the model.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The path '{path}' does not exist.")

    with open(path / "metadata.json") as f:
        metadata = orjson.loads(f.read())

    dependency_versions = get_dependency_versions()
    saved_dependency_versions = metadata["dependency_versions"]
    for key, saved_version in saved_dependency_versions.items():
        if dependency_versions[key] != saved_version:
            logger.warning(
                f"There is a mismatch of {key} version between saved model and current environment. You saved with "
                f"`{key}=={saved_version}`, but now you have `{key}=={dependency_versions[key]}`. This might cause "
                "errors or performance downgrade on the loaded model, please consider loading the model in the same "
                "environment as the saving environment."
            )

    with open(path / "program.pkl", "rb") as f:
        return cloudpickle.load(f)
