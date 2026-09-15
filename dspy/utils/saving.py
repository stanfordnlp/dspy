import copyreg
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import cloudpickle
import orjson

if TYPE_CHECKING:
    from dspy.primitives.module import Module

logger = logging.getLogger(__name__)

UNSAFE_PROGRAM_LM_CONFIG_KEYS = frozenset({"api_key", "api_base", "base_url", "model_list"})
AUTH_HEADER_NAMES = frozenset({"authorization", "proxy-authorization", "api-key", "x-api-key", "cookie", "set-cookie"})


def _sanitize_program_lm_config(config: dict) -> dict:
    """Remove known unsafe values from DSPy-owned LM configuration."""
    sanitized = {key: value for key, value in config.items() if key not in UNSAFE_PROGRAM_LM_CONFIG_KEYS}
    for header_field in ("headers", "extra_headers"):
        headers = sanitized.get(header_field)
        if isinstance(headers, dict):
            sanitized[header_field] = {
                key: value for key, value in headers.items() if str(key).lower() not in AUTH_HEADER_NAMES
            }
    return sanitized


def _reduce_program_lm(lm):
    state = lm.__dict__.copy()
    state["history"] = []
    state["kwargs"] = _sanitize_program_lm_config(state.get("kwargs") or {})
    return copyreg.__newobj__, (type(lm),), state


def _reduce_program_predict(predict):
    state = predict.__getstate__()
    state["config"] = _sanitize_program_lm_config(state.get("config") or {})
    return copyreg.__newobj__, (type(predict),), state


def dump_program(program: "Module", file: BinaryIO) -> None:
    """Serialize a program while sanitizing known DSPy-owned LM configuration."""
    from dspy.clients.lm import LM
    from dspy.predict.predict import Predict

    class ProgramPickler(cloudpickle.CloudPickler):
        dispatch_table = cloudpickle.CloudPickler.dispatch_table.new_child(
            {
                LM: _reduce_program_lm,
                Predict: _reduce_program_predict,
            }
        )

    ProgramPickler(file).dump(program)


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
