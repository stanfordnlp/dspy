import importlib
import logging
import random
import re
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


def _import_class(class_path: str) -> type:
    """Import a class given its fully qualified path (e.g. 'dspy.predict.predict.Predict')."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _init_predict_skeleton(obj):
    """Set minimum viable attributes on a Predict created via ``object.__new__``.

    These placeholders let ``load_state()`` run correctly. All values
    are overwritten by ``load_state()`` except ``stage``, which is a
    random identifier regenerated each time.
    """
    from dspy.signatures.signature import Signature

    obj.stage = random.randbytes(8).hex()
    obj.config = {}
    obj.lm = None
    obj.traces = []
    obj.train = []
    obj.demos = []
    obj.signature = Signature("placeholder_input -> placeholder_output")


_BRACKET_RE = re.compile(r"^(\w+)\[(\d+|'[^']*')\]$")


def _set_by_path(obj, path: str, value):
    """Set an attribute on *obj* using a dotted/bracketed path.

    Supports paths like ``"cot.predict"``, ``"layers[0]"``, and
    ``"layers[0].predict"``.
    """
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        current = _resolve_part(current, part)
    _assign_part(current, parts[-1], value)


def _resolve_part(obj, part: str):
    m = _BRACKET_RE.match(part)
    if m:
        container = getattr(obj, m.group(1))
        idx = m.group(2)
        if idx.startswith("'"):
            return container[idx.strip("'")]
        return container[int(idx)]
    return getattr(obj, part)


def _assign_part(obj, part: str, value):
    m = _BRACKET_RE.match(part)
    if m:
        container_name = m.group(1)
        idx = m.group(2)
        if not hasattr(obj, container_name):
            # Create an empty list or dict depending on the index type.
            if idx.startswith("'"):
                setattr(obj, container_name, {})
            else:
                setattr(obj, container_name, [])
        container = getattr(obj, container_name)
        if isinstance(container, list):
            idx_int = int(idx)
            while len(container) <= idx_int:
                container.append(None)
            container[idx_int] = value
        else:
            container[idx.strip("'")] = value
    else:
        setattr(obj, part, value)


def _reconstruct_module(program_data: dict) -> "Module":
    """Reconstruct a module from safe-serialized program data.

    This creates module instances via ``object.__new__`` (bypassing
    ``__init__``), wires them into the saved tree structure, then calls
    ``load_state()`` to restore all parameter values.
    """
    from dspy.predict.predict import Predict
    from dspy.primitives.module import Module

    # Create the top-level module.
    top_class = _import_class(program_data["module_class"])
    obj = object.__new__(top_class)
    Module._base_init(obj)

    if issubclass(top_class, Predict):
        _init_predict_skeleton(obj)

    # Recreate each sub-module in tree order (parents before children).
    for entry in program_data["module_tree"]:
        sub_class = _import_class(entry["class"])
        sub_obj = object.__new__(sub_class)
        Module._base_init(sub_obj)

        if issubclass(sub_class, Predict):
            _init_predict_skeleton(sub_obj)

        _set_by_path(obj, entry["path"], sub_obj)

    # Restore parameter state.
    obj.load_state(program_data["state"])
    return obj


def load(path: str, allow_pickle: bool = False) -> "Module":
    """Load a saved DSPy model.

    When the saved directory uses the safe format (``save_program=True, safe=True``),
    no pickle is involved and ``allow_pickle`` is not required.  For legacy
    cloudpickle saves, ``allow_pickle=True`` must be passed explicitly.

    Args:
        path: Path to the saved model directory.
        allow_pickle: Whether to allow loading legacy cloudpickle models.

    Returns:
        The loaded model, a ``dspy.Module`` instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The path '{path}' does not exist.")

    with open(path / "metadata.json") as f:
        metadata = orjson.loads(f.read())

    # Version mismatch warnings.
    dependency_versions = get_dependency_versions()
    saved_dependency_versions = metadata["dependency_versions"]
    for key, saved_version in saved_dependency_versions.items():
        current = dependency_versions.get(key)
        if current is not None and current != saved_version:
            logger.warning(
                f"There is a mismatch of {key} version between saved model and current environment. You saved with "
                f"`{key}=={saved_version}`, but now you have `{key}=={current}`. This might cause "
                "errors or performance downgrade on the loaded model, please consider loading the model in the same "
                "environment as the saving environment."
            )

    # Safe JSON format: no pickle needed.
    if metadata.get("format") == "safe_v1":
        with open(path / "program.json") as f:
            program_data = orjson.loads(f.read())
        return _reconstruct_module(program_data)

    # Legacy cloudpickle format.
    if not allow_pickle:
        raise ValueError(
            "Loading with pickle is not allowed. Please set `allow_pickle=True` if you are sure "
            "you trust the source of the model."
        )

    with open(path / "program.pkl", "rb") as f:
        return cloudpickle.load(f)
