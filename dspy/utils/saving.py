from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cloudpickle
import orjson

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.primitives.module import Module

logger = logging.getLogger(__name__)


def get_dependency_versions() -> dict[str, str]:
    import dspy

    cloudpickle_version = ".".join(cloudpickle.__version__.split(".")[:2])

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": dspy.__version__,
        "cloudpickle": cloudpickle_version,
    }


def _check_dependency_versions(metadata: dict[str, Any]) -> None:
    saved_dependency_versions = metadata.get("dependency_versions")
    if not isinstance(saved_dependency_versions, dict):
        raise ValueError("Invalid metadata.json: 'dependency_versions' must be a JSON object.")

    dependency_versions = get_dependency_versions()
    for key, saved_version in saved_dependency_versions.items():
        if key in dependency_versions and dependency_versions[key] != saved_version:
            logger.warning(
                f"There is a mismatch of {key} version between saved model and current environment. You saved with "
                f"`{key}=={saved_version}`, but now you have `{key}=={dependency_versions[key]}`. This might cause "
                "errors or performance downgrade on the loaded model, please consider loading the model in the same "
                "environment as the saving environment."
            )


def load(
    path: str | Path,
    allow_pickle: bool = False,
    *,
    allow_unsafe_lm_state: bool = False,
    safe_modules: Sequence[type[Module]] | None = None,
    tools: Mapping[str, Callable[..., Any] | Tool] | Sequence[Callable[..., Any] | Tool] | None = None,
) -> Module:
    """Load a saved DSPy program created with ``save(..., save_program=True)``.

    Programs saved with ``format="json"`` are loaded declaratively without
    ``cloudpickle`` and work safely with the default ``allow_pickle=False``.
    Legacy pickle programs require ``allow_pickle=True``.

    Args:
        path: Path to the saved program directory.
        allow_pickle: Whether to allow loading legacy cloudpickle programs.
            Only enable this when you trust the source of the files.
        allow_unsafe_lm_state: If True, preserves unsafe LM endpoint/credential
            keys (e.g., ``api_base``, ``base_url``, ``api_key``, ``model_list``)
            and allows importing custom LM classes when loading JSON programs.
        safe_modules: Optional allowlist of custom ``dspy.Module`` subclasses
            permitted during JSON program deserialization, matched by exact
            ``(cls.__module__, cls.__qualname__)``.
        tools: Optional mapping or sequence of callables / ``dspy.Tool``
            instances used to bind tools required by saved modules such as
            ``ReAct``, ``RLM``, or ``Flex``.

    Returns:
        The loaded ``dspy.Module`` instance.
    """
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"The path '{target_path}' does not exist.")

    with open(target_path / "metadata.json", "rb") as f:
        metadata = orjson.loads(f.read())
    if not isinstance(metadata, dict):
        raise ValueError("Invalid metadata.json: expected a JSON object.")

    saved_format = metadata.get("format", "pickle")
    if saved_format == "json":
        from dspy.utils.program_serialization import PROGRAM_SCHEMA_VERSION, deserialize_program

        schema_version = metadata.get("schema_version")
        if not (isinstance(schema_version, int) and not isinstance(schema_version, bool)) or (
            schema_version != PROGRAM_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported metadata.json schema_version: {schema_version!r}. Expected {PROGRAM_SCHEMA_VERSION}."
            )
        _check_dependency_versions(metadata)
        with open(target_path / "program.json", "rb") as f:
            program_data = orjson.loads(f.read())
        return deserialize_program(
            program_data,
            allow_unsafe_lm_state=allow_unsafe_lm_state,
            safe_modules=safe_modules,
            tools=tools,
        )

    if saved_format != "pickle":
        raise ValueError(f"Unsupported saved program format in metadata.json: {saved_format!r}.")

    if not allow_pickle:
        raise ValueError(
            "Loading with pickle is not allowed. Please set `allow_pickle=True` if you are sure you trust the source of the model."
        )

    _check_dependency_versions(metadata)
    with open(target_path / "program.pkl", "rb") as f:
        return cast("Module", cloudpickle.load(f))
