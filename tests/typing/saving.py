"""Static contracts for saving and loading utilities, checked by ty rather than executed by pytest.

Deliberately invalid calls use ignores to assert expected diagnostics. Unused-ignore
errors make these checks fail if the functions stop rejecting invalid calls.
"""

from pathlib import Path

from typing_extensions import assert_type

from dspy.primitives.module import Module
from dspy.utils.saving import get_dependency_versions, load


def check_saving_contracts() -> None:
    # 1. get_dependency_versions returns dict[str, str]
    assert_type(get_dependency_versions(), dict[str, str])
    get_dependency_versions("unexpected_arg")  # ty: ignore[too-many-positional-arguments]

    # 2. load accepts str or Path and returns a Module
    assert_type(load("path/to/model"), Module)
    assert_type(load(Path("path/to/model")), Module)
    assert_type(load("path/to/model", allow_pickle=True), Module)

    # 3. Invalid calls are rejected
    load(123)  # ty: ignore[invalid-argument-type]
    load()  # ty: ignore[missing-argument]
    load("path/to/model", allow_pickle="yes")  # ty: ignore[invalid-argument-type]
