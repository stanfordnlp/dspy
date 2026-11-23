"""Static contracts for scoped by-value serialization, checked by ty.

Unused-ignore diagnostics ensure invalid calls remain rejected.
"""

from collections.abc import Iterable
from types import ModuleType

from typing_extensions import assert_type

from dspy.dsp.utils.settings import Settings
from dspy.utils.pickle_by_value import serialize_by_value


def check_serialize_by_value_contracts(module: ModuleType, modules: Iterable[ModuleType]) -> None:
    with serialize_by_value([module]) as value:
        assert_type(value, None)
    with serialize_by_value((module,)) as value:
        assert_type(value, None)
    with serialize_by_value(iter(modules)) as value:
        assert_type(value, None)
    with serialize_by_value(None) as value:
        assert_type(value, None)

    serialize_by_value(module)  # ty: ignore[invalid-argument-type]
    serialize_by_value(["module_name"])  # ty: ignore[invalid-argument-type]
    serialize_by_value([object()])  # ty: ignore[invalid-argument-type]


def check_settings_save_contracts(settings: Settings, module: ModuleType) -> None:
    settings.save("settings.pkl", modules_to_serialize=[module])
    settings.save("settings.pkl", modules_to_serialize=None)
    settings.save("settings.pkl", modules_to_serialize=["module_name"])  # ty: ignore[invalid-argument-type]
