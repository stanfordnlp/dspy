import pytest

from dspy.utils.lazy_import import _INSTALL_HINTS, _MissingModule, is_available, require


def test_is_available_true_for_stdlib():
    assert is_available("json") is True


def test_is_available_false_for_missing():
    assert is_available("definitely_not_a_real_module_xyz") is False


def test_is_available_does_not_import_module(monkeypatch):
    import sys

    sys.modules.pop("dspy.utils.lazy_import", None)
    is_available.cache_clear()
    before = set(sys.modules)
    assert is_available("dspy.utils.lazy_import") is True
    after = set(sys.modules)
    assert "dspy.utils.lazy_import" not in (after - before)


def test_require_returns_lazy_module_when_present():
    mod = require("json")
    assert mod.dumps({"a": 1}) == '{"a": 1}'


def test_require_returns_cached_module():
    import sys

    mod = require("json")
    assert mod is sys.modules["json"]


def test_require_returns_stub_when_missing():
    stub = require("definitely_not_a_real_module_xyz", feature="dspy.X")
    assert isinstance(stub, _MissingModule)


def test_require_stub_raises_on_access_with_install_hint():
    stub = require("nonexistent_abc", feature="dspy.Test")
    with pytest.raises(ImportError) as exc_info:
        stub.something
    msg = str(exc_info.value)
    assert "dspy[nonexistent_abc]" in msg, msg
    assert "dspy.Test" in msg


def test_require_stub_uses_explicit_extra():
    stub = require("nonexistent_xyz", extra="custom", feature="dspy.X")
    with pytest.raises(ImportError) as exc_info:
        stub.something
    assert "dspy[custom]" in str(exc_info.value)


def test_require_stub_falls_back_to_module_name():
    stub = require("nonexistent_xyz", feature="dspy.X")
    with pytest.raises(ImportError) as exc_info:
        stub.something
    assert "dspy[nonexistent_xyz]" in str(exc_info.value)


def test_install_hints_match_pyproject_extras():
    import pathlib
    import re

    pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present")
    text = pyproject.read_text()
    # Extract extra names from lines like: numpy = ["numpy>=1.26.0"]
    in_section = False
    extras: set[str] = set()
    for line in text.splitlines():
        if line.strip() == "[project.optional-dependencies]":
            in_section = True
            continue
        if in_section:
            if line.startswith("["):
                break
            m = re.match(r"^(\w[\w-]*)\s*=", line)
            if m:
                extras.add(m.group(1))
    for module, hint in _INSTALL_HINTS.items():
        assert hint in extras, (
            f"_INSTALL_HINTS[{module!r}] = {hint!r} is not a declared extra in "
            f"pyproject.toml (declared: {sorted(extras)})"
        )
