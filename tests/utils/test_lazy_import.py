import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from dspy.utils.lazy_import import _INSTALL_HINTS, _detect_dspy_dist, _MissingModule, is_available, require


def test_is_available_true_for_stdlib():
    assert is_available("json") is True


def test_is_available_false_for_missing():
    assert is_available("definitely_not_a_real_module_xyz") is False


def test_is_available_does_not_import_module(monkeypatch):
    import sys

    # Use a stdlib module that dspy never imports, so we can deterministically
    # observe whether is_available() triggers an import as a side effect.
    target = "mailbox"
    monkeypatch.delitem(sys.modules, target, raising=False)
    # is_available is @functools.cache'd; clear so we actually exercise find_spec.
    is_available.cache_clear()

    assert is_available(target) is True
    assert target not in sys.modules


def test_require_returns_lazy_module_when_present():
    mod = require("json")
    assert mod.dumps({"a": 1}) == '{"a": 1}'


def test_require_returns_cached_module():
    mod = require("json")
    assert mod is sys.modules["json"]


def test_require_is_safe_under_concurrent_first_use(tmp_path, monkeypatch):
    module_name = "dspy_lazy_threaded_module"
    counter_path = tmp_path / "imports.txt"
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text(
        "import pathlib\n"
        "import time\n"
        "time.sleep(0.1)\n"
        f"path = pathlib.Path({str(counter_path)!r})\n"
        "path.write_text(path.read_text() + '1' if path.exists() else '1')\n"
        "value = 42\n"
    )

    threads = 8
    barrier = threading.Barrier(threads)

    def read_value(_):
        barrier.wait()
        return require(module_name).value

    with ThreadPoolExecutor(max_workers=threads) as executor:
        assert list(executor.map(read_value, range(threads))) == [42] * threads

    assert counter_path.read_text() == "1"


def test_require_assignment_updates_materialized_module(tmp_path, monkeypatch):
    module_name = "dspy_lazy_assignment_module"
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text("value = 1\n")

    mod = require(module_name)
    mod.value = 2

    assert sys.modules[module_name].value == 2


def test_require_leaves_sys_modules_untouched(tmp_path, monkeypatch):
    # A proxy parked in sys.modules would shadow the real module for every other importer.
    module_name = "dspy_lazy_unshadowed_module"
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text("value = 7\n")

    mod = require(module_name)

    assert module_name not in sys.modules
    assert mod.value == 7
    assert sys.modules[module_name].value == 7


def test_require_retries_after_a_failed_import(tmp_path, monkeypatch):
    module_name = "dspy_lazy_failing_module"
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text("raise RuntimeError('boom')\n")

    mod = require(module_name)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="boom"):
            _ = mod.value
    assert module_name not in sys.modules


def test_require_does_not_break_submodule_imports(tmp_path, monkeypatch):
    # A shadowed package is skipped when one of its submodules is imported, so the package ends up
    # executing from an attribute access made while that submodule is still initializing.
    package_name = "dspy_lazy_submodule_package"
    monkeypatch.syspath_prepend(tmp_path)
    for name in (package_name, f"{package_name}.sub", f"{package_name}.version"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text(f"from {package_name}.sub import LATE\n\nvalue = 42\n")
    (package / "sub.py").write_text(f"from {package_name}.version import VERSION\n\nLATE = VERSION\n")
    (package / "version.py").write_text("VERSION = '1.0'\n")

    require(package_name)
    submodule = importlib.import_module(f"{package_name}.sub")

    assert submodule.LATE == "1.0"
    assert sys.modules[package_name].sub is submodule
    assert require(package_name).value == 42


def test_require_returns_stub_when_missing():
    stub = require("definitely_not_a_real_module_xyz", feature="dspy.X")
    assert isinstance(stub, _MissingModule)


def test_require_stub_raises_on_access_with_install_hint():
    dist = _detect_dspy_dist()
    stub = require("nonexistent_abc", feature="dspy.Test")
    with pytest.raises(ImportError) as exc_info:
        _ = stub.something
    msg = str(exc_info.value)
    assert f"{dist}[nonexistent_abc]" in msg, msg
    assert "dspy.Test" in msg


def test_require_stub_uses_install_hint_for_litellm(monkeypatch):
    import importlib.util
    import sys

    dist = _detect_dspy_dist()
    find_spec = importlib.util.find_spec
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda module: None if module == "litellm" else find_spec(module))

    stub = require("litellm", feature="dspy.LM")
    with pytest.raises(ImportError) as exc_info:
        _ = stub.something
    assert f"{dist}[litellm]" in str(exc_info.value)


def test_require_stub_uses_explicit_extra():
    dist = _detect_dspy_dist()
    stub = require("nonexistent_xyz", extra="custom", feature="dspy.X")
    with pytest.raises(ImportError) as exc_info:
        _ = stub.something
    assert f"{dist}[custom]" in str(exc_info.value)


def test_require_stub_falls_back_to_module_name():
    dist = _detect_dspy_dist()
    stub = require("nonexistent_xyz", feature="dspy.X")
    with pytest.raises(ImportError) as exc_info:
        _ = stub.something
    assert f"{dist}[nonexistent_xyz]" in str(exc_info.value)


def test_install_hints_match_pyproject_extras(pytestconfig):
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    pyproject = pytestconfig.rootpath / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    extras = set(data["project"]["optional-dependencies"])

    for module, hint in _INSTALL_HINTS.items():
        assert hint in extras, (
            f"_INSTALL_HINTS[{module!r}] = {hint!r} is not a declared extra in "
            f"pyproject.toml (declared: {sorted(extras)})"
        )
