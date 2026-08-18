import sys
import threading
import importlib
import types
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


def test_require_module_valued_assignment_does_not_materialize(tmp_path, monkeypatch):
    module_name = "dspy_lazy_module_assignment_module"
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text("value = 1\n")

    mod = require(module_name)
    other = types.ModuleType("an_external_module")
    mod.plugged_in = other

    # Module-valued assignment is treated as the import system binding a submodule
    # onto its parent package: it must not trigger materialization.
    assert sys.modules[module_name] is mod
    assert mod.plugged_in is other

    # Materializing later replays the binding onto the real module.
    assert mod.value == 1
    assert sys.modules[module_name].plugged_in is other


def test_require_submodule_import_does_not_reenter_parent_init(tmp_path, monkeypatch):
    """Importing a submodule while the parent slot still holds the lazy proxy must not
    nest a full exec of the parent inside the submodule's initialization.

    The import system binds submodules onto their parent with ``setattr``; the proxy
    used to treat that as an attribute assignment, materialized the parent from inside
    the import machinery, and the parent's own imports then found the submodule
    mid-initialization. With numpy this surfaces as ``TypeError: data type 'bool' not
    understood`` when ``import numpy._core`` races the lazy parent (e.g. a C extension
    importing ``numpy._core`` directly after ``import dspy``).
    """
    pkg = "dspy_lazy_reentrant_pkg"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        "from .sub import marker\n"
        "parent_value = 1\n"
    )
    (pkg_dir / "sub.py").write_text(
        f"from {pkg}.version import version as __version__\n"
        "marker = 'ok'\n"
    )
    (pkg_dir / "version.py").write_text("version = '1.0'\n")
    monkeypatch.syspath_prepend(tmp_path)
    for suffix in ("", ".sub", ".version"):
        monkeypatch.delitem(sys.modules, pkg + suffix, raising=False)

    proxy = require(pkg)
    assert sys.modules[pkg] is proxy

    # Used to raise ImportError: cannot import name 'marker' from partially
    # initialized module (nested parent exec inside the submodule's init).
    sub = importlib.import_module(f"{pkg}.sub")
    assert sub.marker == "ok"

    # The import system's submodule bindings are visible through the proxy
    # without materializing the parent.
    assert proxy.version.version == "1.0"
    assert sys.modules[pkg] is proxy

    # Materializing the parent afterwards keeps every binding consistent.
    assert proxy.parent_value == 1
    real = sys.modules[pkg]
    assert real is not proxy
    assert real.sub.marker == "ok"
    assert real.version.version == "1.0"


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
