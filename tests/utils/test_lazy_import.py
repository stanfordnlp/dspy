import sys
import importlib
import types
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

def _write_lazy_package(tmp_path, monkeypatch, name: str) -> None:
    """Create an importable package whose child, mid-exec, imports a sibling.

    Mirrors the numpy shape that broke with onnxruntime: the package
    ``__init__`` imports the child, and the child's first act is importing
    a sibling submodule -- which the import machinery binds onto the parent
    module found in ``sys.modules``.
    """
    (tmp_path / name).mkdir()
    (tmp_path / f"{name}/__init__.py").write_text(
        "INIT_RAN = True\n"
        f"from {name}.child import CHILD_VALUE\n"
    )
    (tmp_path / f"{name}/child.py").write_text(
        "import sys\n"
        f"import {name}.version\n"
        "PARENT_INIT_RAN_AT_CHILD_IMPORT = getattr(sys.modules[__package__], 'INIT_RAN', False)\n"
        "CHILD_VALUE = 42\n"
    )
    (tmp_path / f"{name}/version.py").write_text("VERSION = '1.0'\n")
    monkeypatch.syspath_prepend(tmp_path)
    for suffix in ("", ".child", ".version"):
        monkeypatch.delitem(sys.modules, f"{name}{suffix}", raising=False)


def test_direct_submodule_import_runs_package_init_first(tmp_path, monkeypatch):
    """A submodule imported directly still gets the package's __init__ side effects.

    The lazy proxy sits in ``sys.modules`` as the not-yet-loaded parent, and
    a C extension (onnxruntime after ``import dspy``) imports
    ``numpy._core`` directly. The machinery reads the parent's ``__path__``
    to locate the child; answering from the proxy let the child execute
    while the real ``__init__`` -- which registers numpy's Windows DLL
    directory before importing ``._core`` -- never ran.
    """
    name = "dspy_lazy_parent_first_pkg"
    _write_lazy_package(tmp_path, monkeypatch, name)

    require(name)
    assert sys.modules[name] is not None

    child = importlib.import_module(f"{name}.child")

    assert child.CHILD_VALUE == 42
    assert child.PARENT_INIT_RAN_AT_CHILD_IMPORT is True, (
        "the package __init__ must run before any of its submodules"
    )
    assert sys.modules[f"{name}.version"].VERSION == "1.0"


def test_lazy_module_path_access_materializes_the_package(tmp_path, monkeypatch):
    """Reading a lazy package's ``__path__`` materializes it.

    ``__path__`` is what the import machinery consults to import or find
    submodules, so it doubles as the materialization trigger for the
    direct-submodule-import case; any other reader observes the real
    package's path.
    """
    name = "dspy_lazy_path_materialize_pkg"
    _write_lazy_package(tmp_path, monkeypatch, name)

    mod = require(name)
    assert sys.modules[name] is mod

    paths = list(mod.__path__)

    assert paths and str(tmp_path) in str(paths[0])
    assert sys.modules[name] is not mod, "__path__ access must materialize the real package"
    assert sys.modules[name].INIT_RAN is True


def test_submodule_binding_records_without_materializing(tmp_path, monkeypatch):
    """The machinery binding a submodule onto the proxy must not nest a full load.

    Executing the real package from inside ``__setattr__`` runs the
    ``__init__`` while that submodule's import is still in flight; the
    ``__init__`` then re-imports the half-initialized submodule from
    ``sys.modules`` and the double execution corrupts package state (the
    onnxruntime failure died inside numpy with "data type 'bool' not
    understood"). The binding is recorded on the proxy instead.
    """
    name = "dspy_lazy_submodule_binding_pkg"
    (tmp_path / name).mkdir()
    (tmp_path / f"{name}/__init__.py").write_text("INIT_RAN = True\n")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, name, raising=False)

    mod = require(name)
    sibling = types.ModuleType(f"{name}.extra")
    sys.modules[f"{name}.extra"] = sibling

    setattr(mod, "extra", sibling)

    assert sys.modules[name] is mod, "a submodule binding must not materialize the package"
    assert mod.extra is sibling

    # A plain value assignment still materializes and applies to the real module.
    mod.setting = "value"
    assert sys.modules[name] is not mod
    assert sys.modules[name].setting == "value"


def test_direct_submodule_import_executes_the_child_exactly_once(tmp_path, monkeypatch):
    """A direct child import runs the child one time, not twice.

    Materializing the parent from ``__path__`` happens after the outer
    child import has committed to a spec, and the package ``__init__``
    imports the same child -- so without reuse the pending import
    re-executes it and non-idempotent children register state twice.
    """
    name = "dspy_lazy_exact_once_pkg"
    events_path = tmp_path / "events.txt"
    (tmp_path / name).mkdir()
    record = (
        "import pathlib\n"
        f"path = pathlib.Path({str(events_path)!r})\n"
        "def log(event):\n"
        "    with path.open('a') as fh:\n"
        "        fh.write(event + chr(10))\n"
    )
    (tmp_path / f"{name}/__init__.py").write_text(
        record
        + "log('package-init')\n"
        + f"from {name}.child import CHILD_VALUE\n"
    )
    (tmp_path / f"{name}/child.py").write_text(
        record
        + "log('child-exec')\n"
        + "import sys\n"
        + f"import {name}.version\n"
        + "CHILD_VALUE = 42\n"
    )
    (tmp_path / f"{name}/version.py").write_text(record + "log('version-exec')\n")

    monkeypatch.syspath_prepend(tmp_path)
    for suffix in ("", ".child", ".version"):
        monkeypatch.delitem(sys.modules, f"{name}{suffix}", raising=False)

    require(name)
    child = importlib.import_module(f"{name}.child")

    events = events_path.read_text().splitlines()
    assert child.CHILD_VALUE == 42
    assert events.count("package-init") == 1
    assert events.count("child-exec") == 1, f"child must execute exactly once, saw: {events}"
    assert events.count("version-exec") == 1
    assert events.index("package-init") < events.index("child-exec"), (
        "the package initializer runs before the child"
    )
    # The reused module keeps a real spec/loader, not the reuse shims,
    # and the reuse marker is consumed by the one pending import it served.
    assert child.__spec__ is not None and child.__spec__.loader is not None
    assert getattr(child, "_dspy_lazy_reuse", None) is None


def test_reload_reexecutes_a_reused_child(tmp_path, monkeypatch):
    """importlib.reload re-executes a child that was served by the reuse path.

    The reuse marker is consumed by the one pending import it serves, so a
    later reload must fall through to the normal finders and run the
    module body again instead of silently returning the stale module.
    """
    name = "dspy_lazy_reload_pkg"
    events_path = tmp_path / "events.txt"
    (tmp_path / name).mkdir()
    record = (
        "import pathlib\n"
        f"path = pathlib.Path({str(events_path)!r})\n"
        "def log(event):\n"
        "    with path.open('a') as fh:\n"
        "        fh.write(event + chr(10))\n"
    )
    (tmp_path / f"{name}/__init__.py").write_text(
        record
        + "log('package-init')\n"
        + f"from {name}.child import CHILD_VALUE\n"
    )
    (tmp_path / f"{name}/child.py").write_text(
        record + "log('child-exec')\n" + "CHILD_VALUE = 42\n"
    )

    monkeypatch.syspath_prepend(tmp_path)
    for suffix in ("", ".child"):
        monkeypatch.delitem(sys.modules, f"{name}{suffix}", raising=False)

    require(name)
    child = importlib.import_module(f"{name}.child")
    assert events_path.read_text().splitlines().count("child-exec") == 1

    importlib.reload(child)

    assert events_path.read_text().splitlines().count("child-exec") == 2, (
        "reload must re-execute the module body"
    )
    assert child.CHILD_VALUE == 42
