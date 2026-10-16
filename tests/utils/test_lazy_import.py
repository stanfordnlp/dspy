import importlib
import sys
import threading
import time
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


def test_require_initializes_package_before_importing_submodule(tmp_path, monkeypatch):
    package_name = "dspy_lazy_submodule_package"
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text("initialized = True\nfrom .models import Model\n")
    (package / "models.py").write_text("from .constants import VALUE\nclass Model:\n    value = VALUE\n")
    (package / "constants.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(tmp_path)

    try:
        proxy = require(package_name)
        assert "initialized" not in vars(proxy)
        assert f"{package_name}.models" not in sys.modules

        # Binding constants to the parent must not initialize it while models is still importing.
        models = importlib.import_module(f"{package_name}.models")

        assert proxy.initialized is True
        assert proxy.Model is models.Model
        assert models.Model.value == 42
    finally:
        for name in (package_name, f"{package_name}.models", f"{package_name}.constants"):
            sys.modules.pop(name, None)


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


def test_require_does_not_expose_module_accessed_during_its_own_initialization(tmp_path, monkeypatch):
    module_name = "dspy_lazy_reentrant_module"
    hooks = types.SimpleNamespace(started=threading.Event(), release=threading.Event())
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "dspy_lazy_test_hooks", hooks)
    (tmp_path / f"{module_name}.py").write_text(
        "import dspy_lazy_test_hooks as hooks\n"
        "early = 1\n"
        "assert hooks.proxy.early == 1\n"
        "hooks.started.set()\n"
        "assert hooks.release.wait(5)\n"
        "value = 42\n"
    )
    hooks.proxy = require(module_name)

    class ContentionLock:
        """RLock that records when a thread has to wait for another thread's hold."""

        def __init__(self):
            self._lock = threading.RLock()
            self.contended = threading.Event()

        def __enter__(self):
            if not self._lock.acquire(blocking=False):
                self.contended.set()
                self._lock.acquire()

        def __exit__(self, *exc_info):
            self._lock.release()

    lock = ContentionLock()
    hooks.proxy._dspy_lazy_lock = lock

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lambda: hooks.proxy.value)
        assert hooks.started.wait(5)
        # The module is still initializing, so another thread must wait instead of seeing the partial module.
        second = executor.submit(lambda: hooks.proxy.value)
        # Release only after the second read has returned or is blocked on the proxy lock.
        deadline = time.monotonic() + 5
        while not (second.done() or lock.contended.is_set()) and time.monotonic() < deadline:
            time.sleep(0.001)
        hooks.release.set()

        assert first.result() == 42
        assert second.result() == 42


def test_require_resolves_module_that_replaces_itself_during_initialization(tmp_path, monkeypatch):
    module_name = "dspy_lazy_replacing_module"
    hooks = types.SimpleNamespace()
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "dspy_lazy_test_hooks", hooks)
    (tmp_path / f"{module_name}.py").write_text(
        "import sys\n"
        "import types\n"
        "import dspy_lazy_test_hooks as hooks\n"
        "early = 1\n"
        "assert hooks.proxy.early == 1\n"
        "sys.modules[__name__] = types.SimpleNamespace(value='replacement')\n"
    )
    hooks.proxy = require(module_name)

    # A plain import, not the proxy, starts initialization, so the proxy only sees the partial module mid-import.
    importlib.import_module(module_name)

    assert hooks.proxy.value == "replacement"


def test_require_reuses_loaded_module_until_sys_modules_changes(tmp_path, monkeypatch):
    module_name = "dspy_lazy_cached_module"
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text("value = 1\n")
    mod = require(module_name)
    assert mod.value == 1

    imports = []
    import_module = importlib.import_module

    def counting_import_module(name, *args, **kwargs):
        imports.append(name)
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", counting_import_module)

    assert [mod.value for _ in range(3)] == [1, 1, 1]
    assert imports == []

    monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(value=2))
    assert mod.value == 2

    del sys.modules[module_name]
    assert mod.value == 1
    assert imports == [module_name, module_name]


def test_require_retries_after_failed_import(tmp_path, monkeypatch):
    module_name = "dspy_lazy_flaky_module"
    flag = tmp_path / "fail"
    flag.touch()
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text(
        f"import pathlib\nif pathlib.Path({str(flag)!r}).exists():\n    raise RuntimeError('boom')\nvalue = 42\n"
    )

    mod = require(module_name)
    with pytest.raises(RuntimeError, match="boom"):
        _ = mod.value

    flag.unlink()
    assert mod.value == 42


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
