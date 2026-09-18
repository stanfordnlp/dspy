"""DSPy's PyInstaller hook is discoverable and names the files DSPy reads
as data at run time. The real freeze is done by hand (see the PR); this
pins what a freeze depends on."""

import importlib
from importlib.metadata import entry_points
from pathlib import Path


def test_hook_is_registered_and_names_existing_files():
    hook_dirs = importlib.import_module("dspy.__pyinstaller").get_hook_dirs()
    hook = Path(hook_dirs[0]) / "hook-dspy.py"
    assert hook.is_file()
    source = hook.read_text()
    package = Path(importlib.import_module("dspy").__file__).parent
    for name in ("predict/flex/_sandbox_shim.py", "primitives/local_interpreter_worker.py",
                 "clients/model_metadata/snapshot.json.gz", "clients/model_metadata/provenance.json",
                 "_vendor/lm15-provenance.txt"):
        assert name in source and (package / name).is_file(), name
    assert "include_py_files=True" in source  # the shim and the worker are .py files read as text
    # The entry point is what makes PyInstaller find the hook with no flags;
    # it lives in the installed distribution's metadata, so an install that
    # predates it fails here — reinstall (`uv sync` / `pip install -e .`).
    registered = [entry for entry in entry_points(group="pyinstaller40") if entry.value.startswith("dspy.")]
    assert [entry.value for entry in registered] == ["dspy.__pyinstaller:get_hook_dirs"]
    assert registered[0].load()() == hook_dirs
