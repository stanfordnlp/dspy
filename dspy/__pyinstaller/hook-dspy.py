"""PyInstaller hook for DSPy: collect the files DSPy reads as package data.

Modules are frozen as bytecode; these files are read as data at run time.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files(
    "dspy",
    include_py_files=True,  # two of these are .py files read as text, not imported
    includes=[
        "predict/flex/_sandbox_shim.py",
        "primitives/local_interpreter_worker.py",
        "primitives/*.js",
        "clients/model_metadata/snapshot.json.gz",
        "clients/model_metadata/provenance.json",
        "clients/model_metadata/LICENSE",
        "_vendor/lm15-LICENSE",
        "_vendor/lm15-provenance.txt",
        "_vendor/UPDATING.md",
    ],
)
