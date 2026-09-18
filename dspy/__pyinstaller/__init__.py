"""PyInstaller support: DSPy ships its own hook so a frozen application
collects the package data DSPy reads at run time (the Flex sandbox shim,
the local-interpreter worker and Deno runner, the model-metadata snapshot,
the vendored lm15 provenance). Discovered through the ``pyinstaller40``
entry point; no user flags needed."""

import os


def get_hook_dirs():
    return [os.path.dirname(__file__)]
