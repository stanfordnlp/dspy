"""Helpers for running Deno, shared by the interpreter and the sandbox archive builder."""

import os


def deno_subprocess_env() -> dict[str, str]:
    """Build an environment that prevents ambient package.json discovery."""
    env = os.environ.copy()
    env["DENO_NO_PACKAGE_JSON"] = "1"
    return env
