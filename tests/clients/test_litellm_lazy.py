import subprocess
import sys


def test_import_dspy_does_not_load_litellm():
    """Ensure `import dspy` does not eagerly import litellm.

    litellm adds ~400-550ms to import time. It should only be loaded when
    actually needed (e.g. on the first LM() call), not at `import dspy` time.
    If this test fails, someone likely added a module-level `import litellm`
    in a module that is transitively imported by dspy/__init__.py.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import dspy, sys; print('litellm' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "False", (
        "litellm was imported during `import dspy`. "
        "Use `from dspy.clients._litellm import get_litellm` and call get_litellm() "
        "inside functions instead of `import litellm` at module level."
    )
