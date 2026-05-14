import subprocess
import sys
import textwrap


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_import_dspy_succeeds_without_litellm():
    run_python(
        """
        import importlib.util
        import sys

        real_find_spec = importlib.util.find_spec

        def find_spec(name, *args, **kwargs):
            if name == "litellm" or name.startswith("litellm."):
                return None
            return real_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        sys.modules.pop("litellm", None)

        import dspy

        _ = dspy.LM
        _ = dspy.Embedder
        _ = dspy.streamify
        """
    )


def test_lm_litellm_use_raises_helpful_error_without_litellm():
    run_python(
        """
        import importlib.util
        import sys

        real_find_spec = importlib.util.find_spec

        def find_spec(name, *args, **kwargs):
            if name == "litellm" or name.startswith("litellm."):
                return None
            return real_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        sys.modules.pop("litellm", None)

        import dspy

        try:
            _ = dspy.LM("openai/gpt-4o-mini").supports_function_calling
        except ImportError as e:
            msg = str(e)
            assert "[litellm]" in msg
            assert "dspy.LM" in msg
        else:
            raise AssertionError("Expected ImportError")
        """
    )


def test_embedder_litellm_use_raises_helpful_error_without_litellm():
    run_python(
        """
        import importlib.util
        import sys

        real_find_spec = importlib.util.find_spec

        def find_spec(name, *args, **kwargs):
            if name == "litellm" or name.startswith("litellm."):
                return None
            return real_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        sys.modules.pop("litellm", None)

        import dspy

        try:
            dspy.Embedder("openai/text-embedding-3-small")(["hello"])
        except ImportError as e:
            msg = str(e)
            assert "[litellm]" in msg
            assert "dspy.Embedder" in msg
        else:
            raise AssertionError("Expected ImportError")
        """
    )
