"""Test that litellm is not required at import time.

DSPy should be importable and usable for basic operations (e.g., BaseLM
subclasses, signatures, modules) without litellm being installed. LiteLLM
should only be needed when dspy.LM, dspy.Embedder, or streaming features
are actually used.
"""

import subprocess
import sys
import textwrap


def _run_snippet(snippet: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a Python snippet in a subprocess with litellm poisoned."""
    # The snippet is prefixed with code that makes `import litellm` raise ImportError
    # before dspy is imported, simulating litellm not being installed.
    wrapper = textwrap.dedent("""\
        import importlib.abc
        import importlib.machinery
        import sys

        # Poison litellm: any import of litellm or its submodules will raise ImportError
        class _LitellmBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path, target=None):
                if fullname == "litellm" or fullname.startswith("litellm."):
                    return importlib.machinery.ModuleSpec(fullname, self)
                return None
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise ImportError(f"Poisoned: {module.__name__} is not available")

        sys.meta_path.insert(0, _LitellmBlocker())

        # Remove litellm if it was already imported
        for key in list(sys.modules):
            if key == "litellm" or key.startswith("litellm."):
                del sys.modules[key]

    """)
    full_code = wrapper + textwrap.dedent(snippet)
    return subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_import_dspy_without_litellm():
    """Importing dspy should succeed without litellm."""
    result = _run_snippet("import dspy; print('ok')")
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "ok" in result.stdout


def test_baselm_subclass_without_litellm():
    """A custom BaseLM subclass should work without litellm."""
    result = _run_snippet("""\
        import dspy

        class MyLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                class R:
                    choices = [type('C', (), {'message': type('M', (), {'content': 'hello', 'tool_calls': None, 'reasoning_content': None})()})]
                    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    model = 'test'
                return R()

        lm = MyLM(model="test")
        result = lm("hi")
        print(result)
        print("ok")
    """)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "ok" in result.stdout


def test_signatures_without_litellm():
    """Signature creation should work without litellm."""
    result = _run_snippet("""\
        import dspy

        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        print(QA)
        print("ok")
    """)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "ok" in result.stdout


def test_predict_module_creation_without_litellm():
    """Creating a Predict module should work without litellm."""
    result = _run_snippet("""\
        import dspy

        predict = dspy.Predict("question -> answer")
        print(predict)
        print("ok")
    """)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "ok" in result.stdout


def test_adapter_call_with_custom_baselm_without_litellm():
    """A full adapter → custom BaseLM round-trip should work without litellm."""
    result = _run_snippet("""\
        import dspy

        class MyLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                class R:
                    choices = [type('C', (), {
                        'message': type('M', (), {
                            'content': '[[ ## answer ## ]]\\nParis\\n\\n[[ ## completed ## ]]',
                            'tool_calls': None,
                            'reasoning_content': None,
                        })(),
                        'finish_reason': 'stop',
                    })]
                    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    model = 'test'
                return R()

        lm = MyLM(model="test")
        dspy.configure(lm=lm)
        result = dspy.Predict("question -> answer")(question="What is the capital of France?")
        assert result.answer == "Paris", f"got: {result.answer}"
        print("ok")
    """)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "ok" in result.stdout


def test_dspy_lm_fails_gracefully_without_litellm():
    """dspy.LM() should raise a clear error when litellm is not available."""
    result = _run_snippet("""\
        import dspy

        try:
            lm = dspy.LM("openai/gpt-4o")
            print("should have failed")
        except ImportError as e:
            print(f"ImportError: {e}")
            print("ok")
    """)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "ok" in result.stdout
