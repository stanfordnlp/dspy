"""A SUB_DSPY interpreter with OS-level isolation: the subprocess worker under Anthropic's sandbox-runtime.

Needs the ``srt`` CLI (``npm install -g @anthropic-ai/sandbox-runtime``) and dspy-interpreters; skipped
otherwise. Marked ``extra`` so CI runs it in the extra/deno job, which installs both.
"""

import shutil

import pytest

import dspy
from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import SUB_DSPY_FACTORY_NAME
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM

pytestmark = [
    pytest.mark.extra,
    pytest.mark.skipif(shutil.which("srt") is None, reason="srt (sandbox-runtime) is not installed"),
]
sandbox_runtime = pytest.importorskip("tests.sandbox_runtime_interpreter")

NETWORK_PROBE = """
import urllib.request
try:
    urllib.request.urlopen("https://github.com", timeout=15)
    status = "reachable"
except Exception:
    status = "blocked"
"""


class NoNetworkSandbox(sandbox_runtime.SandboxRuntimeInterpreter):
    """Zero-argument factory (so RLM/Flex read its capabilities) with all egress denied."""

    def __init__(self):
        super().__init__()


def scripted_predictor(codes: list[str]):
    class Scripted:
        def __init__(self):
            self.idx = 0

        def __call__(self, **kwargs):
            code = codes[min(self.idx, len(codes) - 1)]
            self.idx += 1
            return Prediction(reasoning="scripted", code=code)

    return Scripted()


def test_worker_runs_native_dspy_under_the_declared_policy():
    interpreter = sandbox_runtime.SandboxRuntimeInterpreter(allowed_domains=("example.com",))
    try:
        output = str(interpreter.execute(
            "import dspy, os, urllib.request\n"
            "print('dspy:', dspy.__file__)\n"
            "print('example:', urllib.request.urlopen('https://example.com', timeout=15).status)\n"
            + NETWORK_PROBE
            + "print('github:', status)\n"
            "try:\n"
            "    open(os.path.expanduser('~/srt-escape.txt'), 'w')\n"
            "    print('home write: allowed')\n"
            "except PermissionError:\n"
            "    print('home write: denied')\n"
        ))
    finally:
        interpreter.shutdown()

    assert "dspy: " in output and "dspy/__init__.py" in output  # the real package, not a facade
    assert "example: 200" in output
    assert "github: blocked" in output
    assert "home write: denied" in output


def test_nested_interpreters_inherit_the_sandbox():
    interpreter = NoNetworkSandbox()
    try:
        output = str(interpreter.execute(
            f"nested = {SUB_DSPY_FACTORY_NAME}()\n"
            "try:\n"
            f"    print(nested.execute({NETWORK_PROBE + 'print(status)'!r}))\n"
            "finally:\n"
            "    nested.shutdown()\n"
        ))
    finally:
        interpreter.shutdown()

    assert "blocked" in output


def test_rlm_runs_sub_dspy_inside_the_sandbox():
    dspy.configure(lm=dspy.LM("openai/host-default", cache=False))
    rlm = RLM("query -> answer", max_iters=2, interpreter_factory=NoNetworkSandbox)
    assert rlm._sub_dspy
    rlm.generate_action = scripted_predictor([
        "import dspy\nprint(dspy.__file__)\nprint(dspy.settings.lm.model)",
        NETWORK_PROBE + "SUBMIT(status)",
    ])

    result = rlm(query="q")

    first_output = result.trajectory[0]["output"]
    assert "dspy/__init__.py" in first_output  # native dspy inside the worker
    assert "openai/host-default" in first_output  # the host's LM crossed as the sub-agent LM
    assert result.answer == "blocked"


def test_flex_module_code_runs_inside_the_sandbox():
    flex = dspy.Flex("question -> answer", interpreter_factory=NoNetworkSandbox)
    flex.load_state({"module_src": (
        "class ProbeModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "\n"
        "    def forward(self, **inputs):\n"
        "        import urllib.request\n"
        "        try:\n"
        "            urllib.request.urlopen('https://github.com', timeout=15)\n"
        "            status = 'reachable'\n"
        "        except Exception:\n"
        "            status = 'blocked'\n"
        "        return dspy.Prediction(answer=status)\n"
    )})

    assert flex(question="probe").answer == "blocked"


def test_flex_bridged_predictor_from_inside_the_sandbox():
    flex = dspy.Flex("question -> answer", interpreter_factory=NoNetworkSandbox)  # baseline dspy.Predict

    with dspy.context(lm=DummyLM([{"answer": "bridged"}])):
        assert flex(question="probe").answer == "bridged"
