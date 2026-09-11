"""These checks also run against the installed wheel outside the source tree."""

import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


def test_public_exports_are_the_vendored_objects():
    import dspy
    from dspy import lm15
    from dspy._vendor import lm15 as vendored
    from dspy.lm15 import Request

    assert importlib.import_module("dspy.lm15") is dspy.lm15
    assert Request is vendored.Request
    assert lm15.__all__ == vendored.__all__
    for name in lm15.__all__:
        assert getattr(lm15, name) is getattr(vendored, name)
    request = Request(model="example", messages=(lm15.Message.user("Hello"),))
    restored = pickle.loads(pickle.dumps(request))
    assert type(restored) is Request
    assert restored == request


def test_openai_converters_are_available_without_provider_calls():
    from dspy.lm15 import Request, Response, request_from_openai_chat, response_from_openai_chat

    request = request_from_openai_chat({"model": "example", "messages": [{"role": "user", "content": "Hello"}]})
    response = response_from_openai_chat({
        "id": "example-response",
        "model": "example",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    assert isinstance(request, Request)
    assert isinstance(response, Response)
    assert response.message.text == "Hi"


def test_vendored_lm15_carries_provenance_and_license():
    from dspy._vendor import lm15

    pkg = Path(lm15.__file__).parent
    assert (pkg / "py.typed").is_file()
    marker = dict(line.split("=", 1) for line in (pkg.parent / "lm15-provenance.txt").read_text().splitlines())
    assert marker["source"] == "https://github.com/cmpnd-ai/lm15-python.git"
    assert marker["version"] == lm15.__version__ != "0.0.0"
    for key in ("commit", "contract", "split"):
        assert len(marker[key]) == 40
        assert all(char in "0123456789abcdef" for char in marker[key])
    assert "MIT" in (pkg.parent / "lm15-LICENSE").read_text()


def test_engine_error_boundaries_and_stream_guard_ship_in_wheel():
    import dspy
    from dspy.clients.engines.stream_guard import checked_stream
    from dspy.lm15 import StreamAssemblyError, StreamDeltaEvent, StreamStartEvent, TextDelta

    class BrokenEngine:
        calls = 0

        def complete(self, request):
            self.calls += 1
            raise RuntimeError("network invariant failed")

    engine = BrokenEngine()
    lm = dspy.LM("custom", engine=engine, cache=False, num_retries=3)
    with pytest.raises(dspy.LMUnexpectedError) as caught:
        lm("hello")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert engine.calls == 1
    assert issubclass(dspy.LMLockTimeoutError, dspy.LMError)
    assert issubclass(dspy.LMStreamAssemblyError, dspy.LMUnexpectedError)
    with pytest.raises(StreamAssemblyError, match="without a completion event"):
        list(checked_stream(iter([StreamStartEvent(), StreamDeltaEvent(TextDelta("partial"))])))


def test_version_ignores_an_unrelated_installed_distribution(tmp_path):
    metadata = tmp_path / "lm15-7.8.9.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: lm15\nVersion: 7.8.9\n")
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from importlib.metadata import version
assert version("lm15") == "7.8.9"
import dspy.lm15
from dspy._vendor import lm15
from dspy._vendor.lm15 import vet
from pathlib import Path
record = Path(lm15.__file__).parent.parent / "lm15-provenance.txt"
expected = dict(line.split("=", 1) for line in record.read_text().splitlines())["version"]
assert dspy.lm15.__version__ == lm15.__version__ == vet.IMPL_VERSION == expected
assert expected != "7.8.9"
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
