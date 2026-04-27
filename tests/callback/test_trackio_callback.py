from unittest.mock import MagicMock, patch

import pytest

from dspy.utils.trackio import TrackioCallback


@pytest.fixture
def fake_trackio():
    """Patch ``trackio`` and return a mock module with ``init``, ``log``, ``Trace``."""
    fake = MagicMock()
    fake.Trace = MagicMock(side_effect=lambda messages, metadata: {"_messages": messages, "_metadata": metadata})
    with patch.dict("sys.modules", {"trackio": fake}):
        yield fake


def test_logs_trace_for_messages_input(fake_trackio):
    cb = TrackioCallback(project="test")
    cb.on_lm_start(
        call_id="c1",
        instance=None,
        inputs={"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"},
    )
    cb.on_lm_end(call_id="c1", outputs=["hello there"], exception=None)

    fake_trackio.init.assert_called_once()
    fake_trackio.log.assert_called_once()
    logged = fake_trackio.log.call_args[0][0]
    assert "lm_call" in logged
    trace = logged["lm_call"]
    assert trace["_messages"][0] == {"role": "user", "content": "hi"}
    assert trace["_messages"][-1] == {"role": "assistant", "content": "hello there"}
    assert trace["_metadata"]["model"] == "gpt-4o"


def test_logs_trace_for_prompt_input(fake_trackio):
    cb = TrackioCallback(project="test")
    cb.on_lm_start(call_id="c1", instance=None, inputs={"prompt": "what is 2+2?"})
    cb.on_lm_end(call_id="c1", outputs=["4"], exception=None)

    trace = fake_trackio.log.call_args[0][0]["lm_call"]
    assert trace["_messages"][0] == {"role": "user", "content": "what is 2+2?"}
    assert trace["_messages"][-1] == {"role": "assistant", "content": "4"}


def test_skips_logging_on_exception(fake_trackio):
    cb = TrackioCallback(project="test")
    cb.on_lm_start(call_id="c1", instance=None, inputs={"prompt": "x"})
    cb.on_lm_end(call_id="c1", outputs=None, exception=RuntimeError("boom"))
    fake_trackio.log.assert_not_called()


def test_step_increments(fake_trackio):
    cb = TrackioCallback(project="test")
    for i in range(3):
        cb.on_lm_start(call_id=f"c{i}", instance=None, inputs={"prompt": str(i)})
        cb.on_lm_end(call_id=f"c{i}", outputs=[f"r{i}"], exception=None)
    steps = [c.kwargs.get("step") for c in fake_trackio.log.call_args_list]
    assert steps == [0, 1, 2]
