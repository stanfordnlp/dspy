import asyncio
import inspect

import pytest

import dspy
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback


@pytest.fixture(autouse=True)
def reset_settings():
    original_settings = dspy.settings.copy()
    yield
    dspy.configure(**original_settings)


class LifecycleRecorder(BaseCallback):
    def __init__(self):
        self.calls = []

    def on_optimizer_start(self, call_id, instance, inputs):
        self.calls.append(
            {
                "handler": "start",
                "call_id": call_id,
                "parent_call_id": ACTIVE_CALL_ID.get(),
                "instance": instance,
                "inputs": inputs,
            }
        )

    def on_optimizer_end(self, call_id, outputs, exception):
        self.calls.append(
            {
                "handler": "end",
                "call_id": call_id,
                "active_call_id": ACTIVE_CALL_ID.get(),
                "outputs": outputs,
                "exception": exception,
            }
        )


def test_optimizer_lifecycle_preserves_compile_contract():
    global_recorder = LifecycleRecorder()
    instance_recorder = LifecycleRecorder()

    class IdentityOptimizer(dspy.Teleprompter):
        def __init__(self):
            self.callbacks = [instance_recorder]

        def compile(self, student, *, trainset, scale=1):
            return student

    optimizer = IdentityOptimizer()
    student = dspy.Predict("question -> answer")
    dspy.configure(callbacks=[global_recorder])

    parent_token = ACTIVE_CALL_ID.set("parent")
    try:
        result = optimizer.compile(student, trainset=["example"])
        assert ACTIVE_CALL_ID.get() == "parent"
    finally:
        ACTIVE_CALL_ID.reset(parent_token)

    assert result is student
    assert str(inspect.signature(IdentityOptimizer.compile)) == "(self, student, *, trainset, scale=1)"

    for recorder in (global_recorder, instance_recorder):
        start, end = recorder.calls
        assert start == {
            "handler": "start",
            "call_id": start["call_id"],
            "parent_call_id": "parent",
            "instance": optimizer,
            "inputs": {"student": student, "trainset": ["example"], "scale": 1},
        }
        assert end == {
            "handler": "end",
            "call_id": start["call_id"],
            "active_call_id": "parent",
            "outputs": student,
            "exception": None,
        }

    assert global_recorder.calls[0]["call_id"] == instance_recorder.calls[0]["call_id"]


def test_super_compile_is_a_nested_lifecycle_and_inheritance_does_not_add_a_wrapper():
    class ParentOptimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            return student

    class ChildOptimizer(ParentOptimizer):
        def compile(self, student, *, trainset):
            return super().compile(student, trainset=trainset)

    class InheritedOptimizer(ParentOptimizer):
        pass

    student = dspy.Predict("question -> answer")

    nested_recorder = LifecycleRecorder()
    with dspy.context(callbacks=[nested_recorder]):
        assert ChildOptimizer().compile(student, trainset=[]) is student

    outer_start, inner_start, inner_end, outer_end = nested_recorder.calls
    assert [call["handler"] for call in nested_recorder.calls] == ["start", "start", "end", "end"]
    assert outer_start["parent_call_id"] is None
    assert inner_start["call_id"] != outer_start["call_id"]
    assert inner_start["parent_call_id"] == outer_start["call_id"]
    assert inner_end["call_id"] == inner_start["call_id"]
    assert inner_end["active_call_id"] == outer_start["call_id"]
    assert outer_end["call_id"] == outer_start["call_id"]
    assert outer_end["active_call_id"] is None

    inherited_recorder = LifecycleRecorder()
    with dspy.context(callbacks=[inherited_recorder]):
        assert InheritedOptimizer().compile(student, trainset=[]) is student

    assert [call["handler"] for call in inherited_recorder.calls] == ["start", "end"]


def test_optimizer_exception_reaches_end_callback_and_restores_parent():
    expected_exception = ValueError("compile failed")

    class FailingOptimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            raise expected_exception

    recorder = LifecycleRecorder()
    dspy.configure(callbacks=[recorder])

    parent_token = ACTIVE_CALL_ID.set("parent")
    try:
        with pytest.raises(ValueError, match="compile failed"):
            FailingOptimizer().compile(object(), trainset=[])
        assert ACTIVE_CALL_ID.get() == "parent"
    finally:
        ACTIVE_CALL_ID.reset(parent_token)

    start, end = recorder.calls
    assert end["call_id"] == start["call_id"]
    assert end["active_call_id"] == "parent"
    assert end["outputs"] is None
    assert end["exception"] is expected_exception


@pytest.mark.asyncio
async def test_async_compile_reports_cancellation_to_optimizer_end():
    class AsyncOptimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset):
            raise asyncio.CancelledError

    recorder = LifecycleRecorder()

    with dspy.context(callbacks=[recorder]):
        with pytest.raises(asyncio.CancelledError):
            await AsyncOptimizer().compile(object(), trainset=[])

    start, end = recorder.calls
    assert start["handler"] == "start"
    assert end["handler"] == "end"
    assert end["call_id"] == start["call_id"]
    assert end["outputs"] is None
    assert isinstance(end["exception"], asyncio.CancelledError)


@pytest.mark.asyncio
async def test_detached_task_does_not_reuse_completed_optimizer_call_id():
    release_detached_task = asyncio.Event()
    detached_task = None

    class Recorder(LifecycleRecorder):
        def __init__(self):
            super().__init__()
            self.module_parent_call_ids = []

        def on_module_start(self, call_id, instance, inputs):
            self.module_parent_call_ids.append(ACTIVE_CALL_ID.get())

    class ChildModule(dspy.Module):
        def forward(self):
            return "done"

    child = ChildModule()

    class AsyncOptimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset):
            nonlocal detached_task

            async def run_after_compile():
                await release_detached_task.wait()
                return child()

            detached_task = asyncio.create_task(run_after_compile())
            return student

    recorder = Recorder()

    with dspy.context(callbacks=[recorder]):
        student = object()
        assert await AsyncOptimizer().compile(student, trainset=[]) is student
        optimizer_call_id = recorder.calls[0]["call_id"]

        release_detached_task.set()
        assert detached_task is not None
        assert await detached_task == "done"

    assert [call["handler"] for call in recorder.calls] == ["start", "end"]
    assert recorder.calls[1]["call_id"] == optimizer_call_id
    assert recorder.module_parent_call_ids == [None]
