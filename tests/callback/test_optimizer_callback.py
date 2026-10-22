import asyncio
from unittest.mock import Mock, patch

import cloudpickle
import pytest

import dspy
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.vanilla import LabeledFewShot
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from dspy.utils.dummies import DummyLM


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.events = []

    def on_compile_start(self, call_id, instance, inputs):
        self.events.append({
            "handler": "compile_start",
            "call_id": call_id,
            "parent_call_id": ACTIVE_CALL_ID.get(),
            "instance": instance,
            "inputs": inputs,
        })

    def on_compile_end(self, call_id, outputs, exception):
        self.events.append({
            "handler": "compile_end",
            "call_id": call_id,
            "parent_call_id": ACTIVE_CALL_ID.get(),
            "outputs": outputs,
            "exception": exception,
        })

    def on_module_start(self, call_id, instance, inputs):
        self.events.append({
            "handler": "module_start",
            "call_id": call_id,
            "parent_call_id": ACTIVE_CALL_ID.get(),
            "instance": instance,
            "inputs": inputs,
        })

    def on_module_end(self, call_id, outputs, exception):
        self.events.append({
            "handler": "module_end",
            "call_id": call_id,
            "parent_call_id": ACTIVE_CALL_ID.get(),
            "outputs": outputs,
            "exception": exception,
        })


def events_for(callback, handler):
    return [event for event in callback.events if event["handler"] == handler]


def test_compile_callbacks_report_bound_inputs_output_and_call_id():
    class Optimizer(Teleprompter):
        def compile(self, student, *, trainset, teacher=None):
            return student

    callback = RecordingCallback()
    optimizer = Optimizer()
    student = dspy.Module()
    trainset = [dspy.Example(question="test").with_inputs("question")]

    with dspy.context(callbacks=[callback]):
        result = optimizer.compile(student, trainset=trainset)

    start, end = callback.events
    assert start["handler"] == "compile_start"
    assert start["instance"] is optimizer
    assert start["inputs"] == {"student": student, "trainset": trainset, "teacher": None}
    assert start["parent_call_id"] is None
    assert end["handler"] == "compile_end"
    assert end["call_id"] == start["call_id"]
    assert end["outputs"] is result is student
    assert end["exception"] is None


def test_compile_callback_reports_and_propagates_same_exception():
    expected_error = ValueError("compile failed")

    class FailingOptimizer(Teleprompter):
        def __init__(self):
            self.should_fail = True

        def compile(self, student, *, trainset):
            if self.should_fail:
                self.should_fail = False
                raise expected_error
            return student

    callback = RecordingCallback()
    optimizer = FailingOptimizer()
    student = dspy.Module()

    with dspy.context(callbacks=[callback]):
        with pytest.raises(ValueError) as exc_info:
            optimizer.compile(student, trainset=[])
        assert optimizer.compile(student, trainset=[]) is student

    first_start, first_end, second_start, second_end = callback.events
    assert first_end["call_id"] == first_start["call_id"]
    assert first_end["outputs"] is None
    assert first_end["exception"] is expected_error
    assert exc_info.value is expected_error
    assert second_end["call_id"] == second_start["call_id"]
    assert second_end["outputs"] is student
    assert second_end["exception"] is None


def test_nested_optimizer_and_module_callbacks_preserve_hierarchy():
    class Student(dspy.Module):
        def forward(self, value):
            return dspy.Prediction(value=value)

    class InnerOptimizer(Teleprompter):
        def compile(self, student, *, trainset):
            student(value=trainset[0])
            return student

    class OuterOptimizer(Teleprompter):
        def compile(self, student, *, trainset, optimizer):
            return optimizer.compile(student, trainset=trainset)

    callback = RecordingCallback()
    outer = OuterOptimizer()
    inner = InnerOptimizer()
    student = Student()

    with dspy.context(callbacks=[callback]):
        assert outer.compile(student, trainset=["value"], optimizer=inner) is student

    assert [event["handler"] for event in callback.events] == [
        "compile_start",
        "compile_start",
        "module_start",
        "module_end",
        "compile_end",
        "compile_end",
    ]
    outer_start, inner_start, module_start, module_end, inner_end, outer_end = callback.events
    assert inner_start["parent_call_id"] == outer_start["call_id"]
    assert module_start["parent_call_id"] == inner_start["call_id"]
    assert module_end["parent_call_id"] == inner_start["call_id"]
    assert inner_end["parent_call_id"] == outer_start["call_id"]
    assert outer_end["parent_call_id"] is None
    assert len({outer_start["call_id"], inner_start["call_id"], module_start["call_id"]}) == 3


def test_inherited_compile_is_not_wrapped_twice_and_override_is_wrapped_once():
    class BaseOptimizer(Teleprompter):
        def compile(self, student, *, trainset):
            return student

    class InheritedOptimizer(BaseOptimizer):
        pass

    class OverridingOptimizer(BaseOptimizer):
        def compile(self, student, *, trainset):
            return super().compile(student, trainset=trainset)

    callback = RecordingCallback()
    student = dspy.Module()

    with dspy.context(callbacks=[callback]):
        InheritedOptimizer().compile(student, trainset=[])
        OverridingOptimizer().compile(student, trainset=[])

    starts = events_for(callback, "compile_start")
    ends = events_for(callback, "compile_end")
    assert len(starts) == len(ends) == 2
    assert isinstance(starts[0]["instance"], InheritedOptimizer)
    assert isinstance(starts[1]["instance"], OverridingOptimizer)
    assert starts[1]["parent_call_id"] is None


@pytest.mark.asyncio
async def test_async_override_calling_super_emits_one_compile_callback_pair():
    class BaseOptimizer(Teleprompter):
        async def compile(self, student, *, trainset):
            return student

    class OverridingOptimizer(BaseOptimizer):
        async def compile(self, student, *, trainset):
            return await super().compile(student, trainset=trainset)

    callback = RecordingCallback()
    optimizer = OverridingOptimizer()
    student = dspy.Module()

    with dspy.context(callbacks=[callback]):
        assert await optimizer.compile(student, trainset=[]) is student

    start, end = callback.events
    assert start["instance"] is optimizer
    assert start["parent_call_id"] is None
    assert end["call_id"] == start["call_id"]


@pytest.mark.asyncio
async def test_same_instance_compile_in_child_task_emits_nested_callback_pair():
    class Optimizer(Teleprompter):
        async def compile(self, student, *, trainset, child=False):
            if not child:
                return await asyncio.create_task(self.compile(student, trainset=trainset, child=True))
            return student

    callback = RecordingCallback()
    optimizer = Optimizer()
    student = dspy.Module()

    with dspy.context(callbacks=[callback]):
        assert await optimizer.compile(student, trainset=[]) is student

    outer_start, child_start, child_end, outer_end = callback.events
    assert [event["handler"] for event in callback.events] == [
        "compile_start",
        "compile_start",
        "compile_end",
        "compile_end",
    ]
    assert outer_start["instance"] is child_start["instance"] is optimizer
    assert outer_start["parent_call_id"] is None
    assert child_start["parent_call_id"] == outer_start["call_id"]
    assert child_end["call_id"] == child_start["call_id"]
    assert child_end["parent_call_id"] == outer_start["call_id"]
    assert outer_end["call_id"] == outer_start["call_id"]
    assert outer_end["parent_call_id"] is None


def test_compile_callback_wrapper_round_trips_through_cloudpickle():
    class Optimizer(Teleprompter):
        def compile(self, student, *, trainset):
            return student

    optimizer = cloudpickle.loads(cloudpickle.dumps(Optimizer))()
    callback = RecordingCallback()
    student = dspy.Module()

    with dspy.context(callbacks=[callback]):
        assert optimizer.compile(student, trainset=[]) is student

    start, end = callback.events
    assert start["instance"] is optimizer
    assert end["call_id"] == start["call_id"]


def test_existing_optimizer_compile_is_instrumented():
    callback = RecordingCallback()
    optimizer = LabeledFewShot(k=0)
    student = dspy.Module()

    with dspy.context(callbacks=[callback]):
        compiled = optimizer.compile(student, trainset=[])

    start, end = callback.events
    assert start["instance"] is optimizer
    assert start["inputs"] == {"student": student, "trainset": [], "sample": True}
    assert end["outputs"] is compiled
    assert end["exception"] is None


def test_bettertogether_nests_child_compile_callback():
    class Student(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.predictor(question=question)

    class ChildOptimizer(Teleprompter):
        def compile(self, student, *, trainset):
            return student

    callback = RecordingCallback()
    child = ChildOptimizer()
    optimizer = BetterTogether(metric=lambda example, prediction, trace=None: 1.0, child=child)
    student = Student()
    student.set_lm(DummyLM([{"answer": "answer"}]))
    trainset = [dspy.Example(question="question", answer="answer").with_inputs("question")]

    with (
        patch("dspy.teleprompt.bettertogether.eval_candidate_program", return_value=Mock(score=1.0)),
        patch("dspy.teleprompt.bettertogether.launch_lms"),
        patch("dspy.teleprompt.bettertogether.kill_lms"),
        dspy.context(callbacks=[callback]),
    ):
        optimizer.compile(student, trainset=trainset, valset=trainset, strategy="child")

    starts = events_for(callback, "compile_start")
    ends = events_for(callback, "compile_end")
    assert len(starts) == len(ends) == 2
    assert starts[0]["instance"] is optimizer
    assert starts[0]["parent_call_id"] is None
    assert starts[1]["instance"] is child
    assert starts[1]["parent_call_id"] == starts[0]["call_id"]
    assert ends[0]["call_id"] == starts[1]["call_id"]
    assert ends[0]["parent_call_id"] == starts[0]["call_id"]
    assert ends[1]["call_id"] == starts[0]["call_id"]
    assert ends[1]["parent_call_id"] is None


def test_bettertogether_reports_child_failure_but_completes_outer_run():
    expected_error = RuntimeError("child compile failed")

    class Student(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.predictor(question=question)

    class FailingChildOptimizer(Teleprompter):
        def compile(self, student, *, trainset):
            raise expected_error

    callback = RecordingCallback()
    child = FailingChildOptimizer()
    optimizer = BetterTogether(metric=lambda example, prediction, trace=None: 1.0, child=child)
    student = Student()
    student.set_lm(DummyLM([{"answer": "answer"}]))
    trainset = [dspy.Example(question="question", answer="answer").with_inputs("question")]

    with (
        patch("dspy.teleprompt.bettertogether.eval_candidate_program", return_value=Mock(score=1.0)),
        patch("dspy.teleprompt.bettertogether.launch_lms"),
        patch("dspy.teleprompt.bettertogether.kill_lms"),
        dspy.context(callbacks=[callback]),
    ):
        compiled = optimizer.compile(student, trainset=trainset, valset=trainset, strategy="child")

    starts = events_for(callback, "compile_start")
    ends = events_for(callback, "compile_end")
    assert len(starts) == len(ends) == 2
    assert starts[1]["instance"] is child
    assert starts[1]["parent_call_id"] == starts[0]["call_id"]
    assert ends[0]["call_id"] == starts[1]["call_id"]
    assert ends[0]["outputs"] is None
    assert ends[0]["exception"] is expected_error
    assert ends[1]["call_id"] == starts[0]["call_id"]
    assert ends[1]["outputs"] is compiled
    assert ends[1]["exception"] is None
    assert compiled.flag_compilation_error_occurred is True
