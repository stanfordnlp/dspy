import importlib
import time

import pytest

from dspy.utils.parallelizer import ParallelExecutor


def test_worker_threads_independence():
    def task(item):
        # Each thread maintains its own state by appending to a thread-local list
        return item * 2

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3)
    results = executor.execute(task, data)

    assert results == [2, 4, 6, 8, 10]


def test_parallel_execution_speed():
    def task(item):
        time.sleep(0.1)  # Simulate a time-consuming task
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=5)

    start_time = time.time()
    executor.execute(task, data)
    end_time = time.time()

    assert end_time - start_time < len(data)


def test_max_errors_handling():
    def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3, max_errors=1)

    with pytest.raises(Exception, match="Execution cancelled due to errors or interruption."):
        executor.execute(task, data)


def test_max_errors_not_met():
    def task(item):
        if item == 3:
            raise ValueError("Intentional error")
        return item

    data = [1, 2, 3, 4, 5]
    executor = ParallelExecutor(num_threads=3, max_errors=2)

    # Ensure that the execution completes without crashing when max_errors is not met
    results = executor.execute(task, data)

    # Verify that the results exclude the failed task
    assert results == [1, 2, None, 4, 5]

@pytest.mark.skipif(not importlib.util.find_spec("opentelemetry"), reason="OpenTelemetry not installed")
@pytest.mark.extra
def test_otel_context_propagation():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("parent_span") as parent_span:
        parent_span.set_attribute("test_attribute", "parent_value")
        parent_span_id = parent_span.get_span_context().span_id
        parent_trace_id = parent_span.get_span_context().trace_id

        seen_spans = []

        def task(item):
            with tracer.start_as_current_span("child_span") as child_span:
                child_span.set_attribute("test_attribute", "child_value")
                seen_spans.append(child_span)
            return item * 2

        executor = ParallelExecutor(num_threads=2, disable_progress_bar=True)
        executor.execute(task, [1, 2, 3])

        assert len(seen_spans) == 3

        for span in seen_spans:
            if span.name == "child_span":
                assert span.parent.span_id == parent_span_id
                assert span.get_span_context().trace_id == parent_trace_id
