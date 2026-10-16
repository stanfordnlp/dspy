"""Cached-runtime timings, not model-quality or cold-install benchmarks."""

import json
import os
import statistics
import time

from cmpnd import CmpndCallback

import dspy
from scripts.mxc_interpreter.interpreter import MXCInterpreter


def measure(factory, runs=5):
    startup, warm, flex = [], [], []
    for _ in range(runs):
        started = time.perf_counter()
        worker = factory()
        try:
            assert worker.execute("19 + 23") == 42
            startup.append(1000 * (time.perf_counter() - started))
            for _ in range(10):
                started = time.perf_counter()
                assert worker.execute("19 + 23") == 42
                warm.append(1000 * (time.perf_counter() - started))
        finally:
            worker.shutdown()
        program = dspy.Flex("value: int -> result: int", interpreter_factory=factory)
        program._bind_code("""
class SumSquares(dspy.Module):
    def forward(self, **inputs):
        return dspy.Prediction(result=sum(x*x for x in range(inputs['value'])))
""")
        started = time.perf_counter()
        assert program(value=17).result == 1496
        flex.append(1000 * (time.perf_counter() - started))
    return {
        "startup_ms": startup,
        "warm_ms": warm,
        "flex_forward_ms": flex,
        "median_ms": {
            "startup": statistics.median(startup),
            "warm": statistics.median(warm),
            "flex_forward": statistics.median(flex),
        },
    }


if __name__ == "__main__":
    factories = {
        "local_unsandboxed_control": dspy.LocalInterpreter,
        "mxc": MXCInterpreter.configured(runtime=os.environ["MXC_EXECUTABLE"]),
        "deno_pyodide": dspy.PythonInterpreter,
    }
    with dspy.context(callbacks=[CmpndCallback(capture_inputs=False, capture_outputs=False)]):
        print(json.dumps({name: measure(factory) for name, factory in factories.items()}, indent=2))
