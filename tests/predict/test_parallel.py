import dspy
from dspy.utils.dummies import DummyLM


def test_parallel_module():
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    dspy.configure(lm=lm)

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor2 = dspy.Predict("input -> output")

            self.parallel = dspy.Parallel(num_threads=2)

        def forward(self, input):
            return self.parallel(
                [
                    (self.predictor, input),
                    (self.predictor2, input),
                    (self.predictor, input),
                    (self.predictor2, input),
                    (self.predictor, input),
                ]
            )

    output = MyModule()(dspy.Example(input="test input").with_inputs("input"))

    expected_outputs = {f"test output {i}" for i in range(1, 6)}
    assert {r.output for r in output} == expected_outputs


def test_batch_module():
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    res_lm = DummyLM(
        [
            {"output": "test output 1", "reasoning": "test reasoning 1"},
            {"output": "test output 2", "reasoning": "test reasoning 2"},
            {"output": "test output 3", "reasoning": "test reasoning 3"},
            {"output": "test output 4", "reasoning": "test reasoning 4"},
            {"output": "test output 5", "reasoning": "test reasoning 5"},
        ]
    )

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor2 = dspy.Predict("input -> output, reasoning")

            self.parallel = dspy.Parallel(num_threads=2)

        def forward(self, input):
            with dspy.context(lm=lm):
                res1 = self.predictor.batch([input] * 5)

            with dspy.context(lm=res_lm):
                res2 = self.predictor2.batch([input] * 5)

            return (res1, res2)

    result, reason_result = MyModule()(dspy.Example(input="test input").with_inputs("input"))

    # Check that we got all expected outputs without caring about order
    expected_outputs = {f"test output {i}" for i in range(1, 6)}
    assert {r.output for r in result} == expected_outputs
    assert {r.output for r in reason_result} == expected_outputs

    # Check that reasoning matches outputs for reason_result
    for r in reason_result:
        num = r.output.split()[-1]  # get the number from "test output X"
        assert r.reasoning == f"test reasoning {num}"


def test_nested_parallel_module():
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    dspy.configure(lm=lm)

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor2 = dspy.Predict("input -> output")

            self.parallel = dspy.Parallel(num_threads=2)

        def forward(self, input):
            return self.parallel(
                [
                    (self.predictor, input),
                    (self.predictor2, input),
                    (
                        self.parallel,
                        [
                            (self.predictor2, input),
                            (self.predictor, input),
                        ],
                    ),
                ]
            )

    output = MyModule()(dspy.Example(input="test input").with_inputs("input"))

    # For nested structure, check first two outputs and nested outputs separately
    assert {output[0].output, output[1].output} <= {f"test output {i}" for i in range(1, 5)}
    assert {output[2][0].output, output[2][1].output} <= {f"test output {i}" for i in range(1, 5)}
    all_outputs = {output[0].output, output[1].output, output[2][0].output, output[2][1].output}
    assert len(all_outputs) == 4


def test_nested_batch_method():
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    dspy.configure(lm=lm)

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")

        def forward(self, input):
            res = self.predictor.batch([dspy.Example(input=input).with_inputs("input")] * 2)

            return res

    result = MyModule().batch([dspy.Example(input="test input").with_inputs("input")] * 2)

    assert {result[0][0].output, result[0][1].output, result[1][0].output, result[1][1].output} == {
        "test output 1",
        "test output 2",
        "test output 3",
        "test output 4",
    }


def test_batch_with_failed_examples():
    class FailingModule(dspy.Module):
        def forward(self, value: int) -> str:
            if value == 42:
                raise ValueError("test error")
            return f"success-{value}"

    module = FailingModule()

    examples = [
        dspy.Example(value=1).with_inputs("value"),
        dspy.Example(value=42).with_inputs("value"),  # This will fail
        dspy.Example(value=3).with_inputs("value"),
    ]

    results, failed_examples, exceptions = module.batch(
        examples,
        return_failed_examples=True,
        provide_traceback=True,
    )

    assert results == ["success-1", None, "success-3"]

    assert len(failed_examples) == 1
    assert failed_examples[0].inputs()["value"] == 42

    assert len(exceptions) == 1
    assert isinstance(exceptions[0], ValueError)
    assert str(exceptions[0]) == "test error"


def test_parallel_with_track_usage():
    """Test that usage tracking works correctly when using Parallel inside a Module."""
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
        ]
    )

    class MyModule(dspy.Module):
        def forward(self):
            predictor = dspy.Predict("input -> output")
            parallel = dspy.Parallel(num_threads=2)
            return parallel(
                [
                    (predictor, {"input": "test input 1"}),
                    (predictor, {"input": "test input 2"}),
                ]
            )

    # Test with track_usage enabled
    with dspy.settings.context(lm=lm, track_usage=True):
        results = MyModule()()

    # Verify that results are Prediction objects with usage data set
    assert len(results) == 2
    assert all(isinstance(r, dspy.Prediction) for r in results)
    assert all(r.get_lm_usage() is not None for r in results)

    # Verify predictions have the expected outputs
    expected_outputs = {"test output 1", "test output 2"}
    assert {r.output for r in results} == expected_outputs


def test_parallel_with_track_usage_and_tuple_return():
    """Test that usage tracking works when Module returns a tuple with list of Predictions."""
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
        ]
    )

    class MyModule(dspy.Module):
        def forward(self):
            predictor = dspy.Predict("input -> output")
            parallel = dspy.Parallel(num_threads=2)
            results = parallel(
                [
                    (predictor, {"input": "test input 1"}),
                    (predictor, {"input": "test input 2"}),
                ]
            )
            # Return tuple mimicking optimizer behavior (e.g., GEPA bootstrap tracing)
            return (results, {"trace": "some_trace_data"})

    # Test with track_usage enabled
    with dspy.settings.context(lm=lm, track_usage=True):
        output = MyModule()()

    # Verify that output is a tuple
    assert isinstance(output, tuple)
    assert len(output) == 2
    results, trace = output

    # Verify that results are Prediction objects with usage data set
    assert len(results) == 2
    assert all(isinstance(r, dspy.Prediction) for r in results)
    assert all(r.get_lm_usage() is not None for r in results)

    # Verify predictions have the expected outputs
    expected_outputs = {"test output 1", "test output 2"}
    assert {r.output for r in results} == expected_outputs
    
    # Verify trace is preserved
    assert trace == {"trace": "some_trace_data"}
