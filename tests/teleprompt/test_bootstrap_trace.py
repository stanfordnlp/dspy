from unittest import mock

from litellm import Choices, Message, ModelResponse

import dspy
from dspy.primitives.example import Example
from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data


def test_bootstrap_trace_data():
    """Test bootstrap_trace_data function with single dspy.Predict program."""

    # Define signature for string -> int conversion
    class StringToIntSignature(dspy.Signature):
        """Convert a string number to integer"""

        text: str = dspy.InputField()
        number: int = dspy.OutputField()

    # Create program with single dspy.Predict
    program = dspy.Predict(StringToIntSignature)

    # Create dummy dataset of size 5
    dataset = [
        Example(text="one", number=1).with_inputs("text"),
        Example(text="two", number=2).with_inputs("text"),
        Example(text="three", number=3).with_inputs("text"),
        Example(text="four", number=4).with_inputs("text"),
        Example(text="five", number=5).with_inputs("text"),
    ]

    # Define exact match metric
    def exact_match_metric(example, prediction, trace=None):
        return example.number == prediction.number

    # Configure dspy
    dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())

    # Mock litellm completion responses
    # 4 successful responses and 1 that will trigger AdapterParseError
    successful_responses = [
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 1}\n```'))],
            model="openai/gpt-4o-mini",
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 2}\n```'))],
            model="openai/gpt-4o-mini",
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 3}\n```'))],
            model="openai/gpt-4o-mini",
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 4}\n```'))],
            model="openai/gpt-4o-mini",
        ),
    ]

    # Create a side effect that will trigger AdapterParseError on the 3rd call (index 2)
    def completion_side_effect(*args, **kwargs):
        call_count = completion_side_effect.call_count
        completion_side_effect.call_count += 1

        if call_count == 5:  # Third call (0-indexed)
            # Return malformed response that will cause AdapterParseError
            return ModelResponse(
                choices=[Choices(message=Message(content="This is an invalid JSON!"))],
                model="openai/gpt-4o-mini",
            )
        else:
            return successful_responses[call_count]

    completion_side_effect.call_count = 0

    with mock.patch("litellm.completion", side_effect=completion_side_effect):
        # Call bootstrap_trace_data
        results = bootstrap_trace_data(
            program=program,
            dataset=dataset,
            metric=exact_match_metric,
            raise_on_error=False,
            capture_failed_parses=True,
        )

    # Verify results
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    # Count successful and failed predictions
    successful_count = 0
    failed_count = 0

    for result in results:
        assert "example" in result
        assert "prediction" in result
        assert "trace" in result
        assert "example_ind" in result
        assert "score" in result

        if isinstance(result["prediction"], FailedPrediction):
            failed_count += 1
            # Verify failed prediction structure
            assert hasattr(result["prediction"], "completion_text")
            assert hasattr(result["prediction"], "format_reward")
            assert result["prediction"].completion_text == "This is an invalid JSON!"
        else:
            successful_count += 1
            # Verify successful prediction structure
            assert hasattr(result["prediction"], "number")

    # Verify we have the expected number of successful and failed bootstrapping
    assert successful_count == 4, f"Expected 4 successful predictions, got {successful_count}"
    assert failed_count == 1, f"Expected 1 failed prediction, got {failed_count}"

    # Verify that traces are present
    for result in results:
        assert len(result["trace"]) > 0, "Trace should not be empty"
        # Each trace entry should be a tuple of (predictor, inputs, prediction)
        for trace_entry in result["trace"]:
            assert len(trace_entry) == 3, "Trace entry should have 3 elements"
