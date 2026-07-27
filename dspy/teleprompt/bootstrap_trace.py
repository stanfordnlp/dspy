import logging
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from dspy.dsp.utils.settings import settings
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None


def bootstrap_trace_data(
    program: Module,
    dataset: list[Example],
    metric: Callable | None = None,
    num_threads: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
) -> list[TraceData]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)

    def process_example(example):
        inputs = {**example.inputs()}
        trace = []
        try:
            with settings.context(trace=trace):
                prediction = program(**inputs)
        except AdapterParseError as e:
            present = list(e.parsed_result.keys()) if e.parsed_result else None
            expected = list(e.signature.output_fields.keys())

            found_pred = None
            for pred in program.predictors():
                if pred.signature == e.signature:
                    found_pred = pred
                    break
            if found_pred is None:
                raise ValueError(f"Failed to find the predictor for the failed signature: {e.signature}")

            if present:
                prediction = FailedPrediction(
                    completion_text=e.lm_response,
                    format_reward=format_failure_score + (failure_score - format_failure_score) * (present / expected),
                )
            else:
                prediction = FailedPrediction(completion_text=e.lm_response, format_reward=format_failure_score)

            trace.append((found_pred, inputs, prediction))

            if log_format_failures:
                logging.warning(
                    "Failed to parse output for example. This is likely due to the LLM response not following "
                    "the adapter's formatting."
                )

        if isinstance(prediction, FailedPrediction):
            score = prediction.format_reward or format_failure_score
        else:
            score = metric(example, prediction, None) if metric else True

        return {"prediction": prediction, "trace": trace, "score": score}

    executor = ParallelExecutor(
        num_threads=num_threads,
        disable_progress_bar=False,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
    )
    results = executor.execute(process_example, dataset)

    data = []
    for example_ind, (example, result) in enumerate(zip(dataset, results, strict=True)):
        if result is None:
            # TODO(GRPO Team): Often during GRPO bootstrapping, the LLM response does not follow dspy formatting.
            # To reproduce this issue, try Qwen/Qwen2.5-Coder-0.5B-Instruct with MATH dataset.
            # Proposal(Lakshya): We should capture the incorrectly-formatted LLM response, and store it in the trace,
            # and pass it to in the GRPO group with a high-negative user-configurable score.
            exception = executor.exceptions_map.get(example_ind)
            logger.warning("Failed to run the program on an example during bootstrapping: %r", exception)
            if raise_on_error:
                if exception is None:
                    raise RuntimeError(
                        f"Example {example_ind} failed during bootstrapping without a recorded exception."
                    )
                raise exception
            else:
                continue
        data_dict = {
            "example": example,
            "prediction": result["prediction"],
            "trace": result["trace"],
            "example_ind": example_ind,
        }
        if metric:
            data_dict["score"] = result["score"]
        data.append(data_dict)

    return data
