import os
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Any, Dict, List

import pydantic
import pytest
import yaml

import dspy

JUDGE_MODEL_NAME = "judge"


def assert_program_output_correct(program_output: Any, grading_guidelines: str):
    """
    With the help of an LLM judge, assert that the specified output of a DSPy program is correct,
    according to the specified grading guidelines.

    Args:
        program_output: The output of a DSPy program.
        grading_guidelines: The grading guidelines for judging the correctness of the
                            program output.
    """
    with _judge_dspy_configuration():
        judge_response = _get_judge_program()(
            program_output=str(program_output),
            guidelines=grading_guidelines,
        ).judge_response
        assert judge_response.correct, f"Output: {program_output}. Reason incorrect: {judge_response.justification}"


def known_failing_models(models: List[str]):
    """
    Decorator to allow specific test cases to fail for certain models. This is useful when a
    model is known to be unable to perform a specific task (e.g. output formatting with complex
    schemas) to the required standard.

    Args:
        models: List of model names for which the test case is allowed to fail.
    """

    def decorator(test_func):
        test_func._known_failing_models = models

        @wraps(test_func)
        def wrapper(*args, **kwargs):
            return test_func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def _judge_dspy_configuration():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(module_dir, "reliability_conf.yaml")
    reliability_conf = parse_reliability_conf_yaml(conf_path)
    judge_params = reliability_conf.models.get(JUDGE_MODEL_NAME)
    if judge_params is None:
        raise ValueError(f"No LiteLLM configuration found for judge model: {JUDGE_MODEL_NAME}")

    with dspy.settings.context(lm=dspy.LM(**judge_params)):
        yield


def _get_judge_program():
    class JudgeResponse(pydantic.BaseModel):
        correct: bool = pydantic.Field("Whether or not the judge output is correct")
        justification: str = pydantic.Field("Justification for the correctness of the judge output")

    class JudgeSignature(dspy.Signature):
        program_output: str = dspy.InputField(description="The output of an AI program / model to be judged")
        guidelines: str = dspy.InputField(
            description=(
                "Grading guidelines for judging the correctness of the program output."
                " If the output satisfies the guidelines, the judge will return correct=True."
            )
        )
        judge_response: JudgeResponse = dspy.OutputField()

    return dspy.Predict(JudgeSignature)


class ReliabilityTestConf(pydantic.BaseModel):
    adapter: str
    models: Dict[str, Any]


@lru_cache(maxsize=None)
def parse_reliability_conf_yaml(conf_file_path: str) -> ReliabilityTestConf:
    try:
        with open(conf_file_path, "r") as file:
            conf = yaml.safe_load(file)

        model_dict = {}
        for conf_entry in conf["model_list"]:
            model_name = conf_entry.get("model_name")
            if model_name is None:
                raise ValueError("Model name missing in reliability_conf.yaml")

            litellm_params = conf_entry.get("litellm_params")
            if litellm_params is not None:
                model_dict[model_name] = litellm_params
            else:
                print(
                    f"Skipping all test cases for model {model_name} without LiteLLM parameters"
                    f" ('litellm_params' section of conf file is missing)."
                )

        adapter = conf.get("adapter")
        if adapter is None:
            raise ValueError("No adapter configuration found in reliability_conf.yaml")

        return ReliabilityTestConf(adapter=adapter, models=model_dict)
    except Exception as e:
        raise ValueError(f"Error parsing LiteLLM configuration file: {conf_file_path}") from e
