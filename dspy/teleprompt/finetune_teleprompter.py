from typing import Any, Callable, Dict, List, Optional, Union

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Program
from dspy.utils.logging import logger


# TODO: Shared below are useful functions. Similar procedures are implemented
# separately and used by other DSPy teleprompters. These can be moved to shared
# locations.
def prepare_teacher(student: Program, teacher: Program = None) -> Program:
    """Prepare the teacher program with respect to the student program.
    Args:
        student: The student program.
        teacher: The teacher program. If `None`, a copy of the student program
            is used as the teacher. Defaults to `None`.
    """
    # If teacher is None, use a copy of the student program as the teacher
    if teacher is None:
        logger.info("No teacher provided. Using a copy of the student program as the teacher.")
        teacher = student.deepcopy()
    else:
        teacher = teacher.deepcopy()

    # Ensure that the student and teacher programs have the same structure
    logger.info("Ensuring that the student and teacher are are structurally equivalent.")
    student._assert_structural_equivalency(teacher)

    # Ensure that the predictors of the programs point to different objects
    logger.info("Ensuring that the student and teacher programs do not share predictors.")
    student._assert_no_shared_predictor(teacher)

    # Ensure that the LM consistency property is satisfied
    logger.info("Ensuring that the teacher program satisfies the LM consistency property.")
    teacher._assert_lm_consistency()

    # If the global LM is being used, set it to the LMs of the copied teacher
    # program predictors to to avoid handling the same edge cases later
    if dspy.settings.lm:
        teacher._set_all_predictor_lms(dspy.settings.lm)

    return teacher


def convert_to_module_level_message_data(
    data: List[Dict],
    keep_data_keys: bool = False,
    exclude_demos: bool = False,
    try_to_record_lm_kwargs: bool = False,
    program: Program = None,
) -> List[Dict]:
    """Wrapper around the function
    `build_messages_from_trace`, calling it on the "trace" field
    of each dictionary in the input data list and combiningin the results into
    a list of prompt-completion data dictionaries."""

    prompt_completion_data = []
    for data_dict in data:
        trace = data_dict["trace"]
        trace_prompt_comletion_data = build_messages_from_trace(
            trace=trace, exclude_demos=exclude_demos, try_to_record_lm_kwargs=try_to_record_lm_kwargs, program=program
        )
        for prompt_completion_dict in trace_prompt_comletion_data:
            if keep_data_keys:
                prompt_completion_dict = {**data_dict, **prompt_completion_dict}
            prompt_completion_data.append(prompt_completion_dict)
    return prompt_completion_data


def build_messages_from_trace(
    trace: List[Dict],
    exclude_demos: bool = False,
    try_to_record_lm_kwargs: bool = False,
    program: Program = None,
) -> Dict[str, List[Dict[str, Any]]]:
    messages = []
    # If the program is provided, build the predictor index to name mapping
    if program:
        pred_ind_to_name = {ind: name for ind, (name, _) in enumerate(program.named_predictors())}

    # Build the prompt-completion data

    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    data = []

    # TODO: Make sure that this works for multi-stage pipelines
    for pred_ind, (pred, inputs, outputs) in enumerate(trace):
        # Get the demos from the predictor if exclude_demos is False
        demos = [] if exclude_demos else pred.demos
        messages = adapter.format(pred.signature, demos, inputs)
        messages.append(
            adapter.format_turn(signature=pred.signature, values=outputs, role="assistant", incomplete=False)
        )
        data.append(messages)

    return data


def bootstrap_data(
    program: Program,
    dataset: List[Example],
    metric: Optional[Callable[[Example, Prediction, Optional[List]], Union[bool, int, float]]] = None,
    num_threads=1,
    max_errors: int = 0,
) -> List[Dict[str, Any]]:
    """Bootstrap example, prediction, trace, example_ind, score data for the program using the dataset."""
    data = []

    # Use Evaluate to call the program have the responses cached
    cname = program.__class__.__name__
    info = f"Bootstrapping data on {len(dataset)} examples with the program {cname}, with {num_threads} threads"
    logger.info(info)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        max_errors=max_errors,
        provide_traceback=True,
    )
    evaluator(program, metric=metric)

    data = []
    for example_ind, example in enumerate(dataset):
        data_dict = bootstrap_one_example(example=example, example_ind=example_ind, program=program, metric=metric)
        if data_dict is not None:
            data.append(data_dict)

    return data


def bootstrap_one_example(
    example: Any, example_ind: int, program: Program, metric: Optional[Callable] = None
) -> Dict[str, Any]:
    with dspy.context(trace=[]):
        prediction = program(**example.inputs())
        trace = dspy.settings.trace
        score = metric(example, prediction, trace) if metric else None

    data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
    if metric:
        data_dict["score"] = score

    return data_dict
