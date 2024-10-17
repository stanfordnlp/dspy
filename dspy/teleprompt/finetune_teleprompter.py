from typing import Any, Callable, Dict, List, Optional, Union

import dspy
from dspy.utils.logging import logger
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.primitives.prediction import Prediction

_INFO_DEFAULT_TEACHER = """No teacher provided. Using a copy of the student \
program as the teacher."""

_INFO_STRUCTURAL_EQUIVALENCY = """Ensuring that the student and teacher are \
are structurally equivalent."""

_INFO_SHARED_PREDICTOR = """Ensuring that the student and teacher programs do \
not share predictors."""

_INFO_LM_CONSISTENCY = """Ensuring that the teacher program satisfies the LM \
consistency property."""

_INFO_BOOTSTRAP_DATA = """Bootstrapping data on {} examples with the program \
{}, with {} threads"""

# TODO: Shared below are useful functions. Similar procedures are implemented
# separately and used by other DSPy teleprompters. These can be moved to shared
# locations.
def prepare_teacher(
        student: Program,
        teacher: Program = None
    ) -> Program:
    """Prepare the teacher program with respect to the student program.
    Args:
        student: The student program.
        teacher: The teacher program. If `None`, a copy of the student program
            is used as the teacher. Defaults to `None`.
    """
    # If teacher is None, use a copy of the student program as the teacher
    if teacher is None:
        logger.info(_INFO_DEFAULT_TEACHER)
        teacher = student.deepcopy()
    else:
        teacher = teacher.deepcopy()

    # Ensure that the student and teacher programs have the same structure
    logger.info(_INFO_STRUCTURAL_EQUIVALENCY)
    student._assert_structural_equivalency(teacher)

    # Ensure that the predictors of the programs point to different objects
    logger.info(_INFO_SHARED_PREDICTOR)
    student._assert_no_shared_predictor(teacher)

    # Ensure that the LM consistency property is satisfied
    logger.info(_INFO_LM_CONSISTENCY)
    teacher._assert_lm_consistency()

    # If the global LM is being used, set it to the LMs of the copied teacher
    # program predictors to to avoid handling the same edge cases later
    if dspy.settings.lm:
        teacher._set_all_predictor_lms(dspy.settings.lm)

    return teacher


# TODO: fix docstring
def convert_to_module_level_message_data(
        data: List[Dict],
        keep_data_keys: bool = False,
        exclude_demos: bool = False,
        try_to_record_lm_kwargs: bool = False,
        program: Program = None
    ) -> List[Dict]:
    """Wrapper around the function
    `build_messages_from_trace`, calling it on the "trace" field
    of each dictionary in the input data list and combiningin the results into
    a list of prompt-completion data dictionaries."""

    prompt_completion_data = []
    for data_dict in data:
        trace = data_dict["trace"]
        trace_prompt_comletion_data = build_messages_from_trace(
            trace=trace, exclude_demos=exclude_demos,
            try_to_record_lm_kwargs=try_to_record_lm_kwargs, program=program
        )
        for prompt_completion_dict in trace_prompt_comletion_data:
            if keep_data_keys:
                prompt_completion_dict = {**data_dict, **prompt_completion_dict}
            prompt_completion_data.append(prompt_completion_dict)
    return prompt_completion_data

def build_messages_from_trace(
        trace: List[Dict],
        exclude_demos: bool=False,
        try_to_record_lm_kwargs: bool = False,
        program: Program = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
    messages = []
    # If the program is provided, build the predictor index to name mapping
    if program:
        pred_ind_to_name = {
            ind: name for ind, (name, _) in enumerate(program.named_predictors())
        }

    # Build the prompt-completion data

    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    data = []

    # TODO: Make sure that this works for multi-stage pipelines
    for pred_ind, (pred, inputs, outputs) in enumerate(trace):
        # Get the demos from the predictor if exclude_demos is False
        demos = [] if exclude_demos else pred.demos
        messages = adapter.format(pred.signature, demos, inputs)
        messages.append(adapter.format_turn(signature=pred.signature, values=outputs, role="assistant", incomplete=False))
        data.append(messages)

    return data

def dummy_metric(example, pred, trace=None, frac=1.0):
    return 1

def bootstrap_data(
        program: Program,
        dataset: List[Example],
        metric: Optional[Callable[
            [Example, Prediction, Optional[List]], Union[bool, int, float]
        ]] = dummy_metric,
        num_threads = 1,
        max_errors: int = 0
    ) -> List[Dict[str, Any]]:
    """Bootstrap example, prediction, trace, example_ind, score data for the program using the dataset."""
    data = []

    # Use Evaluate to call the program have the responses cached
    cname = program.__class__.__name__
    info = _INFO_BOOTSTRAP_DATA.format(len(dataset), cname, num_threads)
    logger.info(info)
    evaluator = Evaluate(
        devset=dataset, num_threads=num_threads, display_progress=True, max_errors=max_errors, provide_traceback=True
    )
    evaluator(program, metric=metric)

    data = []
    for example in dataset:
        data_dict = process_example(example, 0, program, metric)
        if data_dict is not None:
            data.append(data_dict)
    
    return data


def process_example(example: Any, example_ind: int, program: Callable, metric: Optional[Callable] = None) -> Dict[str, Any]:
    # print("Processing example:", example_ind)
    with dspy.context(trace=[]):
        # print("Running program...", example_ind)
        try:
            prediction = program(**example.inputs())
        except Exception as e:
            print(f"Error processing example {example_ind}: {e}")
            return None
        # print("Getting trace...", example_ind)
        trace = dspy.settings.trace
        # print("Getting score...", example_ind)
        score = metric(example, prediction, trace) if metric else None

    data_dict = {
        'example': example,
        'prediction': prediction,
        'trace': trace,
        'example_ind': example_ind
    }
    if metric:
        data_dict['score'] = score
    
    return data_dict

# TODO: If we can ensure to pass the "round" information every time a call is
# issued to an LM, we can make repetitive un-cached calls to the same LM without
# modifying it's temperature. This function can be removed then.
def bootstrap_data_for_round(
        program: Program,
        dataset: List[Example],
        metric: Optional[Callable[
            [Example, Prediction, Optional[List]], Union[bool, int, float]
        ]] = None,
        num_threads = 1,
        sampling_round: int = 0,
        sampling_temperature: Optional[float] = 0.9,
        sampling_temperature_delta: float = 0.001,
        max_errors: int = 0
    ) -> List[Dict]:
    """ Bootstrap data for the given sampling round.

    This is a wrapper function around the `bootstrap_data` function that allows
    for collecting data for the given `sampling_round`. Due to the way caching
    works, one cannot get different completions for the same prompt just by
    querying an LM again.
    """ 
    # TODO: [DSPy 2.5] Migrate to using "seed" for sampling
    # Helper function to adjust the temperature of the LM. If a None temperature
    # is passed, keep the LM's temperature as is as the base temperature, then
    # adjust the temperature for the given round.
    def copy_model_with_updated_temp(lm):
        temp = sampling_temperature
        temp = lm.kwargs["temperature"] if temp is None else temp
        temp = temp + sampling_temperature_delta * sampling_round
        return lm.copy(temperature=temp)

    # Ensure that the LM consistency is satisfied, which ensures that either (1)
    # the global LM is set or (2) all the predictors have an LM set.
    # TODO(isaac): Uncomment this line after the LM consistency property is
    # program._assert_lm_consistency()

    # Deepcopy the program and copy the dataset to avoid modifying the original
    program = program.deepcopy()
    dataset = dataset.copy()

    # Update the temperature of the LM for the given round
    context_lm = None
    if dspy.settings.lm:
        context_lm = copy_model_with_updated_temp(dspy.settings.lm)
    else:
        for pred in program.predictors():
            pred.lm = copy_model_with_updated_temp(pred.lm)

    # Collect the data for the given round
    with dspy.context(lm=context_lm):
        # print(context_lm.kwargs)
        data = bootstrap_data(
            program, dataset, metric=metric, num_threads=num_threads, max_errors=max_errors
        )
    
    # Add the round information to the data
    for data_dict in data:
        data_dict["round"] = sampling_round

    return data
