from typing import Any, Callable, Dict, List, Optional, Union

import dspy
from dspy import logger
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import signature_to_template


#-------------------------------------------------------------------------------
#    Templates for the user-facing strings used by this module
#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------
#    Helper functions
#-------------------------------------------------------------------------------

# TODO: Shared below are useful functions. Similar procedures are implemented
# separately and used by other DSPy teleprompters. These can be moved to shared
# locations.
def prepare_teacher(
        student: Program,
        teacher: Program = None
    ) -> Union[Program, AssertionError]:
    """Prepare the teacher program with respect to the student program.
    
    Args:
        student: The student program.
        teacher: The teacher program. If `None`, a copy of the student program
            is used as the teacher. Defaults to `None`.

    Returns:
        The copied teacher program.
    
    Raises:
        AssertionError: If the teacher is not an instance of the Program class.
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


def build_prompt_completion_data_from_trace(
        trace: List[Dict],
        exclude_demos: bool=False,
        try_to_record_lm_kwargs: bool = False,
        program: Program = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
    """Build prompt completion data from a given trace.
  
    Args:
        trace: The trace from which the prompt-completion data will be built.
        exclude_demos: Exclude the demos from the prompts even if they are
            present in the trace. Defaults to `False`.
        try_to_record_lm_kwargs: Whether to record the LM kwargs in the data.
            Defaults to `False`. If set, the `lm_kwargs` field of the LM used to
            generate the prompt-completion pair is included in the data. To
            find the LM, we first check if the predictor that generated the
            prompt-completion pair has an LM field set (`predictor.lm`). If it
            does, we record it's kwargs. If it doesn't, we get the kwargs from
            `dspy.settings.lm`. If `dspy.settings.lm` is not set either, this
            function will not record the LM kwargs.
        program: Optional argument used to infer the name of the predictor that
            generated the prompt-completion pair. If provided, the returned data
            will include the `predictor_name` field. If not provided, the
            `predictor_name` field is not included in the data, but the caller
            of this function can recover this information by using the
            `predictor_ind` field that' included in the data by default, by
            building the following dictionary:

            {ind: n for ind, (n, _) in enumerate(program.named_predictors())}

            where `ind` is the index of the predictor in the list returned by
            the `named_predictors()` method of the program. Defaults to `None`.

    Returns:
        Data as a list of dictionaries with the keys `prompt`, `completion` and
        optionally with the keys `predictor_name` and `lm_kwargs`. For a given
        prompt-completion pair:
        - The `prompt` field corresponds to the prompt.
        - The `completion` field corresponds to the completion.
        - The `predictor_ind` field corresponds to the index of the predictor
          in the predictor list 
        - The `predictor_name` field corresponds to the index of the predictor
          in the list returned by the the named_predictor() method of the
          program used to generate the trace. This field is included only if
          the `pred_ind_to_name` argument is provided.
        - The `lm_kwargs` field corresponds to the LM kwargs that generated the
          prompt-completion pair. Included only if the `record_lm_kwargs` is
          set and there is an active LM.
    """
    # If the program is provided, build the predictor index to name mapping
    if program:
        pred_ind_to_name = {
            ind: name for ind, (name, _) in enumerate(program.named_predictors())
        }

    # Build the prompt-completion data
    data = []
    for pred_ind, (pred, inputs, outputs) in enumerate(trace):
        # Get the demos from the predictor if exclude_demos is False
        demos = [] if exclude_demos else pred.demos

        # Build prompt and completion strings
        template = signature_to_template(pred.signature)
        prompt = template(Example(demos=demos, **inputs))
        completion = template.query(Example(**outputs))
        
        # TODO: This part of the code could be improved.
        # The method we use to build the completion (template.query) is meant to
        # be used for creating, well, queries, and hence contains field prefixes
        # (e.g. "Reasoning: Let's think step by step in order to"), which are
        # also contained in the last piece of the prompt (separated with a new
        # line) We remove this piece from the completion. This is a hacky
        # solution since it assumes a particular template format.
        prompt_last = prompt.split("\n")[-1]
        completion = completion[len(prompt_last):]

        # Create prompt-completion dictionary and add it to the data; optionally
        # add the predictor_name key as well as the lm_kwargs.
        data_dict = dict(prompt=prompt, completion=completion)
        
        # Record the predictor index and optionally, name
        data_dict['predictor_ind'] = pred_ind
        if program:
            data_dict['predictor_name'] = pred_ind_to_name[pred_ind]

        # Optionally, record the LM kwargs
        lm = pred.lm or dspy.settings.lm
        if try_to_record_lm_kwargs and lm:
            data_dict['lm_kwargs'] = lm.kwargs
        data.append(data_dict)

    return data


def convert_to_module_level_prompt_completion_data(
        data: List[Dict],
        keep_data_keys: bool = False,
        exclude_demos: bool = False,
        try_to_record_lm_kwargs: bool = False,
        program: Program = None
    ) -> List[Dict]:
    """Convert the data to prompt-completion data using the "trace" field.

    This function is a wrapper around the function
    `build_prompt_completion_data_from_trace`, calling it on the "trace" field
    of each dictionary in the input data list and combiningin the results into
    a list of prompt-completion data dictionaries. If the `keep_data_keys`
    is set, the original data keys are also copied over to in the
    prompt-completion dictionaries.

    For example, if the input data includes 10 dictionaries, each containing a
    "trace" field generated by a program with 3 predictors, the returned data
    will have 30 prompt-completion data dictionaries.

    Args:
        data: List of data dictionaries to be converted to prompt-completion
            data. Each dictionary in the list should contain a "trace" field,
            which is passed to the `build_prompt_completion_data_from_trace`
            function to generate the prompt-completion data.
        keep_data_keys: Whether to keep the original data keys in the
            prompt-completion data. Note that if there are keys that are common
            between the original data and the prompt-completion data returned by
            `build_prompt_completion_data_from_trace`, the values from the
            prompt-completion data will overwrite the values from the original
            data. Defaults to `False`.
        exclude_demos: Passed to `build_prompt_completion_data_from_trace`.
            Defaults to `False`.
        try_to_record_lm_kwargs: Passed to
            `build_prompt_completion_data_from_trace`. Defaults to `False`.
        program: Passed to `build_prompt_completion_data_from_trace`.
            Defaults to `None`.
    """
    prompt_completion_data = []
    for data_dict in data:
        trace = data_dict["trace"]
        trace_prompt_comletion_data = build_prompt_completion_data_from_trace(
            trace=trace, exclude_demos=exclude_demos,
            try_to_record_lm_kwargs=try_to_record_lm_kwargs, program=program
        )
        for prompt_completion_dict in trace_prompt_comletion_data:
            if keep_data_keys:
                prompt_completion_dict = {**data_dict, **prompt_completion_dict}
            prompt_completion_data.append(prompt_completion_dict)
    return prompt_completion_data


def bootstrap_data(
        program: Program,
        dataset: List[Example],
        metric: Optional[Callable[
            [Example, Prediction, Optional[List]], Union[bool, int, float]
        ]] = None,
        num_threads = 1,
    ) -> List[Dict[str, Any]]:
    """Bootstrap prediction and trace data for the program using the dataset.
    
    Args:
        program: The program that will be used to generate the traces for data
            collection.
        dataset: The dataset to be used for data collection.
        metric: The optional metric to be used to get a score for the example,
            recorded in a `score` field in the data. If the metric is not
            provided, the `score` field is not included in the data. Defaults
            to `None`.
        num_threads: The number of threads to be used for data collection.
            Defaults to `1`.
    
    Returns:
        Data as a list of dictionaries with the keys `example`, `prediction`,
        `trace`, and optionally, `score` fields. For a given example:
        - The `example` field corresponds to the example itself.
        - The `example_ind` field corresponds to the index of the example in the
            `dataset`.
        - The `prediction` field corresponds to the prediction made by the
            program on the example.
        - The `trace` field corresponds to the trace generated by the program
            on the example.
        - The `score` field corresponds to the metric score of the example, if
            the metric is provided. Otherwise, it is not included in the data.
    """
    data = []

    # Use Evaluate to call the program have the responses cached
    cname = program.__class__.__name__
    info = _INFO_BOOTSTRAP_DATA.format(len(dataset), cname, num_threads)
    logger.info(info)
    evaluator = Evaluate(
        devset=dataset, num_threads=num_threads, display_progress=True
    )
    evaluator(program, metric=metric)

    # Re-iterate over the dataset to build the cached prompt-completion data
    for example_ind, example in enumerate(dataset):

        # Run the program on the example
        with dspy.context(trace=[]):
            prediction = program(**example.inputs())
            trace = dspy.settings.trace
            score = metric(example, prediction, trace) if metric else None

        # Build the data dictionary and extend the data list
        data_dict = dict(example=example, prediction=prediction, trace=trace)
        data_dict['example_ind'] = example_ind
        if metric:
            data_dict['score'] = score
        data.append(data_dict)
    
    return data


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
    ) -> Union[List[Dict], AssertionError]:
    """ Bootstrap data for the given sampling round.

    This is a wrapper function around the `bootstrap_data` function that allows
    for collecting data for the given `sampling_round`. Due to the way caching
    works, one cannot get different completions for the same prompt just by
    querying an LM again, despite using a high sampling temperature. This
    function is a workaround to get different completions for the same prompt
    by modifying the sampling temperature of the LM for the specified
    `sampling_round`. The temperature of an LM is set to the following value
    for the given `sampling_round`:
    
        sampling_temperature + sampling_temperature_delta * sampling_round

    If a `sampling_temperature` of `None` is passed, the already set temperature
    of the LM is used as the sampling temperature instead.

    To sample different completions for the same prompt, this sampling can be
    called multiple times with different `sampling_round` values. For example:
    ```
    num_rounds = 5
    data = []
    for round in range(num_rounds):
        data += bootstrap_data_for_round(
            program, dataset, metric=metric, sampling_round=round
        )
        # Any dataset filtering for the next round can be done here, if needed
    ```

    The dataset filtering can become a powerful tool. For example, it can be
    used to filter out the examples that have already had enough high scoring
    completions generated for them. Here is an illustration:
    ```
    num_correct_target = 3
    correct_score = 1
    num_rounds = 5
    data = []
    for round in range(num_rounds):
        data += bootstrap_data_for_round(
            program, dataset, metric=metric, sampling_round=round
        )
        correct_bootstraps = [d for d in data if d['score'] == correct_score]
        correct_counts = Counter([d['example'] for d in correct_bootstraps])
        dataset = [d for d in dataset if correct_counts[d] < num_correct_target]
    ```

    Args:
        program: The program that will be used to generate the traces for data
            collection.
        dataset: The dataset to be used for data collection.
        metric: The optional metric to be used to get a score for the example,
            recorded in a `score` field in the data. If the metric is not
            provided, the `score` field is not included in the data. Defaults to
            `None`.
        num_threads: The number of threads to be used for data collection.
            Defaults to `1`.
        sampling_round: The round index for which the data is being collected.
            Defaults to `0`.
        sampling_temperature: The sampling temperature to be used for round
            `0`. If a value of `None` is passed, the temperature of the LM is
            used as the sampling temperature instead. Defaults to a high
            temperature of `0.9`.
        sampling_temperature_delta: The small temperature delta that's added to
            the sampling temperature every increment of the round index to
            generate different completions for the same prompt. Defaults to
            `0.001`.

    Returns:
        Data as a list of dictionaries with the keys returned by
        the `bootstrap_data` function, descriptions for which are shared in the
        function documentation. This function adds the following extra field
        to the dictionaries:
        - The `round` field corresponds to the `sampling_round` argument.
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
    program._assert_lm_consistency()

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
        data = bootstrap_data(
            program, dataset, metric=metric, num_threads=num_threads
        )
    
    # Add the round information to the data
    for data_dict in data:
        data_dict["round"] = sampling_round

    return data
