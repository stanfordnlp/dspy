from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

import dspy
import dspy.logger as logger
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
    assert student._assert_structural_equivalency(teacher)

    # Ensure that the predictors of the programs point to different objects
    logger.info(_INFO_SHARED_PREDICTOR)
    assert not student._assert_no_shared_predictor(teacher)

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
        program: Optional[Program] = None,
        exclude_demos: bool=False,
        record_lm_kwargs: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
    """Build prompt completion data from a given trace.
  
    Args:
        trace: The trace from which the prompt-completion data will be built.
        program: Optional argument corresponding to the program that generated
            the trace, used to augment the prompt-completion pairs with the
            names of the predictors that generated them. Defaults to `None`.
        exclude_demos: Exclude the demos from the prompts even if they are
            present in the trace. Defaults to `False`.
        record_lm_kwargs: Whether to record the LM kwargs in the data. Defaults
            to `False`.

    Returns:
        Data as a list of dictionaries with the keys `prompt`, `completion` and
        optionally with the keys `predictor_name` and `lm_kwargs`. For a given
        prompt-completion pair:
        - The `prompt` field corresponds to the prompt.
        - The `completion` field corresponds to the completion.
        - The `predictor_name` field corresponds to the name of the predictor
          that generated the prompt-completion pair.
        - The `lm_kwargs` field corresponds to the LM kwargs that generated the
          prompt-completion pair.
    """
    # If program is provided, build a mapping from predictor ids to names
    pred_id_to_name = None
    if program:
        pred_id_to_name = OrderedDict(
          (id(pred), name) for name, pred in program.named_predictors()
        )

    # Build the prompt-completion data
    data = []
    for pred, inputs, outputs in trace:
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
        
        if program:
            pred_name = pred_id_to_name[id(pred)]
            data_dict['predictor_name'] = pred_name
        if record_lm_kwargs:
            data_dict['lm_kwargs'] = pred.lm.kwargs
        data.append(data_dict)

    return data


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
    for example in dataset:

        # Run the program on the example
        with dspy.context(trace=[]):
            prediction = program(**example.inputs())
            trace = dspy.settings.trace
            score = metric(example, prediction, trace) if metric else None

        # Build the data dictionary and extend the data list
        data_dict = dict(example=example, prediction=prediction, trace=trace)
        if metric:
            data_dict['score'] = score
        data.append(data_dict)
    
    return data


# TODO: If we can ensure to pass the "round" information every time a call is
# issued to an LM, we can make repetitive un-cached calls to the same LM without
# modifying it's temperature. The `temperature_delta` argument can be removed
# then.
def bootstrap_data_multiple_rounds(
        program: Program,
        dataset: List[Example],
        metric: Optional[Callable[
            [Example, Prediction, Optional[List]], Union[bool, int, float]
        ]] = None,
        num_threads = 1,
        num_rounds: int = 1,
        sampling_temperature_base: Optional[float] = 0.9,
        sampling_temperature_delta: float = 0.001,
        next_round_dataset_callback: Optional[Callable[
            [List[Example], List[Dict[str, Any]]], List[Example]
        ]] = lambda x, y: x,
    ) -> List[Dict]:
    """ Bootstrap multiple rounds of data using the dataset and program.
    
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
        num_rounds: The number of rounds of data collection to be performed.
            Defaults to `1`.
        sampling_temperature_base: The sampling temperature to be used in the
            first round. If a value of `None` is passed, the already set
            temperature of the LM is used as the base sampling temperature
            instead. Defaults to a high temperature of `0.9`.
        sampling_temperature_delta: The small temperature difference utilized to
            generate different bootstrapped traces for the different rounds to
            circummvent caching. Defaults to `0.001`.
        next_round_dataset_callback: The callback function to be used to
            generate the dataset for the next round of data collection. The
            function should take the current dataset and the data from the
            current round as arguments and return the dataset for the next
            round, an example signature for which provided below. Defaults to a
            function that returns the current dataset, meaning that the default
            behavior of this function is to sample `num_rounds` bootstrapped
            examples for each example in the original dataset.
  
            ```
            def next_round_dataset_callback(
                dataset: List[Example],
                data: List[Dict[str, Any]]
            ) -> List[Example]:
                # Your code modifying the dataset here
                return dataset
            ```

    Returns:
        Data as a list of dictionaries with the keys `example`, `prediction`,
        `trace`, and optionally, `score` fields, along with the `round` field.
        For a given example:
        - The `example` field corresponds to the example itself.
        - The `prediction` field corresponds to the prediction made by the
            program on the example.
        - The `trace` field corresponds to the trace generated by the program on
            the example.
        - The `score` field corresponds to the metric score of the example, if
            the metric is provided. Otherwise, it is not included in the data.
        - The `round` field corresponds to the index of the round the particular
            data dictionary is collected in.
    """
    # Create a copy of the program and dataset so that they are not modified
    program = program.deepcopy()
    dataset = dataset.copy()

    # Ensure that the LM consistency is satisfied, push the global LM to the
    # predictors if dspy.settings.lm is set
    program.assert_lm_consistency()
    if dspy.settings.lm:
        program.set_all_predictor_lms(dspy.settings.lm)

    # Collect rounds of data collection with different sampling temperatures
    data = []
    for round_ind in range(num_rounds):
        # Adjust the LM temperatures of the program predictors
        for pred in program.predictors:
            # If a None temperature is passed, keep the predictor's temperature
            # as is for the first round
            temp = sampling_temperature_base
            temp = pred.lm.kwargs["temperature"] if temp is None else temp
            temp += sampling_temperature_delta * round_ind
            pred.lm = pred.lm.copy(temperature=temp)

        # Collect the data for the round, add the round index to the data
        # dictionaries, and extend the data list
        info = f"Round {round_ind + 1}/{num_rounds}"
        logger.info(info)
        round_data = bootstrap_data(
            program, dataset, metric=metric, num_threads=num_threads
        )
        for data_dict in round_data:
            data_dict["round"] = round_ind
        data += round_data

        # Update the dataset for the next round
        dataset = next_round_dataset_callback(dataset, round_data)

    return data

class DataCollectionCallback:
    def __init__(self, num_attempts=0, num_correct=1, max_attempts=1):
        self.num_attempts = num_attempts
        self.num_correct = num_correct
        self.max_attempts = max_attempts

    def move_on_callback_correct_with_max(self, dataset_copy, data):
        correct_counts = Counter(list(map(lambda x: x["example"], data)))
        # examples_from_data = map(lambda x: x["example"], data)
        examples_still_incorrect = [x for x in dataset_copy if correct_counts.get(x, 0) < self.num_correct]
        self.num_attempts += 1
        if self.num_attempts >= self.max_attempts:
            return []
        return examples_still_incorrect


callback = DataCollectionCallback(max_attempts=max_attempts)

dc_kwargs = {
    "include": None,
    "exclude_demos":True, 
    "temperature": base_temp,
    "temperature_delta":0.0001,
    "move_on_callback": callback.move_on_callback_correct_with_max,
    "num_threads": NUM_THREADS,
}

# results = bootstrap_multiple_prompt_completion_data(program, **kwargs)