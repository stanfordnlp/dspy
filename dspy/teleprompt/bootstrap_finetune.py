from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import dspy
from dspy.adapters.base import Adapter
from dspy.clients.lm import LM, TrainingJob
from dspy.clients.utils_finetune import infer_data_format
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.predict.predict import Predict
from dspy.primitives.program import Program
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.logging import logger


class BootstrapFinetune(Teleprompter):

    # TODO(check with team)
    def __init__(
        self,
        metric: Optional[Callable] = None,
        multitask: bool = True,
        train_kwargs: Optional[Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]] = None,
        adapter: Optional[Union[Adapter, Dict[LM, Adapter]]] = None,
        num_threads: int = 6,
    ):
        """TODO: Docstring"""
        err = "This is an experimental optimizer."
        err += " Set `dspy.settings.experimental` to `True` to use it."
        assert dspy.settings.experimental, err

        self.metric = metric
        self.multitask = multitask
        self.train_kwargs = self.format_lm_dict_kwargs(train_kwargs)
        self.adapter = self.format_adapter(adapter)
        self.num_threads = num_threads
    
    @staticmethod
    def format_adapter(
        adapter: Optional[Union[Adapter, Dict[LM, Adapter]]]
    ) -> Union[Adapter, Dict[LM, Adapter]]:
        if Adapter is None or isinstance(adapter, Adapter):
            return defaultdict(lambda: adapter)

        assert isinstance(adapter, dict), f"Expected dict, got {type(adapter)}"
        for k, v in adapter.items():
            assert isinstance(k, LM), f"Expected {LM}, got {type(k)}"
            assert isinstance(v, Adapter), f"Expected {Adapter}, got {type(v)}"
        return adapter

    @staticmethod
    def format_lm_dict_kwargs(
        train_kwargs: Optional[Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]]
    ) -> Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]:
        train_kwargs = train_kwargs or {}
        assert isinstance(train_kwargs, dict), f"Expected dict, got {type(train_kwargs)}"

        # Validate the format of the train_kwargs
        kwarg_type = None
        for k, v in train_kwargs.items():
            if not kwarg_type:
                kwarg_type = type(k)
            assert kwarg_type == type(k)
            if isinstance(k, LM) or isinstance(k, str):
                assert isinstance(v, Dict), f"Expected {Dict}, got {type(v)}"
            else:
                raise ValueError("Invalid train_kwargs format")
        
        # If the train_kwargs is a single dict, convert it to an "LM" dict with
        # the same value for all LMs
        if kwarg_type == str:
            train_kwargs = defaultdict(lambda: train_kwargs)
        return train_kwargs
    
    def compile(
        self,
        student: Program,
        trainset: List[Example],
        teacher: Optional[Program] = None,
        **kwargs,  # TODO: Check with team
    ) -> Program:
        """TODO: Docstring"""
        # (1) Prepare the student and teacher programs
        student = prepare_student(student)
        teacher = prepare_teacher(student, teacher)

        # (2) Bootstrap data
        bootstrapped_data = bootstrap_data(program=teacher, dataset=trainset, metric=self.metric, num_threads=self.num_threads)

        # (3) Start fine-tune job(s)
        finetune_key_to_job = {}
        for pred_name, pred in student.named_predictors():
            finetune_key = (pred.lm, self.multitask)
            if pred_name not in finetune_key_to_job:
                job = self._create_pred_finetune_job(pred_name=pred_name, lm=pred.lm, student=student, bootstrapped_data=bootstrapped_data)
                finetune_key_to_job[finetune_key] = job
        
        # (4) Wait for the jobs to finish
        finetune_key_to_lm = {}
        for finetune_key, job in finetune_key_to_job.items():
            finetune_key_to_lm[finetune_key] = job.result()

        # (5) Assign the fine-tuned model(s) to the student program
        for pred_name, pred in student.named_predictors():
            finetune_key = (pred.lm, self.multitask)
            pred.lm = finetune_key_to_lm[finetune_key]
        return student

    def _create_pred_finetune_job(
            self,
            pred_name: str,
            lm: LM,
            student: Program,
            bootstrapped_data: Dict[str, Any]
        ) -> TrainingJob:
        train_kwargs = self.train_kwargs[lm]
        adapter = self.adapter[lm] or lm.infer_adapter()
        data_format = infer_data_format(adapter)
        pred_data = convert_to_predictor_data(data=bootstrapped_data, adapter=adapter, program=student)
        if not self.multitask:
            pred_data = [d for d in pred_data if d["pred_name"] == pred_name]
        pred_data = [d["pred_data"] for d in pred_data]
        job = lm.finetune(train_data=pred_data, train_kwargs=train_kwargs, data_format=data_format)
        return job


# Note: Shared below are useful functions for bootstrapping data for
# teleprompters, which can be moved to shared locations.
def convert_to_predictor_data(
    data: List[Dict],
    adapter: Optional[Adapter] = None,
    program: Optional[Program] = None,
    exclude_demos: bool = False,
    record_lm_kwargs: bool = False,
    keep_data_keys: bool = False,
) -> List[Dict]:
    default_adapter = dspy.settings.adapter or dspy.ChatAdapter()
    if not adapter:
        adapter = default_adapter
        adapter_name = adapter.__class__.__name__
        logger.info(f"No adapter is provided -- using {adapter_name} as the default adapter.")

    data = []
    for data_dict in data:
        trace = data_dict["trace"]
        trace_data = build_pred_data_from_trace(trace=trace, adapter=adapter, program=program, exclude_demos=exclude_demos, record_lm_kwargs=record_lm_kwargs)
        for pred_dict in trace_data:
            if keep_data_keys:
                pred_dict = {**data_dict, **pred_dict}
            data.append(pred_dict)
    return data


def build_pred_data_from_trace(
    trace: List[Dict],
    adapter: Optional[Adapter] = None,
    exclude_demos: bool = False,
    record_lm_kwargs: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    # TODO(feature): A trace is collected using a particular adapter. It would
    # be nice to get this adapter from the trace as opposed to passing it in as
    # an argument.
    if not adapter:
        adapter = dspy.settings.adapter or dspy.ChatAdapter()

    data = []
    for pred_ind, (pred, inputs, outputs) in enumerate(trace):
        # Get the demos from the predictor if exclude_demos is False
        demos = [] if exclude_demos else pred.demos
        pred_data = adapter.format_finetune_data(
            signature=pred.signature,
            demos=demos,
            inputs=inputs,
            outputs=outputs,
        )
        pred_data = dict(pred_data=pred_data, pred_ind=pred_ind)

        # Record the LM kwargs if an LM is available
        lm = pred.lm or dspy.settings.lm
        if record_lm_kwargs and lm:
            pred_data['lm_kwargs'] = lm.kwargs
        data.append(pred_data)

    return data


def bootstrap_data(
    program: Program,
    dataset: List[Example],
    metric: Optional[Callable] = None,
    num_threads=6,
) -> List[Dict[str, Any]]:
    # Use evaluate to utilize multiple threads
    evaluator = Evaluate(
        devset=dataset, num_threads=num_threads, display_progress=True,
        provide_traceback=True  # TODO(check with team)
    )
    evaluator(program, metric=metric)

    data = []
    for example_ind, example in enumerate(dataset):
        data_dict = bootstrap_one_example(
            example=example, program=program, metric=metric
        )
        data_dict["example_ind"] = example_ind
        data.append(data_dict)

    return data


# TODO(check with team)
def bootstrap_one_example(
    example: Example,
    program: Program,
    metric: Optional[Callable] = None
) -> Dict[str, Any]:
    with dspy.context(trace=[]):
        prediction = program(**example.inputs())
        trace = dspy.settings.trace
        score = metric(example, prediction, trace) if metric else None

    data_dict = dict(
        example=example,
        prediction=prediction,
        trace=trace,
    )
    if metric:
        data_dict["score"] = score

    return data_dict


# Note: Shared below are useful functions for preparing student/teacher programs
# Similar methods are implemented separately and used by other DSPy
# teleprompters. These can be moved to shared locations.
def prepare_student(student: Program) -> Program:
    # If there is a global LM, set it to the LMs of the student program
    # predictors
    if dspy.settings.lm:
        student.set_lm(dspy.settings.lm)

    logger.info("Ensuring that the student program satisfies the LM consistency property.")
    assert_lm_consistency(student)

    return student


def prepare_teacher(student: Program, teacher: Program = None) -> Program:
    if teacher is None:
        logger.info("No teacher provided. Using a copy of the student program as the teacher.")
        teacher = student.deepcopy()
    else:
        teacher = teacher.deepcopy()

    logger.info("Ensuring that the student and teacher are are structurally equivalent.")
    assert_structural_equivalency(student, teacher)

    logger.info("Ensuring that the student and teacher programs do not share predictors.")
    assert_no_shared_predictor(student, teacher)

    logger.info("Ensuring that the teacher program satisfies the LM consistency property.")
    assert_lm_consistency(teacher)

    # If there is a global LM, set it to the LMs of the copied teacher program
    # predictors
    if dspy.settings.lm:
        teacher.set_lm(dspy.settings.lm)

    return teacher


def assert_structural_equivalency(program1: object, program2: object):
    assert isinstance(program1, Program)
    assert isinstance(program2, Program)

    num1 = len(program1.predictors())
    num2 = len(program2.predictors())
    err = f"Structurally equivalent programs must have the the number of predictors. The number of predictors for the two modules do not match: {num1} != {num2}"
    assert num1 == num2, err

    pzip = zip(program1.named_predictors(), program2.named_predictors())
    for ind, ((name1, pred1), (name2, pred2)) in enumerate(pzip):
        err =  f"Program predictor names must match at  corresponding indices for structural equivalency. The predictor names for the programs do not match at index {ind}: '{name1}' != '{name2}'"
        assert name1 == name2, err
        assert isinstance(pred1, Predict)
        assert isinstance(pred2, Predict)
        assert pred1.signature.equals(pred2.signature)


def assert_no_shared_predictor(program1: Program, program2: Program):
    id_to_name1 = {id(p): n for n, p in program1.named_predictors()}
    id_to_name2 = {id(p): n for n, p in program2.named_predictors()}
    shared_ids = set(id_to_name1.keys()) & set(id_to_name2.keys())

    pred_names = ", ".join(id_to_name1[id] for id in shared_ids)
    err = f"The programs share the following predictor(s) with each other: {pred_names}"
    assert not shared_ids, err


def assert_lm_consistency(program: Program):
    global_flag = dspy.settings.lm is not None
    lm_info = get_lm_info(program)

    if global_flag:
        err = f"LM consistency violated: Expected all predictors' LMs to be 'NoneType' when 'dspy.settings.lm' is set, but some predictors have LMs set.\n\n{lm_info}"
        all_unset = all(pred.lm is None for pred in program.predictors())
        assert all_unset, err

    err = f"LM consistency violated: Expected all predictors’ LMs to be set when dspy.settings.lm is set to ‘NoneType’, but some predictors have 'NoneType' LMs.\n\n{lm_info}"
    all_set = all(pred.lm is not None for pred in program.predictors())
    assert not all_set, err


def get_lm_info(program: Program):
    lm_name = dspy.settings.lm.__class__.__name__
    info = f"The LM set in 'dspy.settings.lm' is an instance of '{lm_name}' LMs set for the predictors in the module are:"
    for pname, pred in program.named_predictors():
        lm_name = pred.lm.__class__.__name__
        info += f"\n    LM for {pname} is an instance of ’{lm_name}’"
    return info
