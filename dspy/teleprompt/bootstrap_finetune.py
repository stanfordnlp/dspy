from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import dspy
from dspy import LM  # TODO: Remove after the old LM class is removed
from dspy.adapters.base import Adapter
from dspy.clients.utils_finetune import infer_data_format
from dspy.evaluate.evaluate import Evaluate
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.teleprompt.teleprompt import Teleprompter


class FinetuneTeleprompter(Teleprompter):

    def __init__(
        self,
        train_kwargs: Optional[Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]] = None,
    ):
        self.train_kwargs: Dict[LM, Any] = self.convert_to_lm_dict(train_kwargs or {})

    @staticmethod
    def convert_to_lm_dict(arg) -> Dict[LM, Any]:
        non_empty_dict = arg and isinstance(arg, dict)
        if non_empty_dict and all(isinstance(k, LM) for k in arg.keys()):
            return arg
        # Default to using the same value for all LMs
        return defaultdict(lambda: arg)


class BootstrapFinetune(FinetuneTeleprompter):

    # TODO(PR) check with team
    def __init__(
        self,
        metric: Optional[Callable] = None,
        multitask: bool = True,
        train_kwargs: Optional[Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]] = None,
        adapter: Optional[Union[Adapter, Dict[LM, Adapter]]] = None,
        exclude_demos: bool = False,
        num_threads: int = 6,
    ):
        # TODO(feature): Inputs train_kwargs (a dict with string keys) and
        # adapter (Adapter) can depend on the LM they are used with. We are
        # takingthese as parameters for the time being. However, they can be 
        # attached to LMs themselves -- an LM could know which adapter it should
        # be used with along with the train_kwargs. This will lead the only
        # required argument for LM.finetune() to be the train dataset.
        err = "This is an experimental optimizer."
        err += " Set `dspy.settings.experimental` to `True` to use it."
        err += " Constructor arguments subject to change."
        assert dspy.settings.experimental, err
        
        super().__init__(train_kwargs=train_kwargs)
        self.metric = metric
        self.multitask = multitask
        self.adapter: Dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.num_threads = num_threads
    
    def compile(self, student: Program, trainset: List[Example], teacher: Optional[Program] = None) -> Program:
        # TODO: Print statements can be converted to logger.info if we ensure
        # that the default DSPy logger logs info level messages in notebook
        # environments.
        print("[BootstrapFinetune] Preparing the student and teacher programs...")
        student = prepare_student(student)
        teacher = prepare_teacher(student, teacher)
        set_missing_predictor_lms(student)
        set_missing_predictor_lms(teacher)

        print("[BootstrapFinetune] Bootstrapping data...")
        trace_data = bootstrap_trace_data(program=teacher, dataset=trainset, metric=self.metric, num_threads=self.num_threads)

        print("[BootstrapFinetune] Preparing the train data...")
        key_to_data = {}
        for pred_ind, pred in enumerate(student.predictors()):
            data_pred_ind = None if self.multitask else pred_ind
            training_key = (pred.lm, data_pred_ind)
            if training_key not in key_to_data:
                train_data, data_format = self._prepare_finetune_data(trace_data=trace_data, lm=pred.lm, pred_ind=data_pred_ind)
                print(f"[BootstrapFinetune] Using {len(train_data)} data points for fine-tuning the model: {pred.lm.model}")
                finetune_kwargs = dict(lm=pred.lm, train_data=train_data, train_kwargs=self.train_kwargs[pred.lm], data_format=data_format)
                key_to_data[training_key] = finetune_kwargs
        
        print("[BootstrapFinetune] Starting LM fine-tuning...")
        # TODO(feature): We could run batches of fine-tuning jobs in sequence
        # to avoid exceeding the number of threads.
        err = f"BootstrapFinetune requires `num_threads` to be bigger than or equal to the number of fine-tuning jobs. There are {len(key_to_data)} fine-tuning jobs to start, but the number of threads is: {self.num_threads}! If the `multitask` flag is set to False, the number of fine-tuning jobs will be equal to the number of predictors in the student program. If the `multitask` flag is set to True, the number of fine-tuning jobs will be equal to: 1 if there is only a context LM, or the number of unique LMs attached to the predictors in the student program. In any case, the number of fine-tuning jobs will be less than or equal to the number of predictors."
        assert len(key_to_data) <= self.num_threads, err
        print(f"[BootstrapFinetune] {len(key_to_data)} fine-tuning job(s) to start")
        key_to_lm = self.finetune_lms(key_to_data)

        print("[BootstrapFinetune] Updating the student program with the fine-tuned LMs...")
        for pred_ind, pred in enumerate(student.predictors()):
            data_pred_ind = None if self.multitask else pred_ind
            training_key = (pred.lm, data_pred_ind)
            pred.lm = key_to_lm[training_key]
            # TODO: What should the correct behavior be here? Should
            # BootstrapFinetune modify the prompt demos according to the 
            # train data?
            pred.demos = [] if self.exclude_demos else pred.demos
        
        print("[BootstrapFinetune] BootstrapFinetune has finished compiling the student program")
        student._compiled = True
        return student

    @staticmethod
    def finetune_lms(finetune_dict) -> Dict[Any, LM]:
        num_jobs = len(finetune_dict)
        print(f"[BootstrapFinetune] Starting {num_jobs} fine-tuning job(s)...")
        # TODO(nit) Pass an identifier to the job so that we can tell the logs
        # coming from different fine-tune threads.

        key_to_job = {}
        for key, finetune_kwargs in finetune_dict.items():
            lm = finetune_kwargs.pop("lm")
            key_to_job[key] = lm.finetune(**finetune_kwargs)
        
        key_to_lm = {}
        for ind, (key, job) in enumerate(key_to_job.items()):
            key_to_lm[key] = job.result()
            job.thread.join()
            print(f"[BootstrapFinetune] Job {ind + 1}/{num_jobs} is done")

        return key_to_lm

    def _prepare_finetune_data(self, trace_data: List[Dict[str, Any]], lm: LM, pred_ind: Optional[int] = None):
        # TODO(nit) Log dataset details/size; make logs nicer
        if self.metric:
            print(f"[BootstrapFinetune] Collected data for {len(trace_data)} examples")
            trace_data = [d for d in trace_data if d["score"]]
            print(f"[BootstrapFinetune] After filtering with the metric, {len(trace_data)} examples remain")

        data = []
        adapter = self.adapter[lm] or lm.infer_adapter()
        data_format = infer_data_format(adapter)
        for item in trace_data:
            for pred_ind, _ in enumerate(item['trace']):
                include_data = pred_ind is None or pred_ind == pred_ind
                if include_data:
                    call_data = build_call_data_from_trace(trace=item['trace'], pred_ind=pred_ind,  adapter=adapter, exclude_demos=self.exclude_demos)
                    data.append(call_data)

        return data, data_format


def build_call_data_from_trace(
    trace: List[Dict],
    pred_ind: int,
    adapter: Optional[Adapter] = None,
    exclude_demos: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    # Find data that's relevant to the predictor
    pred, inputs, outputs = trace[pred_ind]  # assuming that the order is kept

    if not adapter:
        # TODO(feature): A trace is collected using a particular adapter. It
        # would be nice to get this adapter information from the trace (e.g.
        # pred.lm.adapter) as opposed to using the inference method below.
        adapter = pred.lm.infer_adapter()

    demos = [] if exclude_demos else pred.demos
    call_data = adapter.format_finetune_data(
        signature=pred.signature,
        demos=demos,
        inputs=inputs,
        outputs=outputs,
    )
    return call_data


def bootstrap_trace_data(
    program: Program,
    dataset: List[Example],
    metric: Optional[Callable] = None,
    num_threads=6,
) -> List[Dict[str, Any]]:
    # Return a list of dicts with the following keys:
    #     example_ind, example, prediction, trace, and score (if metric != None)
    evaluator = Evaluate(
        devset=dataset, num_threads=num_threads, display_progress=True,
        provide_traceback=True  # TODO(check with team)
    )
    # TODO(PR): Should "trace" not be included in the lambda function?
    _metric = metric if metric else lambda example, prediction: 1
    evaluator(program, metric=_metric)

    data = []
    for example_ind, example in enumerate(dataset):
        data_dict = bootstrap_trace_data_one_example(
            example=example, program=program, metric=metric
        )
        data_dict["example_ind"] = example_ind
        data.append(data_dict)

    return data


# TODO(PR) check with team
def bootstrap_trace_data_one_example(
    example: Example,
    program: Program,
    metric: Optional[Callable] = None
) -> Dict[str, Any]:
    # Return a dict with the following keys:
    #     example, prediction, trace, and score (if metric != None)
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
def set_missing_predictor_lms(program: Program) -> Program:
    # If the predictors do not have LMs, set them to the global LM
    for pred in program.predictors():
        if not pred.lm:
            pred.lm = dspy.settings.lm

    return program


def prepare_student(student: Program) -> Program:
    if getattr(student, "_compiled", False):
        raise ValueError("The student program should not be compiled.")

    # TODO: Should we use reset_copy here? How would it affect the student
    # program's predictor LMs, if they are set?
    student = student.deepcopy()
    return student


def prepare_teacher(student: Program, teacher: Program = None) -> Program:
    if teacher is None:
        return student.deepcopy()
    else:
        teacher = teacher.deepcopy()

    # Ensuring that the student and teacher are are structurally equivalent
    assert_structural_equivalency(student, teacher)

    # Ensuring that the student and teacher programs do not share predictors
    assert_no_shared_predictor(student, teacher)

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
