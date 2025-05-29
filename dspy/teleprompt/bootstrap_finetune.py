import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.clients.utils_finetune import infer_data_format
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        metric: Optional[Callable] = None,
        multitask: bool = True,
        train_kwargs: Optional[Union[Dict[str, Any], Dict[LM, Dict[str, Any]]]] = None,
        adapter: Optional[Union[Adapter, Dict[LM, Adapter]]] = None,
        exclude_demos: bool = False,
        num_threads: Optional[int] = None,
    ):
        # TODO(feature): Inputs train_kwargs (a dict with string keys) and
        # adapter (Adapter) can depend on the LM they are used with. We are
        # takingthese as parameters for the time being. However, they can be
        # attached to LMs themselves -- an LM could know which adapter it should
        # be used with along with the train_kwargs. This will lead the only
        # required argument for LM.finetune() to be the train dataset.

        super().__init__(train_kwargs=train_kwargs)
        self.metric = metric
        self.multitask = multitask
        self.adapter: Dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.num_threads = num_threads

    def compile(
        self, student: Program, trainset: List[Example], teacher: Optional[Union[Program, List[Program]]] = None
    ) -> Program:
        # TODO: Print statements can be converted to logger.info if we ensure
        # that the default DSPy logger logs info level messages in notebook
        # environments.
        logger.info("Preparing the student and teacher programs...")
        all_predictors_have_lms(student)

        logger.info("Bootstrapping data...")
        trace_data = []

        teachers = teacher if isinstance(teacher, list) else [teacher]
        teachers = [prepare_teacher(student, t) for t in teachers]
        num_threads = self.num_threads or dspy.settings.num_threads
        for t in teachers:
            trace_data += bootstrap_trace_data(program=t, dataset=trainset, metric=self.metric, num_threads=num_threads)

        logger.info("Preparing the train data...")
        key_to_data = {}
        for pred_ind, pred in enumerate(student.predictors()):
            data_pred_ind = None if self.multitask else pred_ind
            training_key = (pred.lm, data_pred_ind)
            if training_key not in key_to_data:
                train_data, data_format = self._prepare_finetune_data(
                    trace_data=trace_data, lm=pred.lm, pred_ind=data_pred_ind
                )
                logger.info(f"Using {len(train_data)} data points for fine-tuning the model: {pred.lm.model}")
                finetune_kwargs = {
                    "lm": pred.lm,
                    "train_data": train_data,
                    "train_data_format": data_format,
                    "train_kwargs": self.train_kwargs[pred.lm],
                }
                key_to_data[training_key] = finetune_kwargs

        logger.info("Starting LM fine-tuning...")
        # TODO(feature): We could run batches of fine-tuning jobs in sequence
        # to avoid exceeding the number of threads.
        if len(key_to_data) > num_threads:
            raise ValueError(
                "BootstrapFinetune requires `num_threads` to be bigger than or equal to the number of fine-tuning "
                f"jobs. There are {len(key_to_data)} fine-tuning jobs to start, but the number of threads is: "
                f"{num_threads}! If the `multitask` flag is set to False, the number of fine-tuning jobs will "
                "be equal to the number of predictors in the student program. If the `multitask` flag is set to True, "
                "the number of fine-tuning jobs will be equal to: 1 if there is only a context LM, or the number of "
                "unique LMs attached to the predictors in the student program. In any case, the number of fine-tuning "
                "jobs will be less than or equal to the number of predictors."
            )
        logger.info(f"{len(key_to_data)} fine-tuning job(s) to start")
        key_to_lm = self.finetune_lms(key_to_data)

        logger.info("Updating the student program with the fine-tuned LMs...")
        for pred_ind, pred in enumerate(student.predictors()):
            data_pred_ind = None if self.multitask else pred_ind
            training_key = (pred.lm, data_pred_ind)
            finetuned_lm = key_to_lm[training_key]
            if isinstance(finetuned_lm, Exception):
                raise RuntimeError(f"Finetuned LM for predictor {pred_ind} failed.") from finetuned_lm
            pred.lm = finetuned_lm
            # TODO: What should the correct behavior be here? Should
            # BootstrapFinetune modify the prompt demos according to the
            # train data?
            pred.demos = [] if self.exclude_demos else pred.demos

        logger.info("BootstrapFinetune has finished compiling the student program")
        student._compiled = True
        return student

    @staticmethod
    def finetune_lms(finetune_dict) -> Dict[Any, LM]:
        num_jobs = len(finetune_dict)
        logger.info(f"Starting {num_jobs} fine-tuning job(s)...")
        # TODO(nit) Pass an identifier to the job so that we can tell the logs
        # coming from different fine-tune threads.

        key_to_job = {}
        for key, finetune_kwargs in finetune_dict.items():
            lm: LM = finetune_kwargs.pop("lm")
            # TODO: The following line is a hack. We should re-think how to free
            # up resources for fine-tuning. This might mean introducing a new
            # provider method (e.g. prepare_for_finetune) that can be called
            # before fine-tuning is started.
            logger.info(
                "Calling lm.kill() on the LM to be fine-tuned to free up resources. This won't have any effect if the "
                "LM is not running."
            )
            lm.kill()
            key_to_job[key] = lm.finetune(**finetune_kwargs)

        key_to_lm = {}
        for ind, (key, job) in enumerate(key_to_job.items()):
            result = job.result()
            if isinstance(result, Exception):
                raise result
            key_to_lm[key] = result
            job.thread.join()
            logger.info(f"Job {ind + 1}/{num_jobs} is done")

        return key_to_lm

    def _prepare_finetune_data(self, trace_data: List[Dict[str, Any]], lm: LM, pred_ind: Optional[int] = None):
        # TODO(nit) Log dataset details/size; make logs nicer
        if self.metric:
            logger.info(f"Collected data for {len(trace_data)} examples")
            trace_data = [d for d in trace_data if d["score"]]
            logger.info(f"After filtering with the metric, {len(trace_data)} examples remain")

        data = []
        adapter = self.adapter[lm] or settings.adapter or ChatAdapter()
        data_format = infer_data_format(adapter)
        for item in trace_data:
            for pred_ind, _ in enumerate(item["trace"]):
                include_data = pred_ind is None or pred_ind == pred_ind
                if include_data:
                    call_data = build_call_data_from_trace(
                        trace=item["trace"], pred_ind=pred_ind, adapter=adapter, exclude_demos=self.exclude_demos
                    )
                    data.append(call_data)

        import random

        random.Random(0).shuffle(data)

        return data, data_format


# Note: Shared below are useful functions for preparing student/teacher programs
# Similar methods are implemented separately and used by other DSPy
# teleprompters. These can be moved to shared locations.
def build_call_data_from_trace(
    trace: List[Dict],
    pred_ind: int,
    adapter: Adapter,
    exclude_demos: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    # Find data that's relevant to the predictor
    pred, inputs, outputs = trace[pred_ind]  # assuming that the order is kept

    demos = [] if exclude_demos else pred.demos
    call_data = adapter.format_finetune_data(
        signature=pred.signature,
        demos=demos,
        inputs=inputs,
        outputs=outputs,
    )
    return call_data


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: Union[float, None] = None


def bootstrap_trace_data(
    program: Program,
    dataset: List[Example],
    metric: Optional[Callable] = None,
    num_threads: Optional[int] = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
) -> List[Dict[str, Any]]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        return_outputs=True,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return metric(example, prediction, trace) if metric else True

    def wrapped_program(**kwargs):
        with dspy.context(trace=[]):
            try:
                return program(**kwargs), dspy.settings.trace.copy()
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature
                failed_inputs = kwargs

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                found_pred = None
                for pred in program.predictors():
                    if pred.signature == failed_signature:
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed signature: {failed_signature}")

                trace = dspy.settings.trace.copy()
                # Trace is Tuple[signature, inputs, prediction outputs]
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score) * (present / expected),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)

                trace.append(
                    (
                        found_pred,
                        failed_inputs,
                        failed_pred,
                    )
                )

                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following the adapter's formatting."
                    )

                return failed_pred, trace

    _, outputs = evaluator(wrapped_program, metric=wrapped_metric)

    data = []
    for example_ind, (example, prediction, score) in enumerate(outputs):
        try:
            prediction, trace = prediction
        except ValueError as ve:
            # TODO(GRPO Team): Often during GRPO bootstrapping, the LLM response does not follow dspy formatting. This leads to a value error.
            # To reproduce this issue, try Qwen/Qwen2.5-Coder-0.5B-Instruct with MATH dataset
            # Proposal(Lakshya): We should capture the incorrectly-formatted LLM response, and store it in the trace, and pass it to in the GRPO group
            # with a high-negative user-configurable score.
            logger.warning(
                "Failed to unpack prediction and trace. This is likely due to the LLM response not following dspy formatting."
            )
            if raise_on_error:
                raise ve
            else:
                continue
        data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data


# # TODO(PR) check with team
# def bootstrap_trace_data_one_example(
#     example: Example,
#     program: Program,
#     metric: Optional[Callable] = None
# ) -> Dict[str, Any]:
#     # Return a dict with the following keys:
#     #     example, prediction, trace, and score (if metric != None)
#     with dspy.context(trace=[]):
#         prediction = program(**example.inputs())
#         trace = dspy.settings.trace
#         score = metric(example, prediction, trace) if metric else None

#     data_dict = dict(
#         example=example,
#         prediction=prediction,
#         trace=trace,
#     )
#     if metric:
#         data_dict["score"] = score

#     return data_dict


# Note: Shared below are useful functions for preparing student/teacher programs
# Similar methods are implemented separately and used by other DSPy
# teleprompters. These can be moved to shared locations.
def all_predictors_have_lms(program: Program) -> bool:
    """Return True if all predictors in the program have an LM set."""
    return all(pred.lm for pred in program.predictors())


def copy_program_with_lms(program: Program) -> Program:
    pred_lms = [pred.lm for pred in program.predictors()]
    program = program.deepcopy()
    for ind, pred in enumerate(program.predictors()):
        pred.lm = pred_lms[ind]
    return program


def prepare_student(student: Program) -> Program:
    if getattr(student, "_compiled", False):
        raise ValueError("The student program should not be compiled.")

    # TODO: Should we use reset_copy here? How would it affect the student
    # program's predictor LMs, if they are set?

    # TODO: Should there be a deepcopy here?
    # student = student.deepcopy()
    return student


def prepare_teacher(student: Program, teacher: Optional[Program] = None) -> Program:
    if teacher is None:
        return student

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
        err = f"Program predictor names must match at  corresponding indices for structural equivalency. The predictor names for the programs do not match at index {ind}: '{name1}' != '{name2}'"
        assert name1 == name2, err
        assert isinstance(pred1, Predict)
        assert isinstance(pred2, Predict)


def assert_no_shared_predictor(program1: Program, program2: Program):
    id_to_name1 = {id(p): n for n, p in program1.named_predictors()}
    id_to_name2 = {id(p): n for n, p in program2.named_predictors()}
    shared_ids = set(id_to_name1.keys()) & set(id_to_name2.keys())

    pred_names = ", ".join(id_to_name1[id] for id in shared_ids)
    err = f"The programs share the following predictor(s) with each other: {pred_names}"
    assert not shared_ids, err


def get_unique_lms(program: Program) -> List[LM]:
    lms = [pred.lm for pred in program.predictors()]
    return list(set(lms))


def launch_lms(program: Program):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.launch()


def kill_lms(program: Program):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.kill()
