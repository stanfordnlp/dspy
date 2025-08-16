import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.clients.utils_finetune import infer_data_format
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


class FinetuneTeleprompter(Teleprompter):
    def __init__(
        self,
        train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
    ):
        self.train_kwargs: dict[LM, Any] = self.convert_to_lm_dict(train_kwargs or {})

    @staticmethod
    def convert_to_lm_dict(arg) -> dict[LM, Any]:
        non_empty_dict = arg and isinstance(arg, dict)
        if non_empty_dict and all(isinstance(k, LM) for k in arg.keys()):
            return arg
        # Default to using the same value for all LMs
        return defaultdict(lambda: arg)


class BootstrapFinetune(FinetuneTeleprompter):
    def __init__(
        self,
        metric: Callable | None = None,
        multitask: bool = True,
        train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
        adapter: Adapter | dict[LM, Adapter] | None = None,
        exclude_demos: bool = False,
        num_threads: int | None = None,
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
        self.adapter: dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.num_threads = num_threads

    def compile(
        self, student: Module, trainset: list[Example], teacher: Module | list[Module] | None = None
    ) -> Module:
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
            if pred.lm is None:
                raise ValueError(
                    f"Predictor {pred_ind} does not have an LM assigned. "
                    f"Please ensure the module's predictors have their LM set before fine-tuning. "
                    f"You can set it using: your_module.set_lm(your_lm)"
                )
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
    def finetune_lms(finetune_dict) -> dict[Any, LM]:
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

    def _prepare_finetune_data(self, trace_data: list[dict[str, Any]], lm: LM, pred_ind: int | None = None):
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
    trace: list[dict],
    pred_ind: int,
    adapter: Adapter,
    exclude_demos: bool = False,
) -> dict[str, list[dict[str, Any]]]:
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
    format_reward: float | None = None

class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None

import inspect

import inspect

# ProgramWrapper inherits from your Module class with ProgramMeta metaclass.

class ProgramWrapper(Module):
    """
    A transparent wrapper around a dspy Module that:
    - Inherits from Module.
    - Delegates behavior to the wrapped program, with an optional call_wrapper(program, **kwargs)
      used when calling.
    - Avoids recursion during Module's metaclass-driven initialization by handling attribute
      setting carefully before _program exists.
    """

    def __init__(self, program, call_wrapper=None):
        # Initialize Module (this sets callbacks/history on the wrapper instance).
        super().__init__(callbacks=getattr(program, "callbacks", None))
        # Set internal fields directly.
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "_call_wrapper", call_wrapper)
        # Make wrapper's callbacks/history reference the wrapped program's lists for consistency.
        try:
            object.__setattr__(self, "history", program.history)
        except Exception:
            pass
        try:
            object.__setattr__(self, "callbacks", program.callbacks)
        except Exception:
            pass

    # Core calling behavior
    def __call__(self, *args, **kwargs):
        """
        If call_wrapper is provided, try to adapt the call to kwargs and call
        call_wrapper(program, **kwargs). Otherwise, or if adaptation fails, call the
        wrapped program directly to preserve behavior.
        """
        if self._call_wrapper is None:
            return self._program(*args, **kwargs)

        # If single dict-like positional arg, treat it as kwargs.
        if args and not kwargs and len(args) == 1 and isinstance(args[0], dict):
            kwargs = dict(args[0])
            args = ()

        # Try binding args to program signature so we can pass only kwargs to call_wrapper.
        if args:
            try:
                sig = inspect.signature(self._program)
                bound = sig.bind_partial(*args, **kwargs)
                kwargs = dict(bound.arguments)
                args = ()
            except Exception:
                # Can't safely adapt; preserve exact behavior by calling program directly.
                return self._program(*args, **kwargs)

        return self._call_wrapper(self._program, **kwargs)

    async def acall(self, *args, **kwargs):
        if hasattr(self._program, "acall"):
            return await self._program.acall(*args, **kwargs)
        # Fallback to sync call if no async available
        return self.__call__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if hasattr(self._program, "forward"):
            return self._program.forward(*args, **kwargs)
        return self._program(*args, **kwargs)

    async def aforward(self, *args, **kwargs):
        if hasattr(self._program, "aforward"):
            return await self._program.aforward(*args, **kwargs)
        if hasattr(self._program, "acall"):
            return await self._program.acall(*args, **kwargs)
        return self.forward(*args, **kwargs)

    # Predictor/LM APIs: delegate to wrapped program to preserve fidelity.
    def named_predictors(self):
        return self._program.named_predictors() if hasattr(self._program, "named_predictors") else []

    def predictors(self):
        return self._program.predictors() if hasattr(self._program, "predictors") else []

    def set_lm(self, lm):
        if hasattr(self._program, "set_lm"):
            return self._program.set_lm(lm)
        return super().set_lm(lm)

    def get_lm(self):
        if hasattr(self._program, "get_lm"):
            return self._program.get_lm()
        return super().get_lm()

    def map_named_predictors(self, func):
        if hasattr(self._program, "map_named_predictors"):
            self._program.map_named_predictors(func)
            return self
        return super().map_named_predictors(func)

    def inspect_history(self, n: int = 1):
        if hasattr(self._program, "inspect_history"):
            return self._program.inspect_history(n)
        return super().inspect_history(n)

    def batch(
        self,
        examples,
        num_threads=None,
        max_errors=None,
        return_failed_examples=False,
        provide_traceback=None,
        disable_progress_bar=False,
    ):
        if hasattr(self._program, "batch"):
            return self._program.batch(
                examples,
                num_threads=num_threads,
                max_errors=max_errors,
                return_failed_examples=return_failed_examples,
                provide_traceback=provide_traceback,
                disable_progress_bar=disable_progress_bar,
            )
        return super().batch(
            examples,
            num_threads=num_threads,
            max_errors=max_errors,
            return_failed_examples=return_failed_examples,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )

    # Transparent attribute access with init-safe guards
    def __getattr__(self, name):
        # Only called if normal lookup fails. Avoid recursion before _program exists.
        if "_program" not in self.__dict__:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._program, name)

    def __setattr__(self, name, value):
        # During metaclass-initialization, _program isn't set yet; keep attributes local.
        if name in {"_program", "_call_wrapper"} or "_program" not in self.__dict__:
            object.__setattr__(self, name, value)
            return

        # If setting a data attribute that already lives on the wrapper instance, set locally.
        if name in getattr(self, "__dict__", {}):
            object.__setattr__(self, name, value)
            return

        # Delegate attribute setting to the wrapped program by default.
        setattr(self._program, name, value)

    def __delattr__(self, name):
        if name in {"_program", "_call_wrapper"} or "_program" not in self.__dict__:
            object.__delattr__(self, name)
            return

        if name in getattr(self, "__dict__", {}):
            object.__delattr__(self, name)
        else:
            delattr(self._program, name)

    # Introspection and representation
    def __dir__(self):
        own = set(dir(type(self))) | set(getattr(self, "__dict__", {}).keys())
        try:
            prog = set(dir(self._program))
        except Exception:
            prog = set()
        return sorted(own | prog)

    def __repr__(self):
        return repr(self._program)

    def __str__(self):
        return str(self._program)

    # Iteration support if the wrapped program is iterable
    def __iter__(self):
        return iter(self._program)

    # Equality and hashing delegate to the wrapped object
    def __eq__(self, other):
        return self._program == (other._program if isinstance(other, ProgramWrapper) else other)

    def __hash__(self):
        return hash(self._program)

    # Pickle support
    def __getstate__(self):
        # Let Module's __getstate__ do its normal pruning on the wrapper's dict,
        # but also include the wrapped program and call_wrapper explicitly.
        state = super().__getstate__()
        state["_program"] = self._program
        state["_call_wrapper"] = self._call_wrapper
        return state

    def __setstate__(self, state):
        # Restore wrapper internals first.
        object.__setattr__(self, "_program", state.pop("_program"))
        object.__setattr__(self, "_call_wrapper", state.pop("_call_wrapper"))
        # Restore the rest via Module's machinery.
        super().__setstate__(state)


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
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return metric(example, prediction, trace) if metric else True

    def wrapped_program_callable(program_to_use: Module, **kwargs):
        with dspy.context(trace=[]):
            try:
                return program_to_use(**kwargs), dspy.settings.trace.copy()
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature
                failed_inputs = kwargs

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                found_pred = None
                for pred in program_to_use.predictors():
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

    wrapped_program = ProgramWrapper(program, wrapped_program_callable)

    results = evaluator(wrapped_program, metric=wrapped_metric).results

    data = []
    for example_ind, (example, prediction, score) in enumerate(results):
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
# ) -> dict[str, Any]:
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
def all_predictors_have_lms(program: Module) -> bool:
    """Return True if all predictors in the program have an LM set."""
    return all(pred.lm for pred in program.predictors())


def copy_program_with_lms(program: Module) -> Module:
    pred_lms = [pred.lm for pred in program.predictors()]
    program = program.deepcopy()
    for ind, pred in enumerate(program.predictors()):
        pred.lm = pred_lms[ind]
    return program


def prepare_student(student: Module) -> Module:
    if getattr(student, "_compiled", False):
        raise ValueError("The student program should not be compiled.")

    # TODO: Should we use reset_copy here? How would it affect the student
    # program's predictor LMs, if they are set?

    # TODO: Should there be a deepcopy here?
    # student = student.deepcopy()
    return student


def prepare_teacher(student: Module, teacher: Module | None = None) -> Module:
    if teacher is None:
        return student

    # Ensuring that the student and teacher are are structurally equivalent
    assert_structural_equivalency(student, teacher)

    # Ensuring that the student and teacher programs do not share predictors
    assert_no_shared_predictor(student, teacher)

    return teacher


def assert_structural_equivalency(program1: object, program2: object):
    assert isinstance(program1, Module)
    assert isinstance(program2, Module)

    num1 = len(program1.predictors())
    num2 = len(program2.predictors())
    err = f"Structurally equivalent programs must have the the number of predictors. The number of predictors for the two modules do not match: {num1} != {num2}"
    assert num1 == num2, err

    pzip = zip(program1.named_predictors(), program2.named_predictors(), strict=False)
    for ind, ((name1, pred1), (name2, pred2)) in enumerate(pzip):
        err = f"Program predictor names must match at  corresponding indices for structural equivalency. The predictor names for the programs do not match at index {ind}: '{name1}' != '{name2}'"
        assert name1 == name2, err
        assert isinstance(pred1, Predict)
        assert isinstance(pred2, Predict)


def assert_no_shared_predictor(program1: Module, program2: Module):
    id_to_name1 = {id(p): n for n, p in program1.named_predictors()}
    id_to_name2 = {id(p): n for n, p in program2.named_predictors()}
    shared_ids = set(id_to_name1.keys()) & set(id_to_name2.keys())

    pred_names = ", ".join(id_to_name1[id] for id in shared_ids)
    err = f"The programs share the following predictor(s) with each other: {pred_names}"
    assert not shared_ids, err


def get_unique_lms(program: Module) -> list[LM]:
    lms = [pred.lm for pred in program.predictors()]
    return list(set(lms))


def launch_lms(program: Module):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.launch()


def kill_lms(program: Module):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.kill()
