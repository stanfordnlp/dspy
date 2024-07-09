import random
import threading
from typing import Dict, Optional

import tqdm

import dsp
import dspy
from dspy.primitives import Example

from .teleprompt import Teleprompter
from .vanilla import LabeledFewShot

# TODO: metrics should return an object with __bool__ basically, but fine if they're more complex.
# They can also be sortable.

# TODO: Switch here from dsp.Example to dspy.Example. Right now, it's okay because it's internal only (predictors).
# NOTE: Notice the places where we don't shuffle examples. I do like that this one doesn't shuffle.
# Other ones that consider options may want to use both unshuffled and then shuffle a few times, when
# considering candidates.

# TODO: the max_rounds via branch_idx to get past the cache, not just temperature.
# In principle, we can also sample multiple outputs from the final generation step
# (or even each step, in case the validation function just wants *one* thing that works, but nah)
# and try them all. Having a pretty solid guess on the "final step" of each example isn't hard by the second round,
# in the sense that we have the trace from the first round. (Yes it may change but that's an edge case that
# won't hurt our "best effort" guarantees.)

# TODO: When this bootstraps for another teleprompter like finetune, we want all demos we gather.
# But when it's for direct use we may want to sample ONE demo per predictor--example pair.
# This is important for "multi-use" modules.

# TODO: Add baselines=[...]

class BootstrapFewShot(Teleprompter):
    def __init__(
        self,
        metric=None,
        metric_threshold=None,
        teacher_settings: Optional[Dict]=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        max_errors=5,
    ):
        """
        A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.
        These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

        Parameters
        ----------
        metric: Callable
            A function that compares an expected value and predicted value, outputting the result of that comparison. 
        metric_threshold: optional float, default `None`
            If the metric yields a numerical value, then check it against this threshold when
            deciding whether or not to accept a bootstrap example.
        teacher_settings: dict, optional
            Settings for the `teacher` model.
        max_bootstrapped_demos: int, default 4
            Maximum number of bootstrapped demonstrations to include
        max_labeled_demos: int, default 16
            Maximum number of labeled demonstrations to include.
        max_rounds: int, default 1
            Number of iterations to attempt generating the required bootstrap examples. If unsuccessful after `max_rounds`, the program ends.
        max_errors: int, default 5
            Maximum number of errors until program ends.
        """
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_settings = {} if teacher_settings is None else teacher_settings

        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()

    def compile(self, student, *, teacher=None, trainset):
        self.trainset = trainset

        self._prepare_student_and_teacher(student, teacher)
        self._prepare_predictor_mappings()
        self._bootstrap()

        self.student = self._train()
        self.student._compiled = True

        # set assert_failures and suggest_failures as attributes of student w/ value 0
        self.student._assert_failures = 0
        self.student._suggest_failures = 0

        return self.student

    def _prepare_student_and_teacher(self, student, teacher):
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.reset_copy()

        assert getattr(self.student, "_compiled", False) is False, "Student must be uncompiled."

        if self.max_labeled_demos and getattr(self.teacher, "_compiled", False) is False:
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = teleprompter.compile(self.teacher.reset_copy(), trainset=self.trainset)

    def _prepare_predictor_mappings(self):
        name2predictor, predictor2name = {}, {}
        student, teacher = self.student, self.teacher

        assert len(student.predictors()) == len(
            teacher.predictors(),
        ), "Student and teacher must have the same number of predictors."

        for (name1, predictor1), (name2, predictor2) in zip(student.named_predictors(), teacher.named_predictors()):
            assert name1 == name2, "Student and teacher must have the same program structure."
            if hasattr(predictor1.signature, "equals"):
                assert predictor1.signature.equals(
                    predictor2.signature,
                ), (f"Student and teacher must have the same signatures. "
                    f"{type(predictor1.signature)} != {type(predictor2.signature)}"
                    )
            else:
                # fallback in case if .equals is not implemented (e.g. dsp.Prompt)
                assert predictor1.signature == predictor2.signature, (
                    f"Student and teacher must have the same signatures. "
                    f"{type(predictor1.signature)} != {type(predictor2.signature)}"
                )
            assert id(predictor1) != id(predictor2), "Student and teacher must be different objects."

            name2predictor[name1] = None  # dict(student=predictor1, teacher=predictor2)
            predictor2name[id(predictor1)] = name1

            # FIXME(shangyint): This is an ugly hack to bind traces of
            # retry.module to retry
            # if isinstance(predictor1, Retry):
            #     predictor2name[id(predictor1.module)] = name1

            predictor2name[id(predictor2)] = name2

        self.name2predictor = name2predictor
        self.predictor2name = predictor2name

    def _bootstrap(self, *, max_bootstraps=None):
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos

        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2predictor}

        for round_idx in range(self.max_rounds):
            for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
                if len(bootstrapped) >= max_bootstraps:
                    break

                if example_idx not in bootstrapped:
                    success = self._bootstrap_one_example(example, round_idx)

                    if success:
                        bootstrapped[example_idx] = True

        print(
            f"Bootstrapped {len(bootstrapped)} full traces after {example_idx + 1} examples in round {round_idx}.",
        )

        # Unbootstrapped training examples

        self.validation = [x for idx, x in enumerate(self.trainset) if idx not in bootstrapped]
        random.Random(0).shuffle(self.validation)

        self.validation = self.validation

        # NOTE: Can't yet use evaluate because we need to trace *per example*
        # evaluate = Evaluate(program=self.teacher, metric=self.metric, num_threads=12)
        # score = evaluate(self.metric, display_table=False, display_progress=True)

    def _bootstrap_one_example(self, example, round_idx=0):
        name2traces = self.name2traces
        teacher = self.teacher  # .deepcopy()
        predictor_cache = {}

        try:
            with dsp.settings.context(trace=[], **self.teacher_settings):
                lm = dsp.settings.lm
                lm = lm.copy(temperature=0.7 + 0.001 * round_idx) if round_idx > 0 else lm
                new_settings = dict(lm=lm) if round_idx > 0 else {}

                with dsp.settings.context(**new_settings):
                    for name, predictor in teacher.named_predictors():
                        predictor_cache[name] = predictor.demos
                        predictor.demos = [x for x in predictor.demos if x != example]

                    prediction = teacher(**example.inputs())
                    trace = dsp.settings.trace

                    for name, predictor in teacher.named_predictors():
                        predictor.demos = predictor_cache[name]

                if self.metric:
                    metric_val = self.metric(example, prediction, trace)
                    if self.metric_threshold:
                        success = metric_val >= self.metric_threshold
                    else:
                        success = metric_val
                else:
                    success = True
        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count
            if current_error_count >= self.max_errors:
                raise e
            dspy.logger.error(f"Failed to run or to evaluate example {example} with {self.metric} due to {e}.")

        if success:
            for step in trace:
                predictor, inputs, outputs = step
                demo = Example(augmented=True, **inputs, **outputs)

                try:
                    predictor_name = self.predictor2name[id(predictor)]
                except KeyError:
                    continue  # FIXME: !

                    # # TODO: Look closer into this. It's a bit tricky to reproduce.
                    # print(f"Failed to find predictor {predictor} in {self.predictor2name}.")
                    # print(
                    #     "Are you doing this in a notebook (Jupyter)? This might be caused by redefining values by rerunning cells.",
                    # )
                    # print("Try restarting the notebook, or open an issue.")
                    # raise KeyError(
                    #     f"Failed to find predictor {id(predictor)} {predictor} in {self.predictor2name}.",
                    # ) from e

                name2traces[predictor_name].append(demo)

        return success

    def _train(self):
        rng = random.Random(0)
        raw_demos = self.validation

        for name, predictor in self.student.named_predictors():
            augmented_demos = self.name2traces[name][: self.max_bootstrapped_demos]

            sample_size = min(self.max_labeled_demos - len(augmented_demos), len(raw_demos))
            sample_size = max(0, sample_size)

            raw_demos = rng.sample(raw_demos, sample_size)

            if dspy.settings.release >= 20230928:
                predictor.demos = raw_demos + augmented_demos
            else:
                predictor.demos = augmented_demos + raw_demos

        return self.student
