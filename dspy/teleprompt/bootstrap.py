import random
import threading
from typing import Callable, Dict, Optional

import tqdm

import dsp
import dspy
from dspy.primitives import Example
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.vanilla import LabeledFewShot

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
        metric: Optional[Callable] = None,
        metric_threshold: Optional[float] = None,
        teacher_settings: Optional[Dict] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        max_errors: int = 5,
    ):
        """
        A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.
        These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

        Args:
            metric (Callable): A function that compares an expected value and predicted value, outputting the result of
                that comparison.
            metric_threshold (float, optional): If the metric yields a numerical value, then check it against this
                threshold when deciding whether or not to accept a bootstrap example. Defaults to `None`.
            teacher_settings (dict, optional): Settings for the `teacher` model.
            max_bootstrapped_demos (int, optional): Maximum number of bootstrapped demonstrations to include. Defaults
                to 4.
            max_labeled_demos (int, optional): Maximum number of labeled demonstrations to include. Defaults to 16.
            max_rounds (int, optional): Number of iterations to attempt generating the required bootstrap examples. If
                unsuccessful after `max_rounds`, the program ends. Defaults to 1.
            max_errors (int, optional): Maximum number of errors until program ends. Defaults to 5.
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
        self._apply_bootstrap_results_on_student()

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

        for (student_predictor_name, student_predictor), (teacher_predictor_name, teacher_predictor) in zip(
            student.named_predictors(), teacher.named_predictors()
        ):
            assert (
                student_predictor_name == teacher_predictor_name
            ), "Student and teacher must have the same program structure."
            assert id(student_predictor) != id(teacher_predictor), "Student and teacher must be different objects."
            if hasattr(student_predictor.signature, "equals"):
                assert student_predictor.signature.equals(
                    teacher_predictor.signature,
                ), (
                    f"Student and teacher must have the same signatures. "
                    f"{type(student_predictor.signature)} != {type(teacher_predictor.signature)}"
                )
            else:
                # fallback in case if .equals is not implemented (e.g. dsp.Prompt)
                assert student_predictor.signature == teacher_predictor.signature, (
                    f"Student and teacher must have the same signatures. "
                    f"{type(student_predictor.signature)} != {type(teacher_predictor.signature)}"
                )

            name2predictor[student_predictor_name] = None
            predictor2name[id(student_predictor)] = student_predictor_name
            predictor2name[id(teacher_predictor)] = teacher_predictor_name

        self.name2predictor = name2predictor
        self.predictor2name = predictor2name

    def _bootstrap(self, *, max_bootstraps=None):
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos

        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2predictor}

        for round_idx in range(self.max_rounds):
            progbar = tqdm.tqdm(total=max_bootstraps, desc=f"Round {round_idx + 1}")
            for example_idx, example in enumerate(self.trainset):
                if len(bootstrapped) >= max_bootstraps:
                    break
                success = self._bootstrap_one_example(example, round_idx)

                if success:
                    bootstrapped[example_idx] = True
                    # Update the progress bar after each successful bootstrap.
                    progbar.update(1)
            progbar.close()

        dspy.logger.debug(
            f"Bootstrapped {len(bootstrapped)} full traces after {example_idx + 1} examples in round {round_idx}.",
        )
        # Examples not used in bootstrapping becomes the validation set.
        self.valset = [x for idx, x in enumerate(self.trainset) if idx not in bootstrapped]
        random.Random(0).shuffle(self.valset)

    def _bootstrap_one_example(self, example, round_idx=0):
        name2traces = self.name2traces
        teacher = self.teacher
        predictor_cache = {}

        try:
            with dsp.settings.context(trace=[], **self.teacher_settings):
                lm = dsp.settings.lm
                lm = lm.copy(temperature=0.7 + 0.001 * round_idx) if round_idx > 0 else lm
                new_settings = dict(lm=lm) if round_idx > 0 else {}

                with dsp.settings.context(**new_settings):
                    # Remove the example to bootstrap from the predictor's `demos` field, and cache the current demos.
                    for name, predictor in teacher.named_predictors():
                        predictor_cache[name] = predictor.demos
                        predictor.demos = [x for x in predictor.demos if x != example]

                    prediction = teacher(**example.inputs())
                    trace = dsp.settings.trace

                    # Restore the original demos from the cache.
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

        if not success:
            return False

        for step in trace:
            predictor, inputs, outputs = step
            demo = Example(augmented=True, **inputs, **outputs)

            try:
                predictor_name = self.predictor2name[id(predictor)]
            except KeyError:
                continue  # FIXME: !
            # Add the demo to the global storage, later it will be used to augment the student's demos.
            name2traces[predictor_name].append(demo)
        return True

    def _apply_bootstrap_results_on_student(self):
        """Assigns the bootstrapped demos to the student's predictors."""
        rng = random.Random(0)
        raw_demos = self.valset

        for name, predictor in self.student.named_predictors():
            augmented_demos = self.name2traces[name][: self.max_bootstrapped_demos]

            sample_size = min(self.max_labeled_demos - len(augmented_demos), len(raw_demos))
            sample_size = max(0, sample_size)

            raw_demos = rng.sample(raw_demos, sample_size)

            if dspy.settings.release >= 20230928:
                predictor.demos = raw_demos + augmented_demos
            else:
                predictor.demos = augmented_demos + raw_demos
        self.student._compiled = True
