import random

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter

from .bootstrap import BootstrapFewShot
from .vanilla import LabeledFewShot

# TODO: Don't forget dealing with the raw demos.
# TODO: Deal with the (pretty common) case of having a metric for filtering and a separate metric for eval.
# The metric itself may tell though by the presence of trace.

# TODO: This function should take a max_budget and max_teacher_budget. That's in the number of program calls.
# In this case, max_student_budget is max_budget - max_teacher_budget.
# For max_teacher_budget, this will just limit the total number of things we bootstrap.
# This can end up implicitly defining the number of candidate programs (i.e., stop when runs out). Cap at 16.
# For max_student_budget, this will be a more upfront calculation.
# Right now, it can also just induce the number of candidate programs. Later, it could be used more interestingly
# for selective early stopping.
# Progressive elimination sounds about right: after 50 examples, drop bottom third, after 100, another third, etc.
# until only 3--5 are left for the end. Could also be systematic and add (earlier) stopping based on error bounds.
# In general, though, the early filtering is just saying: either there are some really bad ones, or some really really
# good ones, or most things are pretty close. In all of these cases, dropping the bottom third is not going to hurt.


class BootstrapFewShotWithRandomSearch(Teleprompter):
    """Randomly search over bootstrapped few-shot demonstration sets.

    Generates multiple candidate programs by running ``BootstrapFewShot`` with
    different random seeds and shuffled training sets, then evaluates each
    candidate on a validation set and returns the best one.  Three special
    seeds are always tried first: a zero-shot baseline (seed -3), a
    labeled-only baseline (seed -2), and an unshuffled bootstrap (seed -1).

    The returned program carries a ``candidate_programs`` attribute with all
    evaluated candidates sorted by descending score, so callers can inspect
    the full search history.
    """

    def __init__(
        self,
        metric,
        teacher_settings=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=None,
        max_errors=None,
        stop_at_score=None,
        metric_threshold=None,
    ):
        """Initialize the random-search optimizer.

        Args:
            metric: Evaluation function used to score each candidate program.
            teacher_settings: Optional dict of settings forwarded to the
                teacher LM during bootstrapping.
            max_bootstrapped_demos: Upper bound on bootstrapped demonstrations
                per predictor.  Each candidate randomly samples between 1 and
                this value.
            max_labeled_demos: Maximum number of labeled demonstrations
                included alongside bootstrapped ones.
            max_rounds: Number of bootstrapping rounds passed to each
                ``BootstrapFewShot`` run.
            num_candidate_programs: How many random-seed candidates to
                generate (in addition to the three special seeds).
            num_threads: Thread count for the ``Evaluate`` calls.
            max_errors: Maximum evaluation errors tolerated before aborting.
                Falls back to ``dspy.settings.max_errors`` when *None*.
            stop_at_score: If set, stop the search early once a candidate
                reaches this score.
            metric_threshold: Passed to ``BootstrapFewShot`` to filter
                bootstrapped demos by quality.
        """
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds

        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

        print(f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor.")
        print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")

    def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
        """Search for the best demonstration set and return the top program.

        Iterates over the special seeds (-3, -2, -1) and then
        ``num_candidate_programs`` random seeds.  For each seed the training
        set is shuffled, a ``BootstrapFewShot`` optimizer produces a candidate
        program, and that candidate is scored on *valset*.  The program with
        the highest score is returned.

        Args:
            student: The ``dspy.Module`` to optimize.
            teacher: Optional teacher module for bootstrapping.  When *None*,
                the student acts as its own teacher.
            trainset: Training examples used to bootstrap demonstrations.
            valset: Validation examples for scoring candidates.  Defaults to
                *trainset* when not provided.
            restrict: Optional collection of seed values.  When given, only
                seeds present in *restrict* are evaluated.
            labeled_sample: Whether to sample labeled demos (forwarded to
                ``LabeledFewShot.compile``).

        Returns:
            The best-scoring ``dspy.Module`` copy, with a
            ``candidate_programs`` list attached (sorted by descending score).
        """
        self.trainset = trainset
        self.valset = valset or trainset  # TODO: FIXME: Note this choice.

        effective_max_errors = self.max_errors if self.max_errors is not None else dspy.settings.max_errors

        scores = []
        all_subscores = []
        score_data = []

        for seed in range(-3, self.num_candidate_sets):
            if (restrict is not None) and (seed not in restrict):
                continue

            trainset_copy = list(self.trainset)

            if seed == -3:
                # zero-shot
                program = student.reset_copy()

            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

            elif seed == -1:
                # unshuffled few-shot
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )
                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            else:
                assert seed >= 0, seed

                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )

                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=effective_max_errors,
                display_table=False,
                display_progress=True,
            )

            result = evaluate(program)

            score, subscores = result.score, [output[2] for output in result.results]

            all_subscores.append(subscores)

            if len(scores) == 0 or score > max(scores):
                print("New best score:", score, "for seed", seed)
                best_program = program

            scores.append(score)
            print(f"Scores so far: {scores}")
            print(f"Best score so far: {max(scores)}")

            score_data.append({"score": score, "subscores": subscores, "seed": seed, "program": program})

            if self.stop_at_score is not None and score >= self.stop_at_score:
                print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # To best program, attach all program candidates in decreasing average score
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(
            best_program.candidate_programs, key=lambda x: x["score"], reverse=True
        )

        print(f"{len(best_program.candidate_programs)} candidate programs found.")

        return best_program


# sample between 4 and 10 examples from traces
# TODO: FIXME: The max number of demos should be determined in part by the LM's tokenizer + max_length.
# This does require executing the program, or at least the predictor.
# # # # # # (Actually we can just combine the token counts of the traces, when formatted via signature/adapter).
# Alternatively, we can keep track of the (zero-shot) number of tokens when we bootstrap.
# As another option, we can just try a wide range and handle failures as penalties on the score.
# The number "24" of traces to collect can also be affected. If we only need 3x10, some overlap is ok.
# We can also consider having short_demos and long_demos.
