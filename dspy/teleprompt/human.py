import random

from IPython.utils import data

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


class BootstrapFewShotWithHuman(Teleprompter):
    def __init__(
        self,
        metric,
        argilla_dataset,
        teacher_settings={},
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=6,
        max_errors=10,
        stop_at_score=None,
        metric_threshold=None,
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds
        self.argilla_dataset = argilla_dataset

        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

        print(
            f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor."
        )
        print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")

    def compile(
        self,
        student,
        *,
        teacher=None,
        trainset,
        valset=None,
        restrict=None,
        labeled_sample=True,
    ):
        self.trainset = trainset
        self.valset = valset or trainset  # TODO: FIXME: Note this choice.

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
                program = teleprompter.compile(
                    student, trainset=trainset_copy, sample=labeled_sample
                )

            elif seed == -1:
                # unshuffled few-shot
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                )
                program = optimizer.compile(
                    student, teacher=teacher, trainset=trainset_copy
                )
            else:
                trainset_copy = self._load_responses_from_argilla(
                    trainset=trainset_copy, student=student
                )
                assert seed >= 0, seed

                random.Random(seed).shuffle(trainset_copy)

                size = random.Random(seed).randint(
                    self.min_num_samples, self.max_num_samples
                )

                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                )

                program = optimizer.compile(
                    student, teacher=teacher, trainset=trainset_copy
                )

            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                display_table=False,
                display_progress=True,
            )

            score, subscores = evaluate(program, return_all_scores=True)

            all_subscores.append(subscores)

            ############ Assertion-aware Optimization ############
            if hasattr(program, "_suggest_failures"):
                score = score - program._suggest_failures * 0.2
            if hasattr(program, "_assert_failures"):
                score = 0 if program._assert_failures > 0 else score
            ######################################################

            if len(scores) == 0 or score > max(scores):
                print("New best score:", score, "for seed", seed)
                best_program = program

            scores.append(score)
            print(f"Scores so far: {scores}")
            print(f"Best score so far: {max(scores)}")

            score_data.append((score, subscores, seed, program))

            if self.stop_at_score is not None and score >= self.stop_at_score:
                print(
                    f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}"
                )
                break

        # To best program, attach all program candidates in decreasing average score
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(
            best_program.candidate_programs, key=lambda x: x[0], reverse=True
        )

        print(f"{len(best_program.candidate_programs)} candidate programs found.")

        return best_program

    def _load_responses_from_argilla(self, trainset, student):
        """Load responses from Argilla dataset into the trainset."""

        field_name = self.argilla_dataset.fields[0].name
        question_names = [question.name for question in self.argilla_dataset.questions]
        sample_names = set(trainset[0].keys())

        assert len(self.argilla_dataset.fields) == 1, "Only one field supported."
        assert set(question_names).issubset(
            sample_names
        ), "All questions must be present."
        assert field_name in sample_names, "Field name must be present."

        for sample in trainset:
            nearest_record = next(
                self.argilla_dataset.records(query=sample[field_name])
            )
            for question_name in question_names:
                human_response = nearest_record.responses[question_name][0].value
                sample[question_name] = human_response

        return trainset
