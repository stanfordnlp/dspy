from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter

from .bootstrap import BootstrapFewShot


class BootstrapFewShotWithOptuna(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_settings=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=None,
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds
        self.num_threads = num_threads
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        # self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        # Semi-hacky way to get the parent class's _bootstrap function to stop early.
        # self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        print("Going to sample between", self.min_num_samples, "and", self.max_num_samples, "traces per predictor.")
        # print("Going to sample", self.max_num_traces, "traces in total.")
        print("Will attempt to train", self.num_candidate_sets, "candidate sets.")

    def objective(self, trial):
        program2 = self.student.reset_copy()
        for (name, compiled_predictor), (_, program2_predictor) in zip(
            self.compiled_teleprompter.named_predictors(), program2.named_predictors(), strict=False,
        ):
            all_demos = compiled_predictor.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            selected_demo = dict(all_demos[demo_index])
            program2_predictor.demos = [selected_demo]
        evaluate = Evaluate(
            devset=self.valset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_table=False,
            display_progress=True,
        )
        result = evaluate(program2)
        trial.set_user_attr("program", program2)
        return result.score

    def compile(self, student, *, teacher=None, max_demos, trainset, valset=None):
        import optuna
        self.trainset = trainset
        self.valset = valset or trainset
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.reset_copy()
        teleprompter_optimize = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=self.max_labeled_demos,
            teacher_settings=self.teacher_settings,
            max_rounds=self.max_rounds,
        )
        self.compiled_teleprompter = teleprompter_optimize.compile(
            self.student, teacher=self.teacher, trainset=self.trainset,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.num_candidate_sets)
        best_program = study.trials[study.best_trial.number].user_attrs["program"]
        print("Best score:", study.best_value)
        print("Best program:", best_program)
        return best_program
