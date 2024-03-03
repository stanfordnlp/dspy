import random

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
    def __init__(self, metric, teacher_settings={}, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=6, max_errors=10, stop_at_score=None):
        self.metric = metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds

        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        # self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        # Semi-hacky way to get the parent class's _bootstrap function to stop early.
        # self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        print("Going to sample between", self.min_num_samples, "and", self.max_num_samples, "traces per predictor.")
        # print("Going to sample", self.max_num_traces, "traces in total.")
        print("Will attempt to train", self.num_candidate_sets, "candidate sets.")

    def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
        self.trainset = trainset
        self.valset = valset or trainset  # TODO: FIXME: Note this choice.

        scores = []
        all_subscores = []
        score_data = []

        for seed in range(-3, self.num_candidate_sets):
            if (restrict is not None) and (seed not in restrict):
                print(seed, restrict)
                continue

            trainset2 = list(self.trainset)

            if seed == -3:
                # zero-shot
                program2 = student.reset_copy()
            
            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program2 = teleprompter.compile(student, trainset=trainset2, sample=labeled_sample)
            
            elif seed == -1:
                # unshuffled few-shot
                program = BootstrapFewShot(metric=self.metric, max_bootstrapped_demos=self.max_num_samples,
                                           max_labeled_demos=self.max_labeled_demos,
                                           teacher_settings=self.teacher_settings, max_rounds=self.max_rounds)
                program2 = program.compile(student, teacher=teacher, trainset=trainset2)

            else:
                assert seed >= 0, seed

                random.Random(seed).shuffle(trainset2)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

                teleprompter = BootstrapFewShot(metric=self.metric, max_bootstrapped_demos=size,
                                                max_labeled_demos=self.max_labeled_demos,
                                                teacher_settings=self.teacher_settings,
                                                max_rounds=self.max_rounds)

                program2 = teleprompter.compile(student, teacher=teacher, trainset=trainset2)

            evaluate = Evaluate(devset=self.valset, metric=self.metric, num_threads=self.num_threads,
                                max_errors=self.max_errors, display_table=False, display_progress=True)

            score, subscores = evaluate(program2, return_all_scores=True)

            all_subscores.append(subscores)

            ############ Assertion-aware Optimization ############
            if hasattr(program2, '_suggest_failures'):
                score = score - program2._suggest_failures * 0.2
            if hasattr(program2, '_assert_failures'):
                score = 0 if program2._assert_failures > 0 else score
            ######################################################

            print('Score:', score, 'for set:', [len(predictor.demos) for predictor in program2.predictors()])

            if len(scores) == 0 or score > max(scores):
                print('New best score:', score, 'for seed', seed)
                best_program = program2

            scores.append(score)
            print(f"Scores so far: {scores}")

            print('Best score:', max(scores))

            score_data.append((score, subscores, seed, program2))

            if len(score_data) > 2:  # We check if there are at least 3 scores to consider
                for k in [1, 2, 3, 5, 8, 9999]:
                    top_3_scores = sorted(score_data, key=lambda x: x[0], reverse=True)[:k]

                    # Transpose the subscores to get max per entry and then calculate their average
                    transposed_subscores = zip(*[subscores for _, subscores, *_ in top_3_scores if subscores])
                    avg_of_max_per_entry = sum(max(entry) for entry in transposed_subscores) / len(top_3_scores[0][1])

                    print(f'Average of max per entry across top {k} scores: {avg_of_max_per_entry}')
            
            if self.stop_at_score is not None and score >= self.stop_at_score:
                print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # To best program, attach all program candidates in decreasing average score
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(best_program.candidate_programs, key=lambda x: x[0], reverse=True)

        print(len(best_program.candidate_programs), "candidate programs found.")

        return best_program




# sample between 4 and 10 examples from traces
# TODO: FIXME: The max number of demos should be determined in part by the LM's tokenizer + max_length.
# This does require executing the program, or at least the predictor.
# # # # # # (Actually we can just combine the token counts of the traces, when formatted via signature/adapter).
# Alternatively, we can keep track of the (zero-shot) number of tokens when we bootstrap.
# As another option, we can just try a wide range and handle failures as penalties on the score.
# The number "24" of traces to collect can also be affected. If we only need 3x10, some overlap is ok.
# We can also consider having short_demos and long_demos.
