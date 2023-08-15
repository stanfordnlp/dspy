import dsp
import tqdm
import random

from .bootstrap import BootstrapFewShot
from .vanilla import LabeledFewShot

from dspy.evaluate.evaluate import Evaluate


# TODO: Don't forget dealing with the raw demos.
# TODO: Deal with the (pretty common) case of having a metric for filtering and a metric for eval

# TODO: There's an extremely strong case (> 90%) to switch this to NOT inherit from BootstrapFewShot.
# Instead, it should wrap it: during compilation just loop, create a copy to compile, shuffle the full/sampled
# trainset and compile with that. This will also make it easier to use raw demos.
# Once all versions exist, define the validation set and evaluate.

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

# Also, this function should be efficient in the following way:
# seed -3, seed -2, and seed -1 so to speak should just be "zero shot", "labeled shots", and "bootstrap" without any tweaks.


class BootstrapFewShotWithRandomSearch(BootstrapFewShot):
    def __init__(self, metric, teacher_settings={}, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=6):
        self.metric = metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds

        self.num_threads = num_threads

        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        # Semi-hacky way to get the parent class's _boostrap function to stop early.
        self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        print("Going to sample between", self.min_num_samples, "and", self.max_num_samples, "traces per predictor.")
        print("Going to sample", self.max_num_traces, "traces in total.")
        print("Will attempt to train", self.num_candidate_sets, "candidate sets.")

        # self.num_candidate_sets = 1

        # import time
        # time.sleep(10)

    def _random_search_instance(self, idx):
        print("Random search instance", idx)

        rng = random.Random(idx)
        program = self.student.deepcopy()

        for predictor in program.predictors():
            sample_size = rng.randint(self.min_num_samples, self.max_num_samples)
            print(f"[{idx}] \t Sampling {sample_size} traces from {len(predictor.traces)} traces.")

            augmented_demos = rng.sample(predictor.traces, min(sample_size, len(predictor.traces)))

            # TODO: FIXME: Figuring out the raw demos here is a bit tricky. We can't just use the unused nor the unaugmented (validation) ones.
            augmented_uuids = set([x.dspy_uuid for x in augmented_demos])
            raw_demos_uuids = set([x.dspy_uuid for x in predictor.traces if x.dspy_uuid not in augmented_uuids])

            raw_demos = [x for x in self.trainset if x.dspy_uuid in raw_demos_uuids]
            raw_demos = rng.sample(raw_demos, min(self.max_labeled_demos - len(augmented_demos), len(raw_demos)))

            print(f'Got {len(augmented_demos)} augmented demos and {len(raw_demos)} raw demos.')
            predictor.demos = augmented_demos + raw_demos
        
        evaluate = Evaluate(devset=self.validation, metric=self.metric, num_threads=self.num_threads, display_table=False, display_progress=True)
        score = evaluate(program)

        print('Score:', score, 'for set:', [len(predictor.demos) for predictor in program.predictors()])
        # dsp.settings.lm.inspect_history(n=1)

        return (score, program)

    def _train(self):
        for name, predictor in self.student.named_predictors():
            predictor.traces = self.name2traces[name]
            pass

        self.candidate_sets = []

        for candidate_set_idx in range(self.num_candidate_sets):
            score, program = self._random_search_instance(candidate_set_idx)
            self.candidate_sets.append((score, program))
        
        best_score, best_program = max(self.candidate_sets, key=lambda x: x[0])
        print('Best score:', best_score)

        return best_program




# sample between 4 and 10 examples from traces
# TODO: FIXME: The max number of demos should be determined in part by the LM's tokenizer + max_length.
# This does require excecuting the program, or at least the predictor.
# # # # # # (Actually we can just combine the token counts of the traces, when formatted via signature/adapter).
# Alternatively, we can keep track of the (zero-shot) number of tokens when we bootstrap.
# As another option, we can just try a wide range and handle failures as penalties on the score.
# The number "24" of traces to collect can also be affected. If we only need 3x10, some overlap is ok.
# We can also consider having short_demos and long_demos.
