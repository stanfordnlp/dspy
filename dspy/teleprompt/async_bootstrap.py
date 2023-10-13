import dsp
import tqdm
import random

from dspy.primitives import Example
from dspy.teleprompt.bootstrap import BootstrapFewShot


class AsyncBootstrapFewShot(BootstrapFewShot):
    async def compile(self, student, *, teacher=None, trainset, valset=None):
        self.prepare_for_bootstrap(
            student, teacher=teacher, trainset=trainset, valset=valset
        )
        await self._bootstrap()

        self.student = self._train()
        self.student._compiled = True

        return self.student

    async def _bootstrap(self, *, max_bootsraps=None):
        max_bootsraps = max_bootsraps or self.max_bootstrapped_demos

        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2predictor}

        for round_idx in range(self.max_rounds):
            for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
                if len(bootstrapped) >= max_bootsraps:
                    break

                if example_idx not in bootstrapped:
                    success = await self._bootstrap_one_example(example, round_idx)

                    if success:
                        bootstrapped[example_idx] = True

        print(
            f"Bootstrapped {len(bootstrapped)} full traces after {example_idx+1} examples in round {round_idx}."
        )

        # Unbootstrapped training examples

        self.validation = [
            x for idx, x in enumerate(self.trainset) if idx not in bootstrapped
        ]
        random.Random(0).shuffle(self.validation)

        self.validation = self.valset or self.validation

        # NOTE: Can't yet use evaluate because we need to trace *per example*
        # evaluate = Evaluate(program=self.teacher, metric=self.metric, num_threads=12)
        # score = evaluate(self.metric, display_table=False, display_progress=True)

    async def _bootstrap_one_example(self, example, round_idx=0):
        name2traces = self.name2traces
        teacher = self.teacher  # .deepcopy()
        predictor_cache = {}

        try:
            with dsp.settings.context(trace=[], **self.teacher_settings):
                new_settings = self._make_new_settings(round_idx)

                with dsp.settings.context(**new_settings):
                    self._cache_and_update_predictor_demos(
                        teacher, example, predictor_cache
                    )

                    prediction = await teacher(**example.inputs())
                    trace = dsp.settings.trace
                    self._restore_predictor_demos_from_cache(teacher, predictor_cache)

                success = (self.metric is None) or self.metric(
                    example, prediction, trace
                )
                # print(success, example, prediction)
        except Exception as e:
            success = False
            # FIXME: remove the reliance on uuid here so the error is printed
            print(
                f"Failed to run or to evaluate example {example} with {self.metric} due to {e}."
            )

        if success:
            self._make_successful_demos(trace, example, name2traces)

        return success
