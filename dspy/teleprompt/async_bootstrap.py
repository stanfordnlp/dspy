import dsp
import tqdm
import random

from dspy.primitives import Example
from dspy.teleprompt.bootstrap import BootstrapFewShotBase


class AsyncBootstrapFewShot(BootstrapFewShotBase):
    async def compile(self, student, *, teacher=None, trainset, valset=None):
        self.trainset = trainset
        self.valset = valset

        self._prepare_student_and_teacher(student, teacher)
        self._prepare_predictor_mappings()
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
                lm = dsp.settings.lm
                lm = (
                    lm.copy(temperature=0.7 + 0.001 * round_idx)
                    if round_idx > 0
                    else lm
                )
                new_settings = dict(lm=lm) if round_idx > 0 else {}

                with dsp.settings.context(**new_settings):
                    for name, predictor in teacher.named_predictors():
                        predictor_cache[name] = predictor.demos
                        predictor.demos = [x for x in predictor.demos if x != example]

                    prediction = await teacher(**example.inputs())
                    trace = dsp.settings.trace

                    for name, predictor in teacher.named_predictors():
                        predictor.demos = predictor_cache[name]

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
            for step in trace:
                predictor, inputs, outputs = step

                if "dspy_uuid" in example:
                    demo = Example(
                        augmented=True, dspy_uuid=example.dspy_uuid, **inputs, **outputs
                    )
                else:
                    # TODO: FIXME: This is a hack. RandomSearch will complain for now in this edge case.
                    demo = Example(augmented=True, **inputs, **outputs)

                try:
                    predictor_name = self.predictor2name[id(predictor)]
                except KeyError as e:
                    continue  # FIXME: !

                    # TODO: Look closer into this. It's a bit tricky to reproduce.
                    print(
                        f"Failed to find predictor {predictor} in {self.predictor2name}."
                    )
                    print(
                        "Are you doing this in a notebook (Jupyter)? This might be caused by redefining values by rerunning cells."
                    )
                    print("Try restarting the notebook, or open an issue.")
                    raise KeyError(
                        f"Failed to find predictor {id(predictor)} {predictor} in {self.predictor2name}."
                    ) from e

                name2traces[predictor_name].append(demo)

        return success
