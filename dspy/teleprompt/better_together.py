import random
from typing import Any, Callable, Dict, List, Optional

import dspy
from dspy.adapters.base import Adapter
from dspy.clients.lm import LM, TrainingJob
from dspy.clients.utils_finetune import infer_data_format
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.predict.predict import Predict
from dspy.primitives.program import Program
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.logging import logger


from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch


class BetterTogether(Teleprompter):

    def __init__(self,
        metric: Callable,
        prompt_optimizer: Optional[Teleprompter] = None,
        weight_optimizer: Optional[Teleprompter] = None,
        random_seed: Optional[int] = None,
      ):
        """TODO: Docstring"""
        err = "This is an experimental optimizer."
        err += " Set `dspy.settings.experimental` to `True` to use it."
        assert dspy.settings.experimental, err

        self.prompt_optimizer = prompt_optimizer if prompt_optimizer else BootstrapFewShotWithRandomSearch(metric=metric)
        self.weight_optimizer = weight_optimizer if weight_optimizer else BootstrapFinetune()
        self.rng = random.Random(random_seed)

    def compile(
        self,
        student: Program,
        trainset: List[Example],
        strategy: str = "p -> w -> p",
        valset_ratio = 0.1,
    ) -> Program:
        # TODO: Validate strategy
        # TODO: Should we take in strategy? Should we search using the valset?
        # TODO: We could record valset errors for the intermediate strategies
        student = BootstrapFinetune.prepare_student(student)

        parsed_strategies = strategy.split(" -> ")
        for p in parsed_strategies:
            self.rng.shuffle(trainset)
            valset = trainset[:int(valset_ratio * len(trainset))]
            if p == "p":
                self.launch_lms(student)
                student = self.prompt_optimizer.compile(student, trainset=trainset, valset=valset)
                self.kill_lms(student)
            elif p == "w":
                student = self.weight_optimizer.compile(student, trainset=trainset, valset=valset)
        return student

    @staticmethod
    def get_unique_lms(program: Program) -> List[LM]:
        lms = [pred.lm for pred in program.predictors()]
        lms = list(set(lms))  # TODO: Assuming LM is hashable
        return lms

    @staticmethod
    def launch_lms(program: Program):
        lms = BetterTogether.get_unique_lms(program)
        for lm in lms:
            lm.launch()
  
    @staticmethod
    def kill_lms(program: Program):
        lms = BetterTogether.get_unique_lms(program)
        for lm in lms:
            lm.kill()
