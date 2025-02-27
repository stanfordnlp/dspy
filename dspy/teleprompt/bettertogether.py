import logging
import random
from typing import Callable, List, Optional

import dspy
from dspy.primitives.example import Example
from dspy.primitives.program import Program
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune, prepare_student, set_missing_predictor_lms, launch_lms, kill_lms
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)


class BetterTogether(Teleprompter):

    STRAT_SEP = " -> "

    def __init__(self,
        metric: Callable,
        prompt_optimizer: Optional[Teleprompter] = None,
        weight_optimizer: Optional[Teleprompter] = None,
        seed: Optional[int] = None,
      ):
        if not dspy.settings.experimental:
            raise ValueError("This is an experimental optimizer. Set `dspy.settings.experimental` to `True` to use it.")

        # TODO: Note that the BetterTogether optimizer is meaningful when
        # BootstrapFinetune uses a metric to filter the training data before
        # fine-tuning. However, one can also choose to run this optimizer with 
        # a BoostrapFinetune without a metric, say, if there aren't labels
        # available for the training data. Should this be noted somewhere?
        # TODO: We should re-consider if the metric should be required.
        self.prompt_optimizer = prompt_optimizer if prompt_optimizer else BootstrapFewShotWithRandomSearch(metric=metric)
        self.weight_optimizer = weight_optimizer if weight_optimizer else BootstrapFinetune(metric=metric)

        is_supported_prompt = isinstance(self.prompt_optimizer, BootstrapFewShotWithRandomSearch)
        is_supported_weight = isinstance(self.weight_optimizer, BootstrapFinetune)
        if not is_supported_prompt or not is_supported_weight:
            raise ValueError(
                "The BetterTogether optimizer only supports the following optimizers for now: BootstrapFinetune, "
                "BootstrapFewShotWithRandomSearch."
            )

        self.rng = random.Random(seed)

    def compile(
        self,
        student: Program,
        trainset: List[Example],
        strategy: str = "p -> w -> p",
        valset_ratio = 0.1,
    ) -> Program:
        # TODO: We could record acc on a different valset to pick the best
        # strategy within the provided strategy
        logger.info("Validating the strategy")
        parsed_strategy = strategy.lower().split(self.STRAT_SEP)

        if not all([s in ["p", "w"] for s in parsed_strategy]):
            raise ValueError(
                f"The strategy should be a sequence of 'p' and 'w' separated by '{self.STRAT_SEP}', but "
                f"found: {strategy}"
            )

        logger.info("Preparing the student program...")
        # TODO: Prepare student returns student.reset_copy(), which is what gets
        # optimized. We should make this clear in the doc comments.
        student = prepare_student(student)
        set_missing_predictor_lms(student)

        # Make a shallow copy of the trainset, so that we don't change the order
        # of the examples in the original trainset
        trainset = trainset[:]
        logger.info("Compiling the student program...")
        student = self._run_strategies(parsed_strategy, student, trainset, valset_ratio)
        
        logger.info("BetterTogether has finished compiling the student program")
        return student
  
    def _run_strategies(self, parsed_strategy, student, trainset, valset_ratio) -> Program:
        # Keep track of all the partial strategies/programs in parsed_strategy
        # "" corresponds to the initial student program
        candidate_programs = []
        candidate_programs.append(("", student))
        launched_flag = False

        for ind, step_code in enumerate(parsed_strategy):
            current_strategy = self.STRAT_SEP.join(parsed_strategy[:ind + 1])
            logger.info(
                f"\n########## Step {ind + 1} of {len(parsed_strategy)} - Strategy "
                f"'{current_strategy}' ##########"
            )

            logger.info("Shuffling the trainset...")
            self.rng.shuffle(trainset)
            if not launched_flag:
                launch_lms(student)
                launched_flag = True

            # TODO: Should we reset or just deepcopy? How does resetting affect
            # the predictor LMs?
            student = student.deepcopy()
            student._compiled = False
            if step_code == "p":
                student = self._compile_prompt_optimizer(student, trainset, valset_ratio)
            elif step_code == "w":
                student = self._compile_weight_optimizer(student, trainset)
                launched_flag = False

            # Record the program corresponding to the current strategy
            candidate_programs.append((current_strategy, student))

        if launched_flag:
            kill_lms(student)

        student.candidate_programs = candidate_programs
        return student
  
    def _compile_prompt_optimizer(self, student, trainset, valset_ratio) -> Program:
        logger.info("Preparing for prompt optimization...")

        # Sampling a validation set from the trainset for the prompt optimizer
        # We drop the hints for prompt optimization
        trainset = [x.with_inputs(*list(set(x.inputs().keys()) - {"hint"})) for x in trainset]
        num_val = int(valset_ratio * len(trainset))
        prompt_valset = trainset[:num_val]
        prompt_trainset = trainset[num_val:]

        # TODO: To make this optimizer general, we need to ensure that all the
        # prompt optimizers are accepting a valset or encode a way to check if
        # a valset should be passed to an optimizer's compile method.
        # TODO: We should ensure that the prompt optimizers in DSPy respect the
        # predictor.lm attributes. In particular,
        # BootstrapFewShotWithRandomSearch seems to be resetting these. We are
        # manually re-setting the LMs here to circumvent this issue, but we
        # should consider adressing it in BFRS.
        logger.info("Compiling the prompt optimizer...")
        pred_lms = [pred.lm for pred in student.predictors()]
        student = self.prompt_optimizer.compile(student, trainset=prompt_trainset, valset=prompt_valset)
        for pred, lm in zip(student.predictors(), pred_lms):
            pred.lm = lm

        return student
    
    def _compile_weight_optimizer(self, student, trainset) -> Program:
        logger.info("Preparing for weight optimization...")

        # Saving the LMs before compiling the weight optimizer
        original_lms = [pred.lm for pred in student.predictors()]

        # TODO: To make this optimizer general, we need to ensure that all the
        # prompt optimizers are accepting a valset or encode a way to check if
        # a valset should be passed to an optimizer's compile.
        logger.info("Compiling the weight optimizer...")
        student = self.weight_optimizer.compile(student, trainset=trainset)     

        # Updating the train kwargs for the new LMs. This is needed because the
        # train_kwargs of the optimizer is configured for the original LMs.
        new_lms = [pred.lm for pred in student.predictors()]
        for original_lm, new_lm in zip(original_lms, new_lms):
            original_params = self.weight_optimizer.train_kwargs[original_lm]
            self.weight_optimizer.train_kwargs[new_lm] = original_params

        return student
