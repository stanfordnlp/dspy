import logging
import random
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional

import optuna

import dspy
from dspy.predict.knn import load_knn_embeddings
from dspy.teleprompt.bootstrap import BootstrapKNN
from dspy.teleprompt.utils import get_signature, set_signature

from .mipro_optimizer_v2 import MIPROv2, BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT, LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT

logger = logging.getLogger(__name__)

DemoCandidate = namedtuple("DemoCandidate", ["demos", "random_seed", "retrieve_demos", "augmented_demos"])


class MIPROv2KNN(MIPROv2):
    def __init__(
        self,
        metric: Callable,
        embedder: "dspy.Embedder",
        prompt_model: Optional[Any] = None,
        task_model: Optional[Any] = None,
        teacher_settings: Dict = {},
        max_bootstrapped_demos: int = 16,
        max_labeled_demos: int = 4,
        auto: Optional[str] = None,
        num_candidates: int = 10,
        num_threads: int = 6,
        max_errors: int = 10,
        seed: int = 9,
        init_temperature: float = 0.5,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: Optional[str] = None,
        metric_threshold: Optional[float] = None,
        logger: logging.Logger = logger,
    ):
        super().__init__(
            metric=metric,
            prompt_model=prompt_model,
            task_model=task_model,
            teacher_settings=teacher_settings,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_threads=num_threads,
            max_errors=max_errors,
            seed=seed,
            init_temperature=init_temperature,
            verbose=verbose,
            track_stats=track_stats,
            log_dir=log_dir,
            metric_threshold=metric_threshold,
            num_candidates=num_candidates,
            auto=auto,
            logger=logger,
        )
        self.embedder = embedder

    def _bootstrap_fewshot_examples(
        self, program: Any, trainset: List, seed: int, teacher: Any
    ) -> dict[int, DemoCandidate]:
        self.logger.info("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            self.logger.info(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            self.logger.info("These will be used for informing instruction proposal.\n")

        self.logger.info(f"Bootstrapping N={self.num_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0

        try:
            demo_candidates = bootstrap_knn_demos(
                student=program,
                embedder=self.embedder,
                num_candidate_sets=self.num_candidates,
                trainset=trainset,
                max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_labeled_demos),
                max_bootstrapped_demos=(
                    BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_bootstrapped_demos
                ),
                metric=self.metric,
                max_errors=self.max_errors,
                teacher=teacher,
                teacher_settings=self.teacher_settings,
                seed=seed,
                metric_threshold=self.metric_threshold,
                rng=self.rng,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.error(f"Error generating few-shot examples: {e}\nRunning without few-shot examples.")
            demo_candidates = None

        return demo_candidates

    def _select_and_insert_instructions_and_demos(
        self,
        candidate_program: Any,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: dict[int, list[DemoCandidate]],
        trial: optuna.trial.Trial,
        trial_logs: Dict,
        trial_num: int,
    ) -> List[str]:
        chosen_params = []

        for i, predictor in enumerate(candidate_program.predictors()):
            # Select instruction
            instruction_idx = trial.suggest_categorical(
                f"{i}_predictor_instruction", range(len(instruction_candidates[i]))
            )
            selected_instruction = instruction_candidates[i][instruction_idx]
            updated_signature = get_signature(predictor).with_instructions(selected_instruction)
            set_signature(predictor, updated_signature)
            trial_logs[trial_num][f"{i}_predictor_instruction"] = instruction_idx
            chosen_params.append(f"Predictor {i}: Instruction {instruction_idx}")

            # Select demos if available
            if demo_candidates:
                demos_idx = trial.suggest_categorical(f"{i}_predictor_demos", range(len(demo_candidates[i])))
                candidate = demo_candidates[i][demos_idx]

                predictor.demos = candidate.demos
                predictor.random_seed = candidate.random_seed
                predictor.retrieve_demos = candidate.retrieve_demos

                trial_logs[trial_num][f"{i}_predictor_demos"] = demos_idx
                chosen_params.append(f"Predictor {i}: Few-Shot Set {demos_idx}")

        self.logger.info("Generating KNN embeddings as required...")
        load_knn_embeddings(candidate_program, self.num_threads)
        return chosen_params

    def _propose_instructions(
        self,
        program: Any,
        trainset: List,
        demo_candidates: Optional[dict[int, list[DemoCandidate]]],
        view_data_batch_size: int,
        program_aware_proposer: bool,
        data_aware_proposer: bool,
        tip_aware_proposer: bool,
        fewshot_aware_proposer: bool,
    ):
        if demo_candidates:
            demo_candidates = {i: [demos for _, _, _, demos in candidates] for i, candidates in demo_candidates.items()}

        return super()._propose_instructions(
            program,
            trainset,
            demo_candidates,
            view_data_batch_size,
            program_aware_proposer,
            data_aware_proposer,
            tip_aware_proposer,
            fewshot_aware_proposer,
        )


def bootstrap_knn_demos(
    student,
    embedder: "dspy.Embedder",
    num_candidate_sets,
    trainset,
    max_labeled_demos,
    max_bootstrapped_demos,
    metric,
    teacher_settings,
    max_errors=10,
    max_rounds=1,
    metric_threshold=None,
    teacher=None,
    seed=0,
    num_threads=6,
    rng=None,
    logger=logger,
) -> dict[int, list[DemoCandidate]]:
    rng = rng or random.Random(seed)

    demo_candidates = defaultdict[int, list[DemoCandidate]](list)

    # Go through and create each candidate set
    for seed in range(num_candidate_sets):
        logger.info(f"Bootstrapping set {seed+1}/{num_candidate_sets}")

        trainset_copy = list(trainset)
        rng.shuffle(trainset_copy)

        num_static_demos = 0 if seed == 0 else rng.randint(1, max_labeled_demos - 1)
        optimizer = BootstrapKNN(
            metric=metric,
            embedder=embedder,
            max_errors=max_errors,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            teacher_settings=teacher_settings,
            max_rounds=max_rounds,
            metric_threshold=metric_threshold,
            num_threads=num_threads,
            random_seed=seed,
            num_static_demos=num_static_demos,
        )

        optimized_student = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

        for i, predictor in enumerate(optimized_student.predictors()):
            demo_candidates[i].append(
                DemoCandidate(
                    demos=predictor.demos,
                    random_seed=predictor.random_seed,
                    retrieve_demos=predictor.retrieve_demos,
                    augmented_demos=rng.sample(predictor.augmented_demos, k=max_labeled_demos),
                )
            )

    return demo_candidates
