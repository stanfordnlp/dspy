import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.propose import GroundedProposer
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import (
    create_minibatch,
    create_n_fewshot_demo_sets,
    eval_candidate_program,
    get_program_with_highest_avg_score,
    get_signature,
    print_full_program,
    save_candidate_program,
    set_signature,
)

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


class MIPROv2(Teleprompter):
    def __init__(
        self,
        metric: Callable,
        prompt_model: Any | None = None,
        task_model: Any | None = None,
        teacher_settings: dict | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: int | None = None,
        num_threads: int | None = None,
        max_errors: int | None = None,
        seed: int = 9,
        init_temperature: float = 1.0,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: str | None = None,
        metric_threshold: float | None = None,
    ):
        # Validate 'auto' parameter
        allowed_modes = {None, "light", "medium", "heavy"}
        if auto not in allowed_modes:
            raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")
        self.auto = auto
        self.num_fewshot_candidates = num_candidates
        self.num_instruct_candidates = num_candidates
        self.num_candidates = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.task_model = task_model if task_model else dspy.settings.lm
        self.prompt_model = prompt_model if prompt_model else dspy.settings.lm
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.teacher_settings = teacher_settings or {}
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.seed = seed
        self.rng = None

        if not self.prompt_model or not self.task_model:
            raise ValueError("Either provide both prompt_model and task_model or set a default LM through dspy.configure(lm=...)")

    def compile(
        self,
        student: Any,
        *,
        trainset: list,
        teacher: Any = None,
        valset: list | None = None,
        num_trials: int | None = None,
        max_bootstrapped_demos: int | None = None,
        max_labeled_demos: int | None = None,
        seed: int | None = None,
        minibatch: bool = True,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool | None = None, # deprecated
        provide_traceback: bool | None = None,
    ) -> Any:
        if requires_permission_to_run == False:
            logger.warning(
                "'requires_permission_to_run' is deprecated and will be removed in a future version."
            )
        elif requires_permission_to_run == True:
            raise ValueError("User confirmation is removed from MIPROv2. Please remove the 'requires_permission_to_run' argument.")

        effective_max_errors = (
            self.max_errors
            if self.max_errors is not None
            else dspy.settings.max_errors
        )
        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (self.max_labeled_demos == 0)

        # If auto is None, and num_trials is not provided (but num_candidates is), raise an error that suggests a good num_trials value
        if self.auto is None and (self.num_candidates is not None and num_trials is None):
            raise ValueError(
                f"If auto is None, num_trials must also be provided. Given num_candidates={self.num_candidates}, we'd recommend setting num_trials to ~{self._set_num_trials_from_num_candidates(student, zeroshot_opt, self.num_candidates)}."
            )

        # If auto is None, and num_candidates or num_trials is None, raise an error
        if self.auto is None and (self.num_candidates is None or num_trials is None):
            raise ValueError("If auto is None, num_candidates must also be provided.")

        # If auto is provided, and either num_candidates or num_trials is not None, raise an error
        if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
            raise ValueError(
                "If auto is not None, num_candidates and num_trials cannot be set, since they would be overridden by the auto settings. Please either set auto to None, or do not specify num_candidates and num_trials."
            )

        # Set random seeds
        seed = seed or self.seed
        self._set_random_seeds(seed)

        # Update max demos if specified
        if max_bootstrapped_demos is not None:
            self.max_bootstrapped_demos = max_bootstrapped_demos
        if max_labeled_demos is not None:
            self.max_labeled_demos = max_labeled_demos

        # Set training & validation sets
        trainset, valset = self._set_and_validate_datasets(trainset, valset)

        # Set hyperparameters based on run mode (if set)
        num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
            student, num_trials, minibatch, zeroshot_opt, valset
        )

        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")

        # Initialize program and evaluator
        program = student.deepcopy()
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
            provide_traceback=provide_traceback,
        )

        with dspy.context(lm=self.task_model):
            # Step 1: Bootstrap few-shot examples
            demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher)

        # Step 2: Propose instruction candidates
        instruction_candidates = self._propose_instructions(
            program,
            trainset,
            demo_candidates,
            view_data_batch_size,
            program_aware_proposer,
            data_aware_proposer,
            tip_aware_proposer,
            fewshot_aware_proposer,
        )

        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None

        with dspy.context(lm=self.task_model):
            # Step 3: Find optimal prompt parameters
            best_program = self._optimize_prompt_parameters(
                program,
                instruction_candidates,
                demo_candidates,
                evaluate,
                valset,
                num_trials,
                minibatch,
                minibatch_size,
                minibatch_full_eval_steps,
                seed,
            )

        return best_program

    def _set_random_seeds(self, seed):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _set_num_trials_from_num_candidates(self, program, zeroshot_opt, num_candidates):
        num_vars = len(program.predictors())
        if not zeroshot_opt:
            num_vars *= 2  # Account for few-shot examples + instruction variables
        # Trials = MAX(c*M*log(N), c=2, 3/2*N)
        num_trials = int(max(2 * num_vars * np.log2(num_candidates), 1.5 * num_candidates))

        return num_trials

    def _set_hyperparams_from_run_mode(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        zeroshot_opt: bool,
        valset: list,
    ) -> tuple[int, list, bool]:
        if self.auto is None:
            return num_trials, valset, minibatch

        auto_settings = AUTO_RUN_SETTINGS[self.auto]

        valset = create_minibatch(valset, batch_size=auto_settings["val_size"], rng=self.rng)
        minibatch = len(valset) > MIN_MINIBATCH_SIZE

        # Set num instruct candidates to 1/2 of N if optimizing with few-shot examples, otherwise set to N
        # This is because we've found that it's generally better to spend optimization budget on few-shot examples
        # When they are allowed.
        self.num_instruct_candidates = auto_settings["n"] if zeroshot_opt else int(auto_settings["n"] * 0.5)
        self.num_fewshot_candidates = auto_settings["n"]

        num_trials = self._set_num_trials_from_num_candidates(program, zeroshot_opt, auto_settings["n"])

        return num_trials, valset, minibatch

    def _set_and_validate_datasets(self, trainset: list, valset: list | None):
        if not trainset:
            raise ValueError("Trainset cannot be empty.")

        if valset is None:
            if len(trainset) < 2:
                raise ValueError("Trainset must have at least 2 examples if no valset specified.")
            valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]
        else:
            if len(valset) < 1:
                raise ValueError("Validation set must have at least 1 example.")

        return trainset, valset

    def _print_auto_run_settings(self, num_trials: int, minibatch: bool, valset: list):
        logger.info(
            f"\nRUNNING WITH THE FOLLOWING {self.auto.upper()} AUTO RUN SETTINGS:"
            f"\nnum_trials: {num_trials}"
            f"\nminibatch: {minibatch}"
            f"\nnum_fewshot_candidates: {self.num_fewshot_candidates}"
            f"\nnum_instruct_candidates: {self.num_instruct_candidates}"
            f"\nvalset size: {len(valset)}\n"
        )

    def _estimate_lm_calls(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        valset: list,
        program_aware_proposer: bool,
    ) -> tuple[str, str]:
        num_predictors = len(program.predictors())

        # Estimate prompt model calls
        estimated_prompt_model_calls = (
            10  # Data summarizer calls
            + self.num_instruct_candidates * num_predictors  # Candidate generation
            + (num_predictors + 1 if program_aware_proposer else 0)  # Program-aware proposer
        )
        prompt_model_line = (
            f"{YELLOW}- Prompt Generation: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + "
            f"{BLUE}{BOLD}{self.num_instruct_candidates}{ENDC}{YELLOW} * "
            f"{BLUE}{BOLD}{num_predictors}{ENDC}{YELLOW} lm calls in program "
            f"+ ({BLUE}{BOLD}{num_predictors + 1}{ENDC}{YELLOW}) lm calls in program-aware proposer "
            f"= {BLUE}{BOLD}{estimated_prompt_model_calls}{ENDC}{YELLOW} prompt model calls{ENDC}"
        )

        # Estimate task model calls
        if not minibatch:
            estimated_task_model_calls = len(valset) * num_trials
            task_model_line = (
                f"{YELLOW}- Program Evaluation: {BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * "
                f"{BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches = "
                f"{BLUE}{BOLD}{estimated_task_model_calls}{ENDC}{YELLOW} LM program calls{ENDC}"
            )
        else:
            full_eval_steps = num_trials // minibatch_full_eval_steps + 1
            estimated_task_model_calls = minibatch_size * num_trials + len(valset) * full_eval_steps
            task_model_line = (
                f"{YELLOW}- Program Evaluation: {BLUE}{BOLD}{minibatch_size}{ENDC}{YELLOW} examples in minibatch * "
                f"{BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches + "
                f"{BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * "
                f"{BLUE}{BOLD}{full_eval_steps}{ENDC}{YELLOW} full evals = "
                f"{BLUE}{BOLD}{estimated_task_model_calls}{ENDC}{YELLOW} LM Program calls{ENDC}"
            )

        return prompt_model_line, task_model_line

    def _bootstrap_fewshot_examples(self, program: Any, trainset: list, seed: int, teacher: Any) -> list | None:
        logger.info("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            logger.info(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            logger.info("These will be used for informing instruction proposal.\n")

        logger.info(f"Bootstrapping N={self.num_fewshot_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0

        # try:
        effective_max_errors = (
            self.max_errors if self.max_errors is not None else dspy.settings.max_errors
        )

        demo_candidates = create_n_fewshot_demo_sets(
            student=program,
            num_candidate_sets=self.num_fewshot_candidates,
            trainset=trainset,
            max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_labeled_demos),
            max_bootstrapped_demos=(
                BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_bootstrapped_demos
            ),
            metric=self.metric,
            max_errors=effective_max_errors,
            teacher=teacher,
            teacher_settings=self.teacher_settings,
            seed=seed,
            metric_threshold=self.metric_threshold,
            rng=self.rng,
        )
        # NOTE: Bootstrapping is essential to MIPRO!
        # Failing silently here makes the rest of the optimization far weaker as a result!
        # except Exception as e:
        #     logger.info(f"!!!!\n\n\n\n\nError generating few-shot examples: {e}")
        #     logger.info("Running without few-shot examples.!!!!\n\n\n\n\n")
        #     demo_candidates = None

        return demo_candidates

    def _propose_instructions(
        self,
        program: Any,
        trainset: list,
        demo_candidates: list | None,
        view_data_batch_size: int,
        program_aware_proposer: bool,
        data_aware_proposer: bool,
        tip_aware_proposer: bool,
        fewshot_aware_proposer: bool,
    ) -> dict[int, list[str]]:
        logger.info("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
        logger.info(
            "We will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions."
        )

        proposer = GroundedProposer(
            program=program,
            trainset=trainset,
            prompt_model=self.prompt_model,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            use_dataset_summary=data_aware_proposer,
            use_task_demos=fewshot_aware_proposer,
            num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
            use_tip=tip_aware_proposer,
            set_tip_randomly=tip_aware_proposer,
            use_instruct_history=False,
            set_history_randomly=False,
            verbose=self.verbose,
            rng=self.rng,
            init_temperature=self.init_temperature,
        )

        logger.info(f"\nProposing N={self.num_instruct_candidates} instructions...\n")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_instruct_candidates,
            trial_logs={},
        )

        for i, pred in enumerate(program.predictors()):
            logger.info(f"Proposed Instructions for Predictor {i}:\n")
            instruction_candidates[i][0] = get_signature(pred).instructions
            for j, instruction in enumerate(instruction_candidates[i]):
                logger.info(f"{j}: {instruction}\n")
            logger.info("\n")

        return instruction_candidates

    def _optimize_prompt_parameters(
        self,
        program: Any,
        instruction_candidates: dict[int, list[str]],
        demo_candidates: list | None,
        evaluate: Evaluate,
        valset: list,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
    ) -> Any | None:
        import optuna

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        logger.info(
            "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
        )

        # Compute the adjusted total trials that we will run (including full evals)
        run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
        adjusted_num_trials = int(
            (num_trials + num_trials // minibatch_full_eval_steps + 1 + run_additional_full_eval_at_end)
            if minibatch
            else num_trials
        )
        logger.info(f"== Trial {1} / {adjusted_num_trials} - Full Evaluation of Default Program ==")

        default_score = eval_candidate_program(len(valset), valset, program, evaluate, self.rng).score
        logger.info(f"Default program score: {default_score}\n")

        trial_logs = {}
        trial_logs[1] = {}
        trial_logs[1]["full_eval_program_path"] = save_candidate_program(program, self.log_dir, -1)
        trial_logs[1]["full_eval_score"] = default_score
        trial_logs[1]["total_eval_calls_so_far"] = len(valset)
        trial_logs[1]["full_eval_program"] = program.deepcopy()

        # Initialize optimization variables
        best_score = default_score
        best_program = program.deepcopy()
        total_eval_calls = len(valset)
        score_data = [{"score": best_score, "program": program.deepcopy(), "full_eval": True}]
        param_score_dict = defaultdict(list)
        fully_evaled_param_combos = {}

        # Define the objective function
        def objective(trial):
            nonlocal program, best_program, best_score, trial_logs, total_eval_calls, score_data

            trial_num = trial.number + 1
            if minibatch:
                logger.info(f"== Trial {trial_num} / {adjusted_num_trials} - Minibatch ==")
            else:
                logger.info(f"===== Trial {trial_num} / {num_trials} =====")

            trial_logs[trial_num] = {}

            # Create a new candidate program
            candidate_program = program.deepcopy()

            # Choose instructions and demos, insert them into the program
            chosen_params, raw_chosen_params = self._select_and_insert_instructions_and_demos(
                candidate_program,
                instruction_candidates,
                demo_candidates,
                trial,
                trial_logs,
                trial_num,
            )

            # Log assembled program
            if self.verbose:
                logger.info("Evaluating the following candidate program...\n")
                print_full_program(candidate_program)

            # Evaluate the candidate program (on minibatch if minibatch=True)
            batch_size = minibatch_size if minibatch else len(valset)
            score = eval_candidate_program(batch_size, valset, candidate_program, evaluate, self.rng).score
            total_eval_calls += batch_size

            # Update best score and program
            if not minibatch and score > best_score:
                best_score = score
                best_program = candidate_program.deepcopy()
                logger.info(f"{GREEN}Best full score so far!{ENDC} Score: {score}")

            # Log evaluation results
            score_data.append(
                {"score": score, "program": candidate_program, "full_eval": batch_size >= len(valset)}
            )  # score, prog, full_eval
            if minibatch:
                self._log_minibatch_eval(
                    score,
                    best_score,
                    batch_size,
                    chosen_params,
                    score_data,
                    trial,
                    adjusted_num_trials,
                    trial_logs,
                    trial_num,
                    candidate_program,
                    total_eval_calls,
                )
            else:
                self._log_normal_eval(
                    score,
                    best_score,
                    chosen_params,
                    score_data,
                    trial,
                    num_trials,
                    trial_logs,
                    trial_num,
                    valset,
                    batch_size,
                    candidate_program,
                    total_eval_calls,
                )
            categorical_key = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append(
                (score, candidate_program, raw_chosen_params),
            )

            # If minibatch, perform full evaluation at intervals (and at the very end)
            if minibatch and (
                (trial_num % (minibatch_full_eval_steps + 1) == 0) or (trial_num == (adjusted_num_trials - 1))
            ):
                best_score, best_program, total_eval_calls = self._perform_full_evaluation(
                    trial_num,
                    adjusted_num_trials,
                    param_score_dict,
                    fully_evaled_param_combos,
                    evaluate,
                    valset,
                    trial_logs,
                    total_eval_calls,
                    score_data,
                    best_score,
                    best_program,
                    study,
                    instruction_candidates,
                    demo_candidates,
                )

            return score

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predictors()))}
        if demo_candidates:
            default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.predictors()))})

        # Add default run as a baseline in optuna (TODO: figure out how to weight this by # of samples evaluated on)
        trial = optuna.trial.create_trial(
            params=default_params,
            distributions=self._get_param_distributions(program, instruction_candidates, demo_candidates),
            value=default_score,
        )
        study.add_trial(trial)
        study.optimize(objective, n_trials=num_trials)

        # Attach logs to best program
        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs
            best_program.score = best_score
            best_program.prompt_model_total_calls = self.prompt_model_total_calls
            best_program.total_calls = self.total_calls
            sorted_candidate_programs = sorted(score_data, key=lambda x: x["score"], reverse=True)
            # Attach all minibatch programs
            best_program.mb_candidate_programs = [
                score_data for score_data in sorted_candidate_programs if not score_data["full_eval"]
            ]
            # Attach all programs that were evaluated on the full trainset, in descending order of score
            best_program.candidate_programs = [
                score_data for score_data in sorted_candidate_programs if score_data["full_eval"]
            ]

        logger.info(f"Returning best identified program with score {best_score}!")

        return best_program

    def _log_minibatch_eval(
        self,
        score,
        best_score,
        batch_size,
        chosen_params,
        score_data,
        trial,
        adjusted_num_trials,
        trial_logs,
        trial_num,
        candidate_program,
        total_eval_calls,
    ):
        trial_logs[trial_num]["mb_program_path"] = save_candidate_program(candidate_program, self.log_dir, trial_num)
        trial_logs[trial_num]["mb_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["mb_program"] = candidate_program.deepcopy()

        logger.info(f"Score: {score} on minibatch of size {batch_size} with parameters {chosen_params}.")
        minibatch_scores = ", ".join([f"{s['score']}" for s in score_data if not s["full_eval"]])
        logger.info(f"Minibatch scores so far: {'[' + minibatch_scores + ']'}")
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(
            f"{'=' * len(f'== Trial {trial.number + 1} / {adjusted_num_trials} - Minibatch Evaluation ==')}\n\n"
        )

    def _log_normal_eval(
        self,
        score,
        best_score,
        chosen_params,
        score_data,
        trial,
        num_trials,
        trial_logs,
        trial_num,
        valset,
        batch_size,
        candidate_program,
        total_eval_calls,
    ):
        trial_logs[trial_num]["full_eval_program_path"] = save_candidate_program(
            candidate_program, self.log_dir, trial_num
        )
        trial_logs[trial_num]["full_eval_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["full_eval_program"] = candidate_program.deepcopy()

        logger.info(f"Score: {score} with parameters {chosen_params}.")
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        logger.info(f"Scores so far: {'[' + full_eval_scores + ']'}")
        logger.info(f"Best score so far: {best_score}")
        logger.info(f"{'=' * len(f'===== Trial {trial.number + 1} / {num_trials} =====')}\n\n")

    def _select_and_insert_instructions_and_demos(
        self,
        candidate_program: Any,
        instruction_candidates: dict[int, list[str]],
        demo_candidates: list | None,
        trial: "optuna.trial.Trial",
        trial_logs: dict,
        trial_num: int,
    ) -> list[str]:
        chosen_params = []
        raw_chosen_params = {}

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
            raw_chosen_params[f"{i}_predictor_instruction"] = instruction_idx
            # Select demos if available
            if demo_candidates:
                demos_idx = trial.suggest_categorical(f"{i}_predictor_demos", range(len(demo_candidates[i])))
                predictor.demos = demo_candidates[i][demos_idx]
                trial_logs[trial_num][f"{i}_predictor_demos"] = demos_idx
                chosen_params.append(f"Predictor {i}: Few-Shot Set {demos_idx}")
                raw_chosen_params[f"{i}_predictor_demos"] = instruction_idx

        return chosen_params, raw_chosen_params

    def _get_param_distributions(self, program, instruction_candidates, demo_candidates):
        from optuna.distributions import CategoricalDistribution

        param_distributions = {}

        for i in range(len(instruction_candidates)):
            param_distributions[f"{i}_predictor_instruction"] = CategoricalDistribution(
                range(len(instruction_candidates[i]))
            )
            if demo_candidates:
                param_distributions[f"{i}_predictor_demos"] = CategoricalDistribution(range(len(demo_candidates[i])))

        return param_distributions

    def _perform_full_evaluation(
        self,
        trial_num: int,
        adjusted_num_trials: int,
        param_score_dict: dict,
        fully_evaled_param_combos: dict,
        evaluate: Evaluate,
        valset: list,
        trial_logs: dict,
        total_eval_calls: int,
        score_data,
        best_score: float,
        best_program: Any,
        study: "optuna.Study",
        instruction_candidates: list,
        demo_candidates: list,
    ):
        import optuna

        logger.info(f"===== Trial {trial_num + 1} / {adjusted_num_trials} - Full Evaluation =====")

        # Identify best program to evaluate fully
        highest_mean_program, mean_score, combo_key, params = get_program_with_highest_avg_score(
            param_score_dict, fully_evaled_param_combos
        )
        logger.info(f"Doing full eval on next top averaging program (Avg Score: {mean_score}) from minibatch trials...")
        full_eval_score = eval_candidate_program(len(valset), valset, highest_mean_program, evaluate, self.rng).score
        score_data.append({"score": full_eval_score, "program": highest_mean_program, "full_eval": True})

        # Log full eval as a trial so that optuna can learn from the new results
        trial = optuna.trial.create_trial(
            params=params,
            distributions=self._get_param_distributions(best_program, instruction_candidates, demo_candidates),
            value=full_eval_score,
        )
        study.add_trial(trial)

        # Log full evaluation results
        fully_evaled_param_combos[combo_key] = {
            "program": highest_mean_program,
            "score": full_eval_score,
        }
        total_eval_calls += len(valset)
        trial_logs[trial_num + 1] = {}
        trial_logs[trial_num + 1]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num + 1]["full_eval_program_path"] = save_candidate_program(
            program=highest_mean_program,
            log_dir=self.log_dir,
            trial_num=trial_num + 1,
            note="full_eval",
        )
        trial_logs[trial_num + 1]["full_eval_program"] = highest_mean_program
        trial_logs[trial_num + 1]["full_eval_score"] = full_eval_score

        # Update best score and program if necessary
        if full_eval_score > best_score:
            logger.info(f"{GREEN}New best full eval score!{ENDC} Score: {full_eval_score}")
            best_score = full_eval_score
            best_program = highest_mean_program.deepcopy()
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(len(f"===== Full Eval {len(fully_evaled_param_combos) + 1} =====") * "=")
        logger.info("\n")

        return best_score, best_program, total_eval_calls
