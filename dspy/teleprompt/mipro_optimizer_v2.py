import random
import sys
import textwrap
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import dspy

from dspy.evaluate.evaluate import Evaluate
from dspy.propose import GroundedProposer
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import (
    create_n_fewshot_demo_sets,
    eval_candidate_program,
    get_program_with_highest_avg_score,
    get_signature,
    print_full_program,
    save_candidate_program,
    set_signature,
    create_minibatch,
)

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

AUTO_RUN_SETTINGS = {
    "light": {"num_trials": 7, "val_size": 100},
    "medium": {"num_trials": 25, "val_size": 300},
    "heavy": {"num_trials": 50, "val_size": 1000},
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
        prompt_model: Optional[Any] = None,
        task_model: Optional[Any] = None,
        teacher_settings: Dict = {},
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
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
    ):
        # Validate 'auto' parameter
        allowed_modes = {None, "light", "medium", "heavy"}
        if auto not in allowed_modes:
            raise ValueError(
                f"Invalid value for auto: {auto}. Must be one of {allowed_modes}."
            )
        self.auto = auto

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
        self.teacher_settings = teacher_settings
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.seed = seed
        self.rng = None

    def compile(
        self,
        student: Any,
        *,
        trainset: List,
        valset: Optional[List] = None,
        num_trials: int = 30,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        seed: Optional[int] = None,
        minibatch: bool = True,
        minibatch_size: int = 25,
        minibatch_full_eval_steps: int = 10,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool = True,
    ) -> Any:
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
        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (
            self.max_labeled_demos == 0
        )
        num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
            student, num_trials, minibatch, zeroshot_opt, valset
        )

        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(
                f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}."
            )

        # Estimate LM calls and get user confirmation
        if requires_permission_to_run:
            if not self._get_user_confirmation(
                student,
                num_trials,
                minibatch,
                minibatch_size,
                minibatch_full_eval_steps,
                valset,
                program_aware_proposer,
            ):
                print("Compilation aborted by the user.")
                return student  # Return the original student program

        # Initialize program and evaluator
        program = student.deepcopy()
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=self.max_errors,
            display_table=False,
            display_progress=True,
        )

        # Step 1: Bootstrap few-shot examples
        demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed)

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
    
    def _set_random_seeds(self,
        seed
    ):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _set_hyperparams_from_run_mode(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        zeroshot_opt: bool,
        valset: List,
    ) -> Tuple[int, List, bool]:
        if self.auto is None:
            return num_trials, valset, minibatch

        num_vars = len(program.predictors())
        if not zeroshot_opt:
            num_vars *= 2  # Account for few-shot examples + instruction variables

        auto_settings = AUTO_RUN_SETTINGS[self.auto]
        num_trials = auto_settings["num_trials"]
        valset = create_minibatch(valset, batch_size=auto_settings["val_size"], rng=self.rng)
        minibatch = len(valset) > MIN_MINIBATCH_SIZE
        self.num_candidates = int(
            np.round(np.min([num_trials * num_vars, (1.5 * num_trials) / num_vars]))
        )

        return num_trials, valset, minibatch

    def _set_and_validate_datasets(self, trainset: List, valset: Optional[List]):
        if not trainset:
            raise ValueError("Trainset cannot be empty.")

        if valset is None:
            if len(trainset) < 2:
                raise ValueError(
                    "Trainset must have at least 2 examples if no valset specified."
                )
            valset_size = min(500, max(1, int(len(trainset) * 0.80)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]
        else:
            if len(valset) < 1:
                raise ValueError("Validation set must have at least 1 example.")

        return trainset, valset

    def _print_auto_run_settings(self, num_trials: int, minibatch: bool, valset: List):
        print(
            f"\nRUNNING WITH THE FOLLOWING {self.auto.upper()} AUTO RUN SETTINGS:"
            f"\nnum_trials: {num_trials}"
            f"\nminibatch: {minibatch}"
            f"\nnum_candidates: {self.num_candidates}"
            f"\nvalset size: {len(valset)}\n"
        )

    def _estimate_lm_calls(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        valset: List,
        program_aware_proposer: bool,
    ) -> Tuple[str, str]:
        num_predictors = len(program.predictors())

        # Estimate prompt model calls
        estimated_prompt_model_calls = (
            10  # Data summarizer calls
            + self.num_candidates * num_predictors  # Candidate generation
            + (
                num_predictors + 1 if program_aware_proposer else 0
            )  # Program-aware proposer
        )
        prompt_model_line = (
            f"{YELLOW}- Prompt Generation: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + "
            f"{BLUE}{BOLD}{self.num_candidates}{ENDC}{YELLOW} * "
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
            estimated_task_model_calls = (
                minibatch_size * num_trials + len(valset) * full_eval_steps
            )
            task_model_line = (
                f"{YELLOW}- Program Evaluation: {BLUE}{BOLD}{minibatch_size}{ENDC}{YELLOW} examples in minibatch * "
                f"{BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches + "
                f"{BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * "
                f"{BLUE}{BOLD}{full_eval_steps}{ENDC}{YELLOW} full evals = "
                f"{BLUE}{BOLD}{estimated_task_model_calls}{ENDC}{YELLOW} LM Program calls{ENDC}"
            )

        return prompt_model_line, task_model_line

    def _get_user_confirmation(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        valset: List,
        program_aware_proposer: bool,
    ) -> bool:
        prompt_model_line, task_model_line = self._estimate_lm_calls(
            program,
            num_trials,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            valset,
            program_aware_proposer,
        )

        user_message = textwrap.dedent(
            f"""\
            {YELLOW}{BOLD}Projected Language Model (LM) Calls{ENDC}

            Based on the parameters you have set, the maximum number of LM calls is projected as follows:

            {prompt_model_line}
            {task_model_line}

            {YELLOW}{BOLD}Estimated Cost Calculation:{ENDC}

            {YELLOW}Total Cost = (Number of calls to task model * (Avg Input Token Length per Call * Task Model Price per Input Token + Avg Output Token Length per Call * Task Model Price per Output Token) 
                        + (Number of program calls * (Avg Input Token Length per Call * Task Prompt Price per Input Token + Avg Output Token Length per Call * Prompt Model Price per Output Token).{ENDC}

            For a preliminary estimate of potential costs, we recommend you perform your own calculations based on the task
            and prompt models you intend to use. If the projected costs exceed your budget or expectations, you may consider:

            {YELLOW}- Reducing the number of trials (`num_trials`), the size of the valset, or the number of LM calls in your program.{ENDC}
            {YELLOW}- Using a cheaper task model to optimize the prompt.{ENDC}
            {YELLOW}- Setting `minibatch=True` if you haven't already.{ENDC}\n"""
        )

        user_confirmation_message = textwrap.dedent(
            f"""\
            To proceed with the execution of this program, please confirm by typing {BLUE}'y'{ENDC} for yes or {BLUE}'n'{ENDC} for no.

            If you would like to bypass this confirmation step in future executions, set the {YELLOW}`requires_permission_to_run`{ENDC} flag to {YELLOW}`False`{ENDC} when calling compile.

            {YELLOW}Awaiting your input...{ENDC}
        """
        )

        print(user_message)
        sys.stdout.flush()
        print(user_confirmation_message)
        user_input = input("Do you wish to continue? (y/n): ").strip().lower()
        return user_input == "y"

    def _bootstrap_fewshot_examples(
        self, program: Any, trainset: List, seed: int
    ) -> Optional[List]:
        print("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            print(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            print("These will be used for informing instruction proposal.\n")

        print(f"Bootstrapping N={self.num_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0

        try:
            demo_candidates = create_n_fewshot_demo_sets(
                student=program,
                num_candidate_sets=self.num_candidates,
                trainset=trainset,
                max_labeled_demos=(
                    LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT
                    if zeroshot
                    else self.max_labeled_demos
                ),
                max_bootstrapped_demos=(
                    BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT
                    if zeroshot
                    else self.max_bootstrapped_demos
                ),
                metric=self.metric,
                max_errors=self.max_errors,
                teacher_settings=self.teacher_settings,
                seed=seed,
                metric_threshold=self.metric_threshold,
                rng=self.rng,
            )
        except Exception as e:
            print(f"Error generating few-shot examples: {e}")
            print("Running without few-shot examples.")
            demo_candidates = None

        return demo_candidates

    def _propose_instructions(
        self,
        program: Any,
        trainset: List,
        demo_candidates: Optional[List],
        view_data_batch_size: int,
        program_aware_proposer: bool,
        data_aware_proposer: bool,
        tip_aware_proposer: bool,
        fewshot_aware_proposer: bool,
    ) -> Dict[int, List[str]]:
        print("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
        print(
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
            use_tip=tip_aware_proposer,
            set_tip_randomly=tip_aware_proposer,
            use_instruct_history=False,
            set_history_randomly=False,
            verbose=self.verbose,
            rng=self.rng
        )

        print("\nProposing instructions...\n")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_candidates,
            T=self.init_temperature,
            trial_logs={},
        )

        for i, pred in enumerate(program.predictors()):
            print(f"Proposed Instructions for Predictor {i}:\n")
            instruction_candidates[i][0] = get_signature(pred).instructions
            for j, instruction in enumerate(instruction_candidates[i]):
                print(f"{j}: {instruction}\n")
            print("\n")

        return instruction_candidates

    def _optimize_prompt_parameters(
        self,
        program: Any,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[List],
        evaluate: Evaluate,
        valset: List,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
    ) -> Optional[Any]:
        print("Evaluating the default program...\n")
        default_score = eval_candidate_program(len(valset), valset, program, evaluate, self.rng)
        print(f"Default program score: {default_score}\n")

        # Initialize optimization variables
        best_score = default_score
        best_program = program.deepcopy()
        trial_logs = {}
        total_eval_calls = 0
        if minibatch:
            scores = []
        else:
            scores = [default_score]
        full_eval_scores = [default_score]
        param_score_dict = defaultdict(list)
        fully_evaled_param_combos = {}

        # Define the objective function
        def objective(trial):
            nonlocal program, best_program, best_score, trial_logs, total_eval_calls, scores, full_eval_scores

            trial_num = trial.number + 1
            if minibatch:
                print(f"== Minibatch Trial {trial_num} / {num_trials} ==")
            else:
                print(f"===== Trial {trial_num} / {num_trials} =====")

            trial_logs[trial_num] = {}

            # Create a new candidate program
            candidate_program = program.deepcopy()

            # Choose instructions and demos, insert them into the program
            chosen_params = self._select_and_insert_instructions_and_demos(
                candidate_program,
                instruction_candidates,
                demo_candidates,
                trial,
                trial_logs,
                trial_num,
            )

            # Log assembled program
            if self.verbose:
                print("Evaluating the following candidate program...\n")
                print_full_program(candidate_program)

            # Save the candidate program
            trial_logs[trial_num]["program_path"] = save_candidate_program(
                candidate_program, self.log_dir, trial_num
            )

            # Evaluate the candidate program
            batch_size = minibatch_size if minibatch else len(valset)
            score = eval_candidate_program(
                batch_size, valset, candidate_program, evaluate, self.rng
            )

            # Update best score and program
            if not minibatch and score > best_score:
                best_score = score
                best_program = candidate_program.deepcopy()
                print(f"{GREEN}Best full score so far!{ENDC} Score: {score}")

            # Log evaluation results
            scores.append(score)
            if minibatch:
                self._log_minibatch_eval(
                    score,
                    best_score,
                    batch_size,
                    chosen_params,
                    scores,
                    full_eval_scores,
                    trial,
                    num_trials,
                )
            else:
                self._log_normal_eval(
                    score, best_score, chosen_params, scores, trial, num_trials
                )
            categorical_key = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append(
                (score, candidate_program),
            )
            trial_logs[trial_num]["num_eval_calls"] = batch_size
            trial_logs[trial_num]["full_eval"] = batch_size >= len(valset)
            trial_logs[trial_num]["score"] = score
            total_eval_calls += batch_size
            trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
            trial_logs[trial_num]["program"] = candidate_program.deepcopy()

            # If minibatch, perform full evaluation at intervals
            if minibatch and (
                (trial_num % minibatch_full_eval_steps == 0)
                or (trial_num == num_trials)
            ):
                best_score, best_program = self._perform_full_evaluation(
                    trial_num,
                    param_score_dict,
                    fully_evaled_param_combos,
                    evaluate,
                    valset,
                    trial_logs,
                    total_eval_calls,
                    full_eval_scores,
                    best_score,
                    best_program,
                )

            return score

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        print(
            "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
        )

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=num_trials)

        # Attach logs to best program
        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs
            best_program.score = best_score
            best_program.prompt_model_total_calls = self.prompt_model_total_calls
            best_program.total_calls = self.total_calls

        print(f"Returning best identified program with score {best_score}!")

        return best_program

    def _log_minibatch_eval(
        self,
        score,
        best_score,
        batch_size,
        chosen_params,
        scores,
        full_eval_scores,
        trial,
        num_trials,
    ):
        print(
            f"Score: {score} on minibatch of size {batch_size} with parameters {chosen_params}."
        )
        print(f"Minibatch scores so far: {'['+', '.join([f'{s}' for s in scores])+']'}")
        trajectory = "[" + ", ".join([f"{s}" for s in full_eval_scores]) + "]"
        print(f"Full eval scores so far: {trajectory}")
        print(f"Best full score so far: {best_score}")
        print(
            f'{"="*len(f"== Minibatch Trial {trial.number+1} / {num_trials} ==")}\n\n'
        )

    def _log_normal_eval(
        self, score, best_score, chosen_params, scores, trial, num_trials
    ):
        print(f"Score: {score} with parameters {chosen_params}.")
        print(f"Scores so far: {'['+', '.join([f'{s}' for s in scores])+']'}")
        print(f"Best score so far: {best_score}")
        print(f'{"="*len(f"===== Trial {trial.number+1} / {num_trials} =====")}\n\n')

    def _select_and_insert_instructions_and_demos(
        self,
        candidate_program: Any,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[List],
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
            updated_signature = get_signature(predictor).with_instructions(
                selected_instruction
            )
            set_signature(predictor, updated_signature)
            trial_logs[trial_num][f"{i}_predictor_instruction"] = instruction_idx
            chosen_params.append(f"Predictor {i+1}: Instruction {instruction_idx}")

            # Select demos if available
            if demo_candidates:
                demos_idx = trial.suggest_categorical(
                    f"{i}_predictor_demos", range(len(demo_candidates[i]))
                )
                predictor.demos = demo_candidates[i][demos_idx]
                trial_logs[trial_num][f"{i}_predictor_demos"] = demos_idx
                chosen_params.append(f"Predictor {i+1}: Few-Shot Set {demos_idx}")

        return chosen_params

    def _perform_full_evaluation(
        self,
        trial_num: int,
        param_score_dict: Dict,
        fully_evaled_param_combos: Dict,
        evaluate: Evaluate,
        valset: List,
        trial_logs: Dict,
        total_eval_calls: int,
        full_eval_scores: List[int],
        best_score: float,
        best_program: Any,
    ):
        print(f"===== Full Eval {len(fully_evaled_param_combos)+1} =====")

        # Identify best program to evaluate fully
        highest_mean_program, mean_score, combo_key = (
            get_program_with_highest_avg_score(
                param_score_dict, fully_evaled_param_combos
            )
        )
        print(
            f"Doing full eval on next top averaging program (Avg Score: {mean_score}) from minibatch trials..."
        )
        full_eval_score = eval_candidate_program(
            len(valset), valset, highest_mean_program, evaluate, self.rng
        )
        full_eval_scores.append(full_eval_score)

        # Log full evaluation results
        fully_evaled_param_combos[combo_key] = {
            "program": highest_mean_program,
            "score": full_eval_score,
        }
        total_eval_calls += len(valset)
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["full_eval"] = True
        trial_logs[trial_num]["program_path"] = save_candidate_program(
            program=highest_mean_program,
            log_dir=self.log_dir,
            trial_num=trial_num,
            note="full_eval",
        )
        trial_logs[trial_num]["score"] = full_eval_score

        # Update best score and program if necessary
        if full_eval_score > best_score:
            print(f"{GREEN}New best full eval score!{ENDC} Score: {full_eval_score}")
            best_score = full_eval_score
            best_program = highest_mean_program.deepcopy()
        trajectory = "[" + ", ".join([f"{s}" for s in full_eval_scores]) + "]"
        print(f"Full eval scores so far: {trajectory}")
        print(f"Best full score so far: {best_score}")
        print(len(f"===== Full Eval {len(fully_evaled_param_combos)+1} =====") * "=")
        print("\n")

        return best_score, best_program
