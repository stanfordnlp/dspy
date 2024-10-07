import random
import sys
import textwrap
from collections import defaultdict

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
    get_task_model_history_for_full_example,
    print_full_program,
    save_candidate_program,
    set_signature,
)

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter using MIPRO, and evaluate it on an end task:

``` python
from dspy.teleprompt import MIPROv2

teleprompter = MIPROv2(prompt_model=prompt_model, task_model=task_model, metric=metric, num_candidates=10, init_temperature=1.0)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program, trainset=trainset[:TRAIN_NUM], num_trials=100, max_bootstrapped_demos=3, max_labeled_demos=5)
eval_score = evaluate(compiled_prompt_opt, devset=valset[:EVAL_NUM], **kwargs)
```

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (i.e., dspy.settings.configure(lm=task_model)).
* task_model: The model used for running your task. When unspecified, defaults to the model set in settings (i.e., dspy.settings.configure(lm=task_model)).
* teacher_settings: The settings used for the teacher model. When unspecified, defaults to the settings set in settings (i.e., dspy.settings.configure(lm=task_model)).
    The teacher settings are used to generate the fewshot examples.  This is the LLM/settings to use as a task model for the bootstrapping runs.
    Typically you would want to use a model of equal or greater quality to your task model.
* metric: The task metric used for optimization.
* num_candidates: The number of new prompts and sets of fewshot examples to generate and evaluate. Default=10.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.0.
* verbose: Tells the method whether or not to print intermediate steps.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track a dictionary with a key corresponding to the trial number, 
                and a value containing a dict with the following keys:
                    * program: the program being evaluated at a given trial
                    * score: the last average evaluated score for the program
                    * pruned: whether or not this program was pruned
                This information will be returned as attributes of the best program.
* log_dir: The directory to save logs and other information to. If unspecified, no logs will be saved.
* view_data_batch_size: The number of examples to view in the data batch when producing the dataset summary. Default=10.
* minibatch_size: The size of the minibatch to use when evaluating the program if using minibatched evaluations. Default=25.
* minibatch_full_eval_steps: The number of steps to take before doing a full evaluation of the program if using minibatched evaluations. Default=10.
* metric_threshold: If the metric yields a numerical value, then check it against this threshold when deciding whether or not to accept a bootstrap example.
"""

BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0

class MIPROv2(Teleprompter):
    def __init__(
        self,
        metric,
        prompt_model=dspy.settings.lm,
        task_model=dspy.settings.lm,
        teacher_settings={},
        num_candidates=10,
        num_threads=6,
        max_errors=10,
        init_temperature=0.5,
        verbose=False,
        track_stats=True,
        log_dir=None,
        metric_threshold=None,
    ):
        self.num_candidates = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.task_model = task_model
        self.prompt_model = prompt_model
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.teacher_settings = teacher_settings
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold

    def compile(
        self,
        student,
        *,
        trainset,
        valset=None,
        num_trials=30,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        seed=9,
        minibatch=True,
        minibatch_size=25,
        minibatch_full_eval_steps=10,
        program_aware_proposer=True,
        data_aware_proposer=True,
        view_data_batch_size=10,
        tip_aware_proposer=True,
        fewshot_aware_proposer=True,
        requires_permission_to_run=True,
    ):
        # Define ANSI escape codes for colors
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        ENDC = "\033[0m"  # Resets the color to default

        random.seed(seed)

        # Validate inputs
        if not trainset:
            raise ValueError("Trainset cannot be empty.")

        if not valset:
            if len(trainset) < 2:
                raise ValueError("Trainset must have at least 2 examples if no valset specified, or at least 1 example with external validation set.")
            
            valset_size = min(500, max(1, int(len(trainset) * 0.80))) # 80% of trainset, capped at 500
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]

        else:
            if len(valset) < 1:
                raise ValueError("Validation set must have at least 1 example if specified.")
            
        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset.  Note that your validation set contains {len(valset)} examples.  Your train set contains {len(trainset)} examples.")
        
        if minibatch and num_trials < minibatch_full_eval_steps:
            raise ValueError(f"Number of trials (num_trials={num_trials}) must be greater than or equal to the number of minibatch full eval steps (minibatch_full_eval_steps={minibatch_full_eval_steps}).")

        estimated_prompt_model_calls = 10 + self.num_candidates * len(
                student.predictors(),
            ) + (0 if not program_aware_proposer else len(student.predictors()) + 1)  # num data summary calls + N * P + (P + 1)
        
        prompt_model_line = ""
        if not program_aware_proposer:
            prompt_model_line = f"""{YELLOW}- Prompt Model: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + {BLUE}{BOLD}{self.num_candidates}{ENDC}{YELLOW} * {BLUE}{BOLD}{len(student.predictors())}{ENDC}{YELLOW} lm calls in program = {BLUE}{BOLD}{estimated_prompt_model_calls}{ENDC}{YELLOW} prompt model calls{ENDC}"""
        else:
            prompt_model_line = f"""{YELLOW}- Prompt Model: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + {BLUE}{BOLD}{self.num_candidates}{ENDC}{YELLOW} * {BLUE}{BOLD}{len(student.predictors())}{ENDC}{YELLOW} lm calls in program + ({BLUE}{BOLD}{len(student.predictors()) + 1}{ENDC}{YELLOW}) lm calls in program aware proposer = {BLUE}{BOLD}{estimated_prompt_model_calls}{ENDC}{YELLOW} prompt model calls{ENDC}"""

        estimated_task_model_calls_wo_module_calls = 0
        task_model_line = ""
        if not minibatch:
            estimated_task_model_calls_wo_module_calls = len(trainset) * num_trials  # M * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * {BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches * {BLUE}{BOLD}# of LM calls in your program{ENDC}{YELLOW} = ({BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls} * # of LM calls in your program{ENDC}{YELLOW}) task model calls{ENDC}"""
        else:
            estimated_task_model_calls_wo_module_calls = minibatch_size * num_trials + (len(trainset) * (num_trials // minibatch_full_eval_steps))  # B * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{minibatch_size}{ENDC}{YELLOW} examples in minibatch * {BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches + {BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * {BLUE}{BOLD}{num_trials // minibatch_full_eval_steps}{ENDC}{YELLOW} full evals = {BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls}{ENDC}{YELLOW} task model calls{ENDC}"""
            

        user_message = textwrap.dedent(f"""\
            {YELLOW}{BOLD}Projected Language Model (LM) Calls{ENDC}

            Please be advised that based on the parameters you have set, the maximum number of LM calls is projected as follows:

            
            {prompt_model_line}
            {task_model_line}

            {YELLOW}{BOLD}Estimated Cost Calculation:{ENDC}

            {YELLOW}Total Cost = (Number of calls to task model * (Avg Input Token Length per Call * Task Model Price per Input Token + Avg Output Token Length per Call * Task Model Price per Output Token) 
                        + (Number of calls to prompt model * (Avg Input Token Length per Call * Task Prompt Price per Input Token + Avg Output Token Length per Call * Prompt Model Price per Output Token).{ENDC}

            For a preliminary estimate of potential costs, we recommend you perform your own calculations based on the task
            and prompt models you intend to use. If the projected costs exceed your budget or expectations, you may consider:

            {YELLOW}- Reducing the number of trials (`num_trials`), the size of the valset, or the number of LM calls in your program.{ENDC}
            {YELLOW}- Using a cheaper task model to optimize the prompt.{ENDC}\n""")

        user_confirmation_message = textwrap.dedent(f"""\
            To proceed with the execution of this program, please confirm by typing {BLUE}'y'{ENDC} for yes or {BLUE}'n'{ENDC} for no.

            If you would like to bypass this confirmation step in future executions, set the {YELLOW}`requires_permission_to_run`{ENDC} flag to {YELLOW}`False`{ENDC} when calling compile.

            {YELLOW}Awaiting your input...{ENDC}
        """)

        if requires_permission_to_run: print(user_message)

        sys.stdout.flush()  # Flush the output buffer to force the message to print

        run = True
        # TODO: make sure these estimates are good for mini-batching
        if requires_permission_to_run:
            print(user_confirmation_message)
            user_input = input("Do you wish to continue? (y/n): ").strip().lower()
            if user_input != "y":
                print("Compilation aborted by the user.")
                run = False

        if run:
            # Setup random seeds
            random.seed(seed)
            np.random.seed(seed)

            # Set up program and evaluation function
            program = student.deepcopy()
            evaluate = Evaluate(
                devset=valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                display_table=False,
                display_progress=True,
            )

            # Determine the number of fewshot examples to use to generate demos for prompt
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0:
                max_bootstrapped_demos_for_candidate_gen = BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT
                max_labeled_demos_for_candidate_gen = LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT
            else:
                max_bootstrapped_demos_for_candidate_gen = max_bootstrapped_demos
                max_labeled_demos_for_candidate_gen = max_labeled_demos
            
            # Generate N few shot example sets (these will inform instruction creation, and be used as few-shot examples in our prompt)
            print("Beginning MIPROv2 optimization process...")
            print("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
            if max_bootstrapped_demos > 0:
                print("These will be used for as few-shot examples candidates for our program and for creating instructions.\n")
            else:
                print("These will be used for informing instruction proposal.\n")
            print(f"Bootstrapping N={self.num_candidates} sets of demonstrations...")
            try:
                demo_candidates = create_n_fewshot_demo_sets(
                    student=program,
                    num_candidate_sets=self.num_candidates,
                    trainset=trainset,
                    max_labeled_demos=max_labeled_demos_for_candidate_gen,
                    max_bootstrapped_demos=max_bootstrapped_demos_for_candidate_gen,
                    metric=self.metric,
                    max_errors=self.max_errors,
                    teacher_settings=self.teacher_settings,
                    seed=seed,
                    metric_threshold=self.metric_threshold,
                )
            except Exception as e:
                print(f"Error generating fewshot examples: {e}")
                print("Running without fewshot examples.")
                demo_candidates = None
            
            # Generate N candidate instructions

            # Setup our proposer 
            print("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
            print("In this step, by default we will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions.")

            proposer = GroundedProposer(
                program=program,
                trainset=trainset,
                prompt_model=self.prompt_model,
                # program_code_string=self.program_code_string,
                view_data_batch_size=view_data_batch_size,
                program_aware=program_aware_proposer,
                use_dataset_summary=data_aware_proposer,
                use_task_demos=fewshot_aware_proposer,
                use_tip=tip_aware_proposer,
                set_tip_randomly=tip_aware_proposer,
                use_instruct_history=False,
                set_history_randomly=False,
                verbose = self.verbose,
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

            # If we're doing zero-shot, reset demo_candidates to none now that we've used them for instruction proposal
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0:
                demo_candidates = None

            # Initialize variables to track during the optimization process
            best_scoring_trial = 0
            trial_logs = {}
            total_eval_calls = 0
            param_score_dict = defaultdict(list) # Dictionaries of paramater combinations we've tried, and their associated scores
            fully_evaled_param_combos = {} # List of the parameter combinations we've done full evals of

            # Evaluate the default program
            print("Evaluating the default program...\n")
            default_score = eval_candidate_program(len(valset), valset, program, evaluate)
            print(f"Default program score: {default_score}\n")

            best_score = default_score
            best_program = program.deepcopy()


            # Define our trial objective
            def create_objective(
                baseline_program,
                instruction_candidates,
                demo_candidates,
                evaluate,
                valset,
            ):
                def objective(trial):
                    nonlocal best_program, best_score, best_scoring_trial, trial_logs, total_eval_calls  # Allow access to the outer variables
                    
                    # Kick off trial
                    if minibatch:
                        print(f"== Minibatch Trial {trial.number+1} / {num_trials} ==")
                    else:
                        print(f"===== Trial {trial.number+1} / {num_trials} =====")
                    trial_logs[trial.number+1] = {}

                    # Create a new candidate program
                    candidate_program = baseline_program.deepcopy()

                    # Choose set of instructions & demos to use for each predictor
                    chosen_params = []
                    for i, p_new in enumerate(candidate_program.predictors()):

                        # Get instruction candidates / demos for our given predictor
                        p_instruction_candidates = instruction_candidates[i]
                        if demo_candidates:
                            p_demo_candidates = demo_candidates[i]

                        # Suggest the index of the instruction / demo candidate to use in our trial
                        instruction_idx = trial.suggest_categorical(
                            f"{i}_predictor_instruction",
                            range(len(p_instruction_candidates)),
                        )
                        # chosen_params.append(instruction_idx)
                        chosen_params.append(f"Predictor {i+1}: Instruction {instruction_idx}")
                        if demo_candidates:
                            demos_idx = trial.suggest_categorical(
                                f"{i}_predictor_demos", range(len(p_demo_candidates)),
                            )
                            chosen_params.append(f"Predictor {i+1}: Few-Shot Set {demos_idx}")

                        # Log the selected instruction / demo candidate
                        trial_logs[trial.number+1][
                            f"{i}_predictor_instruction"
                        ] = instruction_idx
                        if demo_candidates:
                            trial_logs[trial.number+1][f"{i}_predictor_demos"] = demos_idx

                        dspy.logger.debug(f"instruction_idx {instruction_idx}")
                        if demo_candidates:
                            dspy.logger.debug(f"demos_idx {demos_idx}")

                        # Set the instruction
                        selected_instruction = p_instruction_candidates[instruction_idx]
                        updated_signature = get_signature(p_new).with_instructions(
                            selected_instruction,
                        )
                        set_signature(p_new, updated_signature)

                        # Set the demos
                        if demo_candidates:
                            p_new.demos = p_demo_candidates[demos_idx]

                    # Log assembled program
                    if self.verbose: print("Evaluating the following candidate program...\n")
                    if self.verbose: print_full_program(candidate_program)

                    # Save the candidate program
                    trial_logs[trial.number+1]["program_path"] = save_candidate_program(
                        candidate_program, self.log_dir, trial.number+1,
                    )

                    trial_logs[trial.number+1]["num_eval_calls"] = 0

                    # Evaluate the candidate program with relevant batch size
                    batch_size = minibatch_size if minibatch else len(valset)

                    score = eval_candidate_program(
                        batch_size, valset, candidate_program, 
                        evaluate,
                    )

                    # Print out a full trace of the program in use
                    if self.verbose:
                        print("Full trace of prompts in use on an example...")
                        get_task_model_history_for_full_example(
                            candidate_program, self.task_model, valset, evaluate,
                        )

                    # Log relevant information
                    categorical_key = ",".join(map(str, chosen_params))
                    param_score_dict[categorical_key].append(
                        (score, candidate_program),
                    )
                    trial_logs[trial.number+1]["num_eval_calls"] = batch_size
                    trial_logs[trial.number+1]["full_eval"] = batch_size >= len(valset)
                    trial_logs[trial.number+1]["score"] = score
                    trial_logs[trial.number+1]["pruned"] = False
                    total_eval_calls += trial_logs[trial.number+1]["num_eval_calls"]
                    trial_logs[trial.number+1]["total_eval_calls_so_far"] = total_eval_calls
                    trial_logs[trial.number+1]["program"] = candidate_program.deepcopy()

                    # If this score was from a full evaluation, update the best program if the new score is better
                    best_score_updated = False
                    if score > best_score and trial_logs[trial.number+1]["full_eval"] and not minibatch:
                        best_score = score
                        best_scoring_trial = trial.number+1
                        best_program = candidate_program.deepcopy()
                        best_score_updated = True
                    
                    if minibatch:
                        print(f"Score: {score} on minibatch of size {batch_size} with parameters {chosen_params}.\n\n")
                    else:
                        print(f"Score: {score} with parameters {chosen_params}.")
                        if best_score_updated:
                            print(f"{GREEN}New best score updated!{ENDC} Score: {best_score} on trial {best_scoring_trial}.\n\n")
                        else:
                            print(f"Best score so far: {best_score} on trial {best_scoring_trial}.\n\n")
                        

                    # If we're doing minibatching, check to see if it's time to do a full eval
                    if minibatch and (((trial.number+1) % minibatch_full_eval_steps == 0) or (trial.number+1 == num_trials)):
                        print(f"===== Full Eval {len(fully_evaled_param_combos)+1} =====")
                        
                        # Save old information as the minibatch version
                        trial_logs[trial.number+1]["mb_score"] = score
                        trial_logs[trial.number+1]["mb_program_path"] = trial_logs[trial.number+1]["program_path"]

                        # Identify our best program (based on mean of scores so far, and do a full eval on it)
                        highest_mean_program, mean, combo_key = get_program_with_highest_avg_score(param_score_dict, fully_evaled_param_combos)
                        
                        if trial.number+1 // minibatch_full_eval_steps > 0:
                            print(f"Doing full eval on next top averaging program (Avg Score: {mean}) so far from mini-batch trials...")
                        else:
                            print(f"Doing full eval on top averaging program (Avg Score: {mean}) so far from mini-batch trials...")
                        full_val_score = eval_candidate_program(
                            len(valset), valset, highest_mean_program, evaluate,
                        )

                        # Log relevant information
                        fully_evaled_param_combos[combo_key] = {"program":highest_mean_program, "score": full_val_score}
                        total_eval_calls += len(valset)
                        trial_logs[trial.number+1]["total_eval_calls_so_far"] = total_eval_calls
                        trial_logs[trial.number+1]["full_eval"] = True
                        trial_logs[trial.number+1]["program_path"] = save_candidate_program(
                            program=highest_mean_program, log_dir=self.log_dir, trial_num=trial.number+1, note="full_eval",
                        )
                        trial_logs[trial.number+1]["score"] = full_val_score
                        
                        if full_val_score > best_score:
                            print(f"{GREEN}Best full eval score so far!{ENDC} Score: {full_val_score}")
                            best_score = full_val_score
                            best_scoring_trial = trial.number+1
                            best_program = highest_mean_program.deepcopy()
                            best_score_updated = True
                        else:
                            print(f"Full eval score: {full_val_score}")
                            print(f"Best full eval score so far: {best_score}")
                        print("=======================\n\n")
                    
                    return score

                return objective

            # Run the trial
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            print("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
            print("In this step, we will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination. Bayesian Optimization will be used for this search process.\n")
            objective_function = create_objective(
                program, instruction_candidates, demo_candidates, evaluate, valset,
            )
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            _ = study.optimize(objective_function, n_trials=num_trials)

            if best_program is not None and self.track_stats:
                best_program.trial_logs = trial_logs
                best_program.score = best_score
                best_program.prompt_model_total_calls = self.prompt_model_total_calls
                best_program.total_calls = self.total_calls

            return best_program

        return student
