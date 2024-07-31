import logging
import os
import pickle
import random
import sys
import textwrap
from collections import defaultdict

import numpy as np
import optuna

from dspy.evaluate.evaluate import Evaluate
from dspy.propose import GroundedProposer
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import (
    create_n_fewshot_demo_sets,
    eval_candidate_program,
    get_dspy_source_code,
    get_program_with_highest_avg_score,
    get_signature,
    get_task_model_history_for_full_example,
    print_full_program,
    save_candidate_program,
    save_file_to_log_dir,
    set_signature,
    setup_logging,
)

try:
    import wandb
except ImportError:
    wandb = None

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter using MIPRO, and evaluate it on an end task:

``` python
from dspy.teleprompt import MIPROv2

teleprompter = MIPROv2(prompt_model=prompt_model, task_model=task_model, metric=metric, num_candidates=10, init_temperature=1.0)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program, trainset=trainset[:TRAIN_NUM], num_batches=100, max_bootstrapped_demos=3, max_labeled_demos=5, eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)
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

MB_FULL_EVAL_STEPS = 10
MINIBATCH_SIZE = 25#50

class MIPROv2(Teleprompter):
    def __init__(
        self,
        prompt_model=None,
        task_model=None,
        teacher_settings={},
        num_candidates=10,
        metric=None,
        init_temperature=1.4,
        verbose=False,
        track_stats=True,
        log_dir=None,
        view_data_batch_size=10,
        minibatch_size=MINIBATCH_SIZE,
        minibatch_full_eval_steps=MB_FULL_EVAL_STEPS,
        metric_threshold=None,
    ):
        self.n = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.task_model = task_model
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.view_data_batch_size = view_data_batch_size
        self.teacher_settings = teacher_settings
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.minibatch_size = minibatch_size
        self.minibatch_full_eval_steps = minibatch_full_eval_steps
        self.metric_threshold = None

        # Check if WANDB_RUN_ID is set in the environment
        self.wandb_run_id = None
    
    def _get_batch_size(
        self,
        minibatch,
        trainset,
    ):
        if minibatch:
            return self.minibatch_size
        else:
            return len(trainset)

    def compile(
        self,
        student,
        *,
        trainset,
        valset=None,
        num_batches=30,
        max_bootstrapped_demos=5,
        max_labeled_demos=2,
        eval_kwargs={},
        seed=9,
        minibatch=True,
        program_aware_proposer=True,
        requires_permission_to_run=True,
    ):
        # Define ANSI escape codes for colors
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        ENDC = "\033[0m"  # Resets the color to default

        random.seed(seed)
        valset = valset or trainset
        estimated_prompt_model_calls = 10 + self.n * len(
                student.predictors(),
            ) + (0 if not program_aware_proposer else len(student.predictors()) + 1)  # num data summary calls + N * P + (P + 1)
        
        prompt_model_line = ""
        if not program_aware_proposer:
            prompt_model_line = f"""{YELLOW}- Prompt Model: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + {BLUE}{BOLD}{self.n}{ENDC}{YELLOW} * {BLUE}{BOLD}{len(student.predictors())}{ENDC}{YELLOW} lm calls in program = {BLUE}{BOLD}{estimated_prompt_model_calls}{ENDC}{YELLOW} prompt model calls{ENDC}"""
        else:
            prompt_model_line = f"""{YELLOW}- Prompt Model: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + {BLUE}{BOLD}{self.n}{ENDC}{YELLOW} * {BLUE}{BOLD}{len(student.predictors())}{ENDC}{YELLOW} lm calls in program + ({BLUE}{BOLD}{len(student.predictors()) + 1}{ENDC}{YELLOW}) lm calls in program aware proposer = {BLUE}{BOLD}{estimated_prompt_model_calls}{ENDC}{YELLOW} prompt model calls{ENDC}"""

        estimated_task_model_calls_wo_module_calls = 0
        task_model_line = ""
        if not minibatch:
            estimated_task_model_calls_wo_module_calls = len(trainset) * num_batches  # M * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{len(trainset)}{ENDC}{YELLOW} examples in train set * {BLUE}{BOLD}{num_batches}{ENDC}{YELLOW} batches * {BLUE}{BOLD}# of LM calls in your program{ENDC}{YELLOW} = ({BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls} * # of LM calls in your program{ENDC}{YELLOW}) task model calls{ENDC}"""
        else:
            estimated_task_model_calls_wo_module_calls = self.minibatch_size * num_batches + (len(trainset) * (num_batches // self.minibatch_full_eval_steps))  # B * T * P
            task_model_line = f"""{YELLOW}- Task Model: {BLUE}{BOLD}{self.minibatch_size}{ENDC}{YELLOW} examples in minibatch * {BLUE}{BOLD}{num_batches}{ENDC}{YELLOW} batches + {BLUE}{BOLD}{len(trainset)}{ENDC}{YELLOW} examples in train set * {BLUE}{BOLD}{num_batches // self.minibatch_full_eval_steps}{ENDC}{YELLOW} full evals = {BLUE}{BOLD}{estimated_task_model_calls_wo_module_calls}{ENDC}{YELLOW} task model calls{ENDC}"""
            

        user_message = textwrap.dedent(f"""\
            {YELLOW}{BOLD}WARNING: Projected Language Model (LM) Calls{ENDC}

            Please be advised that based on the parameters you have set, the maximum number of LM calls is projected as follows:

            
            {prompt_model_line}
            {task_model_line}

            {YELLOW}{BOLD}Estimated Cost Calculation:{ENDC}

            {YELLOW}Total Cost = (Number of calls to task model * (Avg Input Token Length per Call * Task Model Price per Input Token + Avg Output Token Length per Call * Task Model Price per Output Token) 
                        + (Number of calls to prompt model * (Avg Input Token Length per Call * Task Prompt Price per Input Token + Avg Output Token Length per Call * Prompt Model Price per Output Token).{ENDC}

            For a preliminary estimate of potential costs, we recommend you perform your own calculations based on the task
            and prompt models you intend to use. If the projected costs exceed your budget or expectations, you may consider:

            {YELLOW}- Reducing the number of trials (`num_batches`), the size of the trainset, or the number of LM calls in your program.{ENDC}
            {YELLOW}- Using a cheaper task model to optimize the prompt.{ENDC}""")

        user_confirmation_message = textwrap.dedent(f"""\
            To proceed with the execution of this program, please confirm by typing {BLUE}'y'{ENDC} for yes or {BLUE}'n'{ENDC} for no.

            If you would like to bypass this confirmation step in future executions, set the {YELLOW}`requires_permission_to_run`{ENDC} flag to {YELLOW}`False` when calling compile.{ENDC}

            {YELLOW}Awaiting your input...{ENDC}
        """)

        print(user_message)

        sys.stdout.flush()  # Flush the output buffer to force the message to print

        run = True
        if requires_permission_to_run:
            print(user_confirmation_message)
            user_input = input("Do you wish to continue? (y/n): ").strip().lower()
            if user_input != "y":
                print("Compilation aborted by the user.")
                run = False

        if run:
            if program_aware_proposer:
                try:
                    self.program_code_string = get_dspy_source_code(student)
                    if self.verbose:
                        print("SOURCE CODE:",self.program_code_string)
                except Exception as e:
                    print(f"Error getting source code: {e}.\n\nRunning without program aware proposer.")
                    self.program_code_string = None
                    program_aware_proposer = False
            else:
                self.program_code_string = None

            # Setup our proposer 
            proposer = GroundedProposer(
                trainset=trainset,
                prompt_model=self.prompt_model,
                program_code_string=self.program_code_string,
                view_data_batch_size=self.view_data_batch_size,
                program_aware=program_aware_proposer,
            )

            # Setup logging
            logging.basicConfig(level=logging.WARNING)
            setup_logging(self.log_dir)

            self.wandb_run_id = os.getenv("WANDB_RUN_ID", None)

            if self.wandb_run_id:
                # Initialize wandb with the same run ID
                wandb.init(
                    project="prompt_optimizers",
                    id=self.wandb_run_id,
                    resume="must",
                )

            # Setup random seeds
            random.seed(seed)
            np.random.seed(seed)

            # Log current file to log_dir
            curr_file = os.path.abspath(__file__)
            save_file_to_log_dir(curr_file, self.log_dir)

            # Set up program and evaluation function
            program = student.deepcopy()
            evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)

            # Determine the number of fewshot examples to use to generate demos for prompt
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0:
                max_bootstrapped_demos_for_candidate_gen = BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT
                max_labeled_demos_for_candidate_gen = LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT
            else:
                max_bootstrapped_demos_for_candidate_gen = max_bootstrapped_demos
                max_labeled_demos_for_candidate_gen = max_labeled_demos

            # # Generate N few shot example sets
            # if not demo_candidates and not instruction_candidates:
            #     demo_candidates = create_n_fewshot_example_sets(program=program, trainset=trainset, n=self.n, hard_fewshot=hard_fewshot, metric=self.metric, teacher_settings=self.teacher_settings, max_bootstrapped_demos=max_bootstrapped_demos_for_candidate_gen, max_labeled_demos=max_labeled_demos_for_candidate_gen)

            # Generate N few shot example sets
            try:
                demo_candidates = create_n_fewshot_demo_sets(
                    student=program,
                    num_candidate_sets=self.n,
                    trainset=trainset,
                    max_labeled_demos=max_labeled_demos_for_candidate_gen,
                    max_bootstrapped_demos=max_bootstrapped_demos_for_candidate_gen,
                    metric=self.metric,
                    teacher_settings=self.teacher_settings,
                    seed=seed,
                    metric_threshold=self.metric_threshold,
                )
            except Exception as e:
                print(f"Error generating fewshot examples: {e}")
                print("Running without fewshot examples.")
                demo_candidates = None

            # Generate N candidate prompts
            proposer.program_aware = program_aware_proposer
            proposer.use_tip = True
            proposer.use_instruct_history = False
            proposer.set_history_randomly = False
            instruction_candidates = proposer.propose_instructions_for_program(
                trainset=trainset,
                program=program,
                demo_candidates=demo_candidates,
                N=self.n,
                prompt_model=self.prompt_model,
                T=self.init_temperature,
                trial_logs={},
            )
            for i, pred in enumerate(program.predictors()):
                instruction_candidates[i][0] = get_signature(pred).instructions

            # instruction_candidates[1][0] = "Given the question, and context, respond with the number of the document that is most relevant to answering the question in the field 'Answer' (ex. Answer: '3')."

            # Save the candidate instructions generated
            if self.log_dir:
                fp = os.path.join(self.log_dir, "instructions_to_save.pickle")
                with open(fp, "wb") as file:
                    pickle.dump(instruction_candidates, file)

            # If we're doing zero-shot, reset demo_candidates to none
            if max_bootstrapped_demos == 0 and max_labeled_demos == 0:
                demo_candidates = None

            # Initialize variables to store the best program and its score
            best_score = float("-inf")
            best_program = None
            trial_logs = {}
            total_eval_calls = 0
            param_score_dict = defaultdict(list) # Dictionaries of paramater combinations we've tried, and their associated scores
            fully_evaled_param_combos = {} # List of the parameter combinations we've done full evals of

            # Define our trial objective
            def create_objective(
                baseline_program,
                instruction_candidates,
                demo_candidates,
                evaluate,
                trainset,
            ):
                def objective(trial):
                    nonlocal best_program, best_score, trial_logs, total_eval_calls  # Allow access to the outer variables
                    
                    # Kick off trial
                    logging.info(f"Starting trial num: {trial.number}")
                    trial_logs[trial.number] = {}

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
                        chosen_params.append(instruction_idx)
                        if demo_candidates:
                            demos_idx = trial.suggest_categorical(
                                f"{i}_predictor_demos", range(len(p_demo_candidates)),
                            )
                            chosen_params.append(demo_candidates)

                        # Log the selected instruction / demo candidate
                        trial_logs[trial.number][
                            f"{i}_predictor_instruction"
                        ] = instruction_idx
                        if demo_candidates:
                            trial_logs[trial.number][f"{i}_predictor_demos"] = demos_idx

                        logging.info(f"instruction_idx {instruction_idx}")
                        if demo_candidates:
                            logging.info(f"demos_idx {demos_idx}")

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
                    print("CANDIDATE PROGRAM:")
                    print_full_program(candidate_program)
                    print("...")

                    # Save the candidate program
                    trial_logs[trial.number]["program_path"] = save_candidate_program(
                        candidate_program, self.log_dir, trial.number,
                    )

                    trial_logs[trial.number]["num_eval_calls"] = 0

                    # Evaluate the candidate program with relevant batch size
                    batch_size = self._get_batch_size(minibatch, trainset)
                    score = eval_candidate_program(
                        batch_size, trainset, candidate_program, evaluate,
                    )

                    # Print out a full trace of the program in use
                    print("FULL TRACE")
                    full_trace = get_task_model_history_for_full_example(
                        candidate_program, self.task_model, trainset, evaluate,
                    )
                    print("...")

                    # Log relevant information
                    print(f"Score {score}")                
                    categorical_key = ",".join(map(str, chosen_params))
                    param_score_dict[categorical_key].append(
                        (score, candidate_program),
                    )
                    trial_logs[trial.number]["num_eval_calls"] = batch_size
                    trial_logs[trial.number]["full_eval"] = batch_size >= len(trainset)
                    trial_logs[trial.number]["eval_example_call"] = full_trace
                    trial_logs[trial.number]["score"] = score
                    trial_logs[trial.number]["pruned"] = False
                    total_eval_calls += trial_logs[trial.number]["num_eval_calls"]
                    trial_logs[trial.number]["total_eval_calls_so_far"] = total_eval_calls
                    trial_logs[trial.number]["program"] = candidate_program.deepcopy()
                    if self.wandb_run_id:
                        wandb.log(
                            {
                                "score": score,
                                "num_eval_calls": trial_logs[trial.number]["num_eval_calls"],
                                "total_eval_calls": total_eval_calls,
                            },
                        )

                    # Update the best program if the current score is better, and if we're not using minibatching
                    best_score_updated = False
                    if score > best_score and trial_logs[trial.number]["full_eval"] and not minibatch:
                        print("Updating best score")
                        best_score = score
                        best_program = candidate_program.deepcopy()
                        best_score_updated = True
                        

                    # If we're doing minibatching, check to see if it's time to do a full eval
                    if minibatch and trial.number % self.minibatch_full_eval_steps == 0:
                        
                        # Save old information as the minibatch version
                        trial_logs[trial.number]["mb_score"] = score
                        trial_logs[trial.number]["mb_program_path"] = trial_logs[trial.number]["program_path"]

                        # Identify our best program (based on mean of scores so far, and do a full eval on it)
                        highest_mean_program, combo_key = get_program_with_highest_avg_score(param_score_dict, fully_evaled_param_combos)
                        full_train_score = eval_candidate_program(
                            len(trainset), trainset, highest_mean_program, evaluate,
                        )

                        # Log relevant information
                        fully_evaled_param_combos[combo_key] = {"program":highest_mean_program, "score": full_train_score}
                        total_eval_calls += len(trainset)
                        trial_logs[trial.number]["total_eval_calls_so_far"] = total_eval_calls
                        trial_logs[trial.number]["full_eval"] = True
                        trial_logs[trial.number]["program_path"] = save_candidate_program(
                            program=highest_mean_program, log_dir=self.log_dir, trial_num=trial.number, note="full_eval",
                        )
                        trial_logs[trial.number]["score"] = full_train_score
                        
                        if full_train_score > best_score:
                            print(f"UPDATING BEST SCORE WITH {full_train_score}")
                            best_score = full_train_score
                            best_program = highest_mean_program.deepcopy()
                            best_score_updated = True
                    
                    # If the best score was updated, do a full eval on the dev set
                    if best_score_updated:
                        full_dev_score = evaluate(
                            best_program,
                            devset=valset,
                            display_table=0,
                        )
                        if self.wandb_run_id:
                            wandb.log(
                                {
                                    "best_prog_so_far_train_score": best_score,
                                    "best_prog_so_far_dev_score": full_dev_score,
                                },
                            )

                    return score

                return objective

            # Run the trial
            objective_function = create_objective(
                program, instruction_candidates, demo_candidates, evaluate, trainset,
            )

            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            score = study.optimize(objective_function, n_trials=num_batches)

            if best_program is not None and self.track_stats:
                best_program.trial_logs = trial_logs
                best_program.score = best_score
                best_program.prompt_model_total_calls = self.prompt_model_total_calls
                best_program.total_calls = self.total_calls

            # program_file_path = os.path.join(self.log_dir, 'best_program.pickle')
            if self.log_dir:
                program_file_path = os.path.join(self.log_dir, "best_program")
                best_program.save(program_file_path)

                optuna_study_file_path = os.path.join(self.log_dir, "optuna_study.pickle")
                with open(optuna_study_file_path, "wb") as file:
                    pickle.dump(study, file)

            return best_program
        return student
