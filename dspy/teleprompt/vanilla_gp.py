import logging
import random
import textwrap
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import select
import sys
import time
import re
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
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

logger = logging.getLogger(__name__)

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 2000},
}

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


# class BertCLSEmbedder:
#     def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.eval()
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     @torch.no_grad()
#     def encode(self, text: str, max_length: int = 512, l2_normalize: bool = True):
#         # truncate to BERT's max length
#         toks = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=False,
#             max_length=max_length,
#             padding=False
#         ).to(self.device)

#         # last_hidden_state: [batch, seq_len, hidden]
#         out = self.model(**toks)
#         cls = out.last_hidden_state[:, 0, :]          # [CLS] 위치 벡터 (shape: [1, hidden_size])
#         vec = cls.squeeze(0)

#         if l2_normalize:
#             vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
#         import ipdb; ipdb.set_trace()
#         return vec.cpu().tolist()                      # list[hidden_size]

class VanillaGP(Teleprompter):
    def __init__(
        self,
        metric: Callable,
        prompt_pool: Optional[Dict] = None,
        prompt_model: Optional[Any] = None,
        task_model: Optional[Any] = None,
        teacher_settings: Dict = {},
        max_bootstrapped_demos: int = 5, # 4
        max_labeled_demos: int = 5, # 4
        auto: Optional[Literal["light", "medium", "heavy"]] = "light",
        num_candidates: Optional[int] = None,
        num_threads: Optional[int] = None,
        max_errors: int = 100,
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
            raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")
        self.auto = auto
        self.num_fewshot_candidates = num_candidates
        self.num_instruct_candidates = num_candidates
        self.num_candidates = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.task_model = task_model if task_model else dspy.settings.lm
        self.prompt_pool = prompt_pool if prompt_pool else {}
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
        teacher: Any = None,
        valset: Optional[List] = None,
        num_trials: Optional[int] = None,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        seed: Optional[int] = None,
        minibatch: bool = False, #True
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool = True,
        provide_traceback: Optional[bool] = None,
    ) -> Any:
        
        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (self.max_labeled_demos == 0)
        
        # If auto is None, and num_trials is not provided (but num_candidates is), raise an error that suggests a good num_trials value
        if self.auto is None and (self.num_candidates is not None and num_trials is None):
            raise ValueError(f"If auto is None, num_trials must also be provided. Given num_candidates={self.num_candidates}, we'd recommend setting num_trials to ~{self._set_num_trials_from_num_candidates(student, zeroshot_opt, self.num_candidates)}.")
        
        # If auto is None, and num_candidates or num_trials is None, raise an error
        if self.auto is None and (self.num_candidates is None or num_trials is None):
            raise ValueError("If auto is None, num_candidates must also be provided.")
        
        # If auto is provided, and either num_candidates or num_trials is not None, raise an error
        if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
            raise ValueError("If auto is not None, num_candidates and num_trials cannot be set, since they would be overrided by the auto settings. Please either set auto to None, or do not specify num_candidates and num_trials.")
        
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
        # num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
        #     student, num_trials, minibatch, zeroshot_opt, valset
        # )
        minibatch = False # for reproducing HbBoPs experiments
        num_trials = 25

        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")

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
                logger.info("Compilation aborted by the user.")
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
            provide_traceback=provide_traceback,
        )

        # Step 1: Bootstrap few-shot examples
        # demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher)
        if self.prompt_pool and "demo_candidates" in self.prompt_pool:
            demo_candidates = self.prompt_pool["demo_candidates"]
        else:
            demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher)

        # Step 2: Propose instruction candidates
        # instruction_candidates = self._propose_instructions(
        #     program,
        #     trainset,
        #     demo_candidates,
        #     view_data_batch_size,
        #     program_aware_proposer,
        #     data_aware_proposer,
        #     tip_aware_proposer,
        #     fewshot_aware_proposer,
        # )
        if self.prompt_pool and "instruction_candidates" in self.prompt_pool:
            instruction_candidates = self.prompt_pool["instruction_candidates"]
        else:
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

        # import ipdb; ipdb.set_trace()

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
        valset: List,
    ) -> Tuple[int, List, bool]:
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

    def _set_and_validate_datasets(self, trainset: List, valset: Optional[List]):
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

    def _print_auto_run_settings(self, num_trials: int, minibatch: bool, valset: List):
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
        valset: List,
        program_aware_proposer: bool,
    ) -> Tuple[str, str]:
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
            If no input is received within 20 seconds, the program will proceed automatically.

            If you would like to bypass this confirmation step in future executions, set the {YELLOW}`requires_permission_to_run`{ENDC} flag to {YELLOW}`False`{ENDC} when calling compile.

            {YELLOW}Awaiting your input...{ENDC}
        """
        )

        print(f"{user_message}\n{user_confirmation_message}\nDo you wish to continue? (y/n): ", end='', flush=True)
        
        # Wait for input with timeout
        start_time = time.time()
        while time.time() - start_time < 20:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.readline().strip().lower()
                return user_input == "y"
            time.sleep(0.1)
        
        print("\nNo input received within 20 seconds. Proceeding with execution...")
        return True

    def _bootstrap_fewshot_examples(self, program: Any, trainset: List, seed: int, teacher: Any) -> Optional[List]:
        logger.info("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            logger.info(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            logger.info("These will be used for informing instruction proposal.\n")

        logger.info(f"Bootstrapping N={self.num_fewshot_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0

        try:
            demo_candidates = create_n_fewshot_demo_sets(
                student=program,
                num_candidate_sets=self.num_fewshot_candidates,
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
            )
        except Exception as e:
            logger.info(f"Error generating few-shot examples: {e}")
            logger.info("Running without few-shot examples.")
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
        )

        logger.info(f"\nProposing N={self.num_instruct_candidates} instructions...\n")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_instruct_candidates,
            T=self.init_temperature,
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
        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        logger.info(
            "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
        )
        # import ipdb; ipdb.set_trace()

        # Compute the adjusted total trials that we will run (including full evals)
        # run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
        # adjusted_num_trials = int((num_trials + num_trials // minibatch_full_eval_steps + 1 + run_additional_full_eval_at_end) if minibatch else num_trials)
        # logger.info(f"== Trial {1} / {adjusted_num_trials} - Full Evaluation of Default Program ==")

        # default_score, _ = eval_candidate_program(
        #     len(valset), valset, program, evaluate, self.rng, return_all_scores=True
        # )
        # logger.info(f"Default program score: {default_score}\n")

        trial_logs = {}
        # trial_logs[1] = {}
        # trial_logs[1]["full_eval_program_path"] = save_candidate_program(program, self.log_dir, -1)
        # trial_logs[1]["full_eval_score"] = default_score
        # trial_logs[1]["total_eval_calls_so_far"] = len(valset)
        # trial_logs[1]["full_eval_program"] = program.deepcopy()

        # 0) 후보 전수 조합 만들기 / i =0인 경우만 고려한 상태
        prompt_candidate_indicies = [] # i =0인 경우만 고려한 상태
        for i, predictor in enumerate(program.predictors()):
            for j in range(len(instruction_candidates[i])):
                for k in range(len(demo_candidates[i]) if demo_candidates else 1):
                    combo = {f"{i}_predictor_instruction": j}
                    if demo_candidates:
                        combo[f"{i}_predictor_demos"] = k
                        combo["key"] = i*len(instruction_candidates[i])*len(demo_candidates[i]) + j*len(demo_candidates[i]) + k
                    prompt_candidate_indicies.append(combo)
        # combo 예시: [{"0_instr": i, "0_demo": j, "1_instr": k, ...}, ...]'

        # # 1) 조합 → 프롬프트 텍스트 직렬화
        def render_prompt(program) -> str:
            # program deepcopy → 각 predictor에 combo의 instruction/demos 주입 (평가 때 쓰는 로직을 재사용)
            from gepa_artifact.utils.dspy.dspy.adapters.chat_adapter import ChatAdapter
            adapter = ChatAdapter()
            prompt = adapter.format(program.predict.signature, program.predict.demos, {})
            # prompt_text = [
            #         {'role': 'system', 'content': 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `reasoning` (str): \n2. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        # Solve the math word problem. Show your step-by-step reasoning with calculations. Your final answer should be a single number, preceded by ####.'},
            #         {'role': 'user', 'content': '[[ ## question ## ]]\n' + val_set[i]['question']}, 
            #         {'role': 'assistant', 'content': '[[ ## reasoning ## ]]\n' + val_set[i]['reasoning'] + '\n\n[[ ## answer ## ]]\n' + val_set[i]['answer']}, 
            #         {'role': 'user', 'content': '[[ ## question ## ]]\n' + val_set[i+1]['question']}, 
            #         {'role': 'assistant', 'content': '[[ ## reasoning ## ]]\n' + val_set[i+1]['reasoning'] + '\n\n[[ ## answer ## ]]\n' + val_set[i+1]['answer']}, 
            #         {'role': 'user', 'content': '[[ ## question ## ]]\n' + val_set[i+2]['question']}, 
            #         {'role': 'assistant', 'content': '[[ ## reasoning ## ]]\n' + val_set[i+1]['reasoning'] + '\n\n[[ ## answer ## ]]\n' + val_set[i+1]['answer']}, 
            #         {'role': 'user', 'content': '[[ ## question ## ]]\n' + val_set[i+3]['question']}, 
            #         {'role': 'assistant', 'content': '[[ ## reasoning ## ]]\n' + val_set[i+1]['reasoning'] + '\n\n[[ ## answer ## ]]\n' + val_set[i+1]['answer']}, 
            #         {'role': 'user', 'content': '[[ ## question ## ]]\n' + val_set[i+4]['question']}, 
            #         {'role': 'assistant', 'content': "[[ ## reasoning ## ]]\n" + val_set[i+1]['reasoning'] + '\n\n[[ ## answer ## ]]\n' + val_set[i+1]['answer']}, 
            #         {'role': 'user', 'content': "Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."}
            #     ]

            def normalize_whitespace(text: str) -> str:
                """
                텍스트를 임베딩/캐시용으로 '결정적(deterministic)' 문자열로 정규화.
                - 개행을 \n 으로 통일
                - non-breaking space, 탭 → 보통 공백으로
                - 각 줄의 앞뒤 공백 제거
                - 줄 내부의 반복 공백을 1칸으로 축소
                - 연속 빈 줄은 1개로 축소
                - 파일 전체 앞뒤 공백 제거
                """
                if text is None:
                    return ""

                # 1) 개행 통일
                s = text.replace("\r\n", "\n").replace("\r", "\n")

                # 2) 특수 공백을 일반 공백으로
                s = s.replace("\xa0", " ").replace("\u200b", "")  # NBSP, zero-width space 제거
                s = s.expandtabs(1)  # 탭 → 1칸 공백 (필요하면 4로 바꿔도 OK)

                # 3) 줄 단위 트림 + 줄 내부 다중 공백 축소
                lines = []
                for line in s.split("\n"):
                    # 앞뒤 공백 제거
                    line = line.strip()
                    # 줄 내부의 여러 공백 → 1칸
                    line = re.sub(r"[ \t]+", " ", line)
                    lines.append(line)

                s = "\n".join(lines)

                # 4) 연속 빈 줄 1개로 축소
                s = re.sub(r"\n{3,}", "\n\n", s)

                # 5) 전체 트림
                return s.strip()
            
            parts = []
            for message in prompt:
                role = message['role'].strip()
                content = normalize_whitespace(message['content'])
                parts.append(f"<|{role}|>\n{content}")
            # 3) 안정적 구분자 추가 (개행 2줄)
            return "\n\n<|sep|>\n\n".join(parts)
        
        def embed_prompt_text(
            text: str,
            model: SentenceTransformer, # BertCLSEmbedder
            l2_normalize: bool = True,
        ):
            """
            text: "<|system|>\nYour input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be..."
            return: 1D list (embedding vector)
            """

            # Sentence-Transformers는 바로 문자열 입력 가능
            vec = model.encode(text, normalize_embeddings=l2_normalize)
            # vec = vec.tolist()  # numpy -> python list
            return vec
        
        # # 2) 임베딩 전수 캐시
        embedded_vectors = {}  # dict: key(combo) -> embedding vector
        embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # BertCLSEmbedder(model_name="bert-base-uncased")
        for prompt_indicies in prompt_candidate_indicies:
            program = program.deepcopy()
            self._insert_instructions_and_demos(program, instruction_candidates, demo_candidates, prompt_indicies)
            flattened_prompt_text = render_prompt(program)
            embedded_vectors[prompt_indicies['key']] = embed_prompt_text(flattened_prompt_text, model=embedder) # 사전 학습 임베더 사용
        import ipdb; ipdb.set_trace()

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
            score = eval_candidate_program(batch_size, valset, candidate_program, evaluate, self.rng)
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
            if minibatch and ((trial_num % (minibatch_full_eval_steps+1) == 0) or (trial_num == (adjusted_num_trials-1))):
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
            # import ipdb; ipdb.set_trace()  

            return score

        sampler = optuna.samplers.GPSampler(seed=seed)
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

        # print(trial_logs)
        # import ipdb; ipdb.set_trace()

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

    def _insert_instructions_and_demos(
        self,
        candidate_program: Any,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[List],
        prompt_indices,
    ):
        for i, predictor in enumerate(candidate_program.predictors()): # insert instruction and demos on given candidate_program
            selected_instruction = instruction_candidates[i][prompt_indices[f"{i}_predictor_instruction"]]
            updated_signature = get_signature(predictor).with_instructions(selected_instruction)
            set_signature(predictor, updated_signature)
            # Select demos if available
            if demo_candidates:
                predictor.demos = demo_candidates[i][prompt_indices[f"{i}_predictor_demos"]]

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
        param_score_dict: Dict,
        fully_evaled_param_combos: Dict,
        evaluate: Evaluate,
        valset: List,
        trial_logs: Dict,
        total_eval_calls: int,
        score_data,
        best_score: float,
        best_program: Any,
        study: optuna.Study,
        instruction_candidates: List,
        demo_candidates: List,
    ):
        logger.info(f"===== Trial {trial_num + 1} / {adjusted_num_trials} - Full Evaluation =====")

        # Identify best program to evaluate fully
        highest_mean_program, mean_score, combo_key, params = get_program_with_highest_avg_score(
            param_score_dict, fully_evaled_param_combos
        )
        logger.info(f"Doing full eval on next top averaging program (Avg Score: {mean_score}) from minibatch trials...")
        full_eval_score = eval_candidate_program(len(valset), valset, highest_mean_program, evaluate, self.rng)
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
