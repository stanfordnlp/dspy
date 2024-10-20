import random
from typing import List, Optional, Dict
import torch
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.vanilla import LabeledFewShot
from dspy.utils.pez_utils import optimize_prompt


class BootstrapFewShotWithPEZ(Teleprompter):
    def __init__(
            self,
            metric,
            teacher_settings: Optional[Dict] = None,
            max_bootstrapped_demos: int = 4,
            max_labeled_demos: int = 16,
            max_rounds: int = 1,
            num_candidate_programs: int = 16,
            num_threads: int = 6,
            max_errors: int = 10,
            stop_at_score: Optional[float] = None,
            metric_threshold: Optional[float] = None,
            prompt_len: int = 5,
            opt_iters: int = 500,
            lr: float = 5e-5,
            weight_decay: float = 1e-4,
            print_step: int = 50,
            loss_weight: float = 1.0,
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds
        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_errors = max_errors
        self.num_candidate_programs = num_candidate_programs

        # PEZ-specific parameters for prompt optimization
        self.prompt_len = prompt_len
        self.opt_iters = opt_iters
        self.lr = lr
        self.weight_decay = weight_decay
        self.print_step = print_step
        self.loss_weight = loss_weight

        print(f"Will attempt to bootstrap {self.num_candidate_programs} candidate sets.")

    def compile(
            self,
            student,
            *,
            teacher=None,
            trainset,
            valset=None,
            restrict=None,
            labeled_sample=True
    ) -> Teleprompter:
        """
        Compile the student program by optimizing bootstrapped few-shot examples using PEZ.

        Parameters:
        - student: The student model to be trained.
        - teacher: The teacher model providing the few-shot examples.
        - trainset: The training set to bootstrap few-shot examples from.
        - valset: Optional validation set.
        - restrict: Optionally restrict the number of programs to run.
        - labeled_sample: Whether to use labeled sampling for few-shot examples.

        Returns:
        - The fine-tuned student program with optimized prompts.
        """
        self.trainset = trainset
        self.valset = valset or trainset

        scores = []
        all_subscores = []
        score_data = []

        for seed in range(-3, self.num_candidate_programs):
            if restrict and seed not in restrict:
                continue

            trainset_copy = list(self.trainset)

            if seed == -3:
                # zero-shot
                program = student.reset_copy()

            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

            else:
                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(1, self.max_bootstrapped_demos)

                # Bootstrap few-shot examples using the teacher
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                )
                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

                # Optimize bootstrapped examples using PEZ
                bootstrapped_prompts = self._get_bootstrapped_prompts(program)
                program = self._optimize_with_pez(program, bootstrapped_prompts)

            # Evaluate the program with the optimized prompts
            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                display_table=False,
                display_progress=True,
            )

            score, subscores = evaluate(program, return_all_scores=True)
            all_subscores.append(subscores)

            # Update the best program based on scores
            if len(scores) == 0 or score > max(scores):
                print(f"New best score: {score} for seed {seed}")
                best_program = program

            scores.append(score)
            score_data.append((score, subscores, seed, program))

            if self.stop_at_score and score >= self.stop_at_score:
                print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # Attach candidate programs to the best program
        best_program.candidate_programs = sorted(score_data, key=lambda x: x[0], reverse=True)
        return best_program

    def _get_bootstrapped_prompts(self, program) -> List[str]:
        """
        Extract the bootstrapped prompts from the program.

        Returns:
        - List of bootstrapped prompts.
        """
        # Assuming `program` has an attribute `demos` containing the bootstrapped examples.
        bootstrapped_prompts = []

        for name, predictor in program.named_predictors():
            for demo in predictor.demos:
                prompt = demo['prompt']  # Assuming 'prompt' key exists in demo dictionary
                bootstrapped_prompts.append(prompt)

        return bootstrapped_prompts

    def _optimize_with_pez(self, program, bootstrapped_prompts: List[str]) -> Teleprompter:
        """
        Use PEZ to optimize the bootstrapped prompts.

        Parameters:
        - program: The student program.
        - bootstrapped_prompts: List of prompts to be optimized.

        Returns:
        - The program with optimized prompts.
        """
        prompt_args = {
            "prompt_len": self.prompt_len,
            "iter": self.opt_iters,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "print_step": self.print_step,
            "loss_weight": self.loss_weight,
            "prompt_bs": 1  # Batch size for optimizing prompts
        }

        # Optimize the bootstrapped prompts using PEZ
        optimized_prompts = optimize_prompt(
            model=program,
            preprocess=None,
            args=prompt_args,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            target_prompts=bootstrapped_prompts
        )

        return optimized_prompts  # Return the program with optimized prompts
