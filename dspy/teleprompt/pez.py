from typing import Dict, Optional, List
from dspy.utils.pez_utils import *
from dspy.teleprompt import LabeledFewShot
from datasets import load_dataset
from dspy.teleprompt.bootstrap import BootstrapFewShot


class PEZFewshot(BootstrapFewShot):
    def __init__(
        self,
        metric=None,
        metric_threshold=None,
        teacher_settings: Optional[Dict] = None,
        prompt_len=5,
        _iter=500,
        lr=5e-5,
        weight_decay=1e-4,
        print_step=50,
        loss_weight=1.0,
        glue_task: str = "sst2",
        num_prompts: int = 5
    ):
        super().__init__(
            metric=metric,
            metric_threshold=metric_threshold,
            teacher_settings=teacher_settings
        )

        # PEZ-specific attributes
        self.prompt_len = prompt_len
        self.iter = _iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.print_step = print_step
        self.loss_weight = loss_weight
        self.glue_task = glue_task
        self.num_prompts = num_prompts

        self.trainset = None
        self.student = None
        self.teacher = None

    def    compile(self, student, *, teacher=None, trainset):
        self.trainset = trainset

        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher else student.reset_copy()

        assert not getattr(self.student, "_compiled", False), "Student must be uncompiled."

        if self.max_labeled_demos:
            labeled_fewshot = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = labeled_fewshot.compile(self.teacher.reset_copy(), trainset=self.trainset)

        self._prepare_predictor_mappings()

        # Optimize prompts using PEZ
        prompt_args = {
            "prompt_len": self.prompt_len,
            "iter": self.iter,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "print_step": self.print_step,
            "loss_weight": self.loss_weight,
            "prompt_bs": 1  # Batch size of prompts to optimize
        }

        # Use the existing labeled prompts from a GLUE dataset
        self.student._optimized_prompt = optimize_prompt(
            model=self.student,
            preprocess=None,  # assuming text-based, not images
            args=prompt_args,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            target_prompts=self._get_few_shot_prompts(),
        )

        self.student._compiled = True
        return self.student

    def _get_few_shot_prompts(self) -> List[str]:
        """
        Fetch few-shot examples from a GLUE task.

        Returns
        -------
        List[str]
            A list of formatted few-shot prompts based on the chosen GLUE task.
        """
        dataset = load_dataset("glue", self.glue_task)
        prompts = []

        # Extract examples from the training set (or validation if needed)
        for i in range(self.num_prompts):
            example = dataset["train"][i]

            # Format the prompts depending on the GLUE task
            if self.glue_task == "sst2":
                prompts.append(example["sentence"])  # SST-2 has single sentence
            elif self.glue_task in ["mrpc", "qqp"]:
                # MRPC and QQP are sentence-pair tasks
                prompts.append(f"Sentence 1: {example['sentence1']}, Sentence 2: {example['sentence2']}")
            elif self.glue_task == "cola":
                # CoLA task has a single sentence and a grammaticality judgment label
                prompts.append(f"Sentence: {example['sentence']}, Label: {example['label']}")
            # TODO: Add additional GLUE task handling as needed
            else:
                # For tasks with unknown structure, just use the raw text
                prompts.append(str(example))

        return prompts
