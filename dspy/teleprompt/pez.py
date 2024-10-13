class PEZFewshot(Teleprompter):
    def __init__(
        self,
        metric=None,
        metric_threshold=None,
        teacher_settings: Optional[Dict] = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        max_errors=5,
        prompt_len=5,
        iter=500,
        lr=5e-5,
        weight_decay=1e-4,
        print_step=50,
        loss_weight=1.0,
    ):
        """
        A Teleprompter class that composes a set of demos/examples into an optimized hard prompt
        using PEZ.

        Parameters
        ----------
        metric: Callable
            A function that compares an expected value and predicted value, outputting the result of that comparison.
        metric_threshold: optional float, default `None`
            Threshold to accept or reject bootstrapped demos.
        teacher_settings: dict, optional
            Settings for the `teacher` model.
        max_bootstrapped_demos: int, default 4
            Maximum number of bootstrapped demos.
        max_labeled_demos: int, default 16
            Maximum number of labeled demos to include.
        prompt_len: int, default 5
            The number of tokens in the prompt.
        iter: int, default 500
            The number of optimization iterations for PEZ.
        lr: float, default 5e-5
            Learning rate for PEZ optimizer.
        weight_decay: float, default 1e-4
            Weight decay for the optimizer.
        print_step: int, default 50
            Print step interval for logging optimization progress.
        loss_weight: float, default 1.0
            Weight for the loss term in optimization.
        """
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_settings = teacher_settings or {}

        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.prompt_len = prompt_len
        self.iter = iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.print_step = print_step
        self.loss_weight = loss_weight

    def compile(self, student, *, teacher=None, trainset):
        self.trainset = trainset

        # Prepare student and teacher
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher else student.reset_copy()

        assert not getattr(self.student, "_compiled", False), "Student must be uncompiled."

        if self.max_labeled_demos:
            labeled_fewshot = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = labeled_fewshot.compile(self.teacher.reset_copy(), trainset=self.trainset)

        self._prepare_predictor_mappings()
        self._bootstrap()

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

        # TODO: contact
        self.student._optimized_prompt = optimize_prompt(
            model=self.student,
            preprocess=None,  # assuming text-based, not images
            args=prompt_args,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            target_prompts=self._get_few_shot_prompts(),
        )

        self.student._compiled = True
        return self.student

    def _get_few_shot_prompts(self):
        # Generate or get the few-shot prompts from teacher/student
        return ["Sample prompt 1", "Sample prompt 2"]  # Replace with actual prompt generation logic

    # Additional helper methods like _prepare_predictor_mappings(), _bootstrap(), etc.
