import inspect
import logging
import random
from typing import Any, Callable

import dspy
from dspy.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.bootstrap_finetune import (
    BootstrapFinetune,
    all_predictors_have_lms,
    kill_lms,
    launch_lms,
    prepare_student,
    prepare_teacher,
)
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import eval_candidate_program

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"


class BetterTogether(Teleprompter):
    """A meta-optimizer that combines prompt and weight optimization in configurable sequences.

    BetterTogether is a meta-optimizer proposed in the paper
    [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930).
    It combines prompt optimization and weight optimization (fine-tuning) by applying them in a
    configurable sequence, allowing a student program to iteratively improve both its prompts and
    model parameters.

    The core insight is that prompt and weight optimization can complement each other: prompt
    optimization can potentially discover effective task decompositions and reasoning strategies,
    while weight optimization can specialize the model to execute these patterns more efficiently.
    Using these approaches together in sequences (e.g., prompt optimization then weight optimization)
    may allow each to build on the improvements made by the other. Empirically, this approach often
    outperforms either strategy alone, even with state-of-the-art optimizers. For example, a
    [Databricks case study](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization)
    shows that combining BetterTogether with GEPA and fine-tuning outperforms either approach
    alone.

    The optimizer is initialized with a metric and custom optimizers. For example, you can combine
    GEPA for prompt optimization with BootstrapFinetune for weight optimization:
    ``BetterTogether(metric=metric, p=GEPA(...), w=BootstrapFinetune(...))``. The ``compile()``
    method takes a student program, trainset, and strategy string where strategy keys correspond
    to the optimizer names from initialization. It executes each optimizer in the specified sequence.
    When a validation set is provided, the best performing program is returned; otherwise, the
    latest program is returned. Note: Weight optimizers like BootstrapFinetune require student
    programs to have LMs explicitly set (not relying on global dspy.settings.lm), and BetterTogether
    mirrors this requirement for simplicity. Therefore we call ``set_lm`` before compiling.

        >>> from dspy.teleprompt import GEPA, BootstrapFinetune
        >>>
        >>> # Combine GEPA for prompt optimization with BootstrapFinetune for weight optimization
        >>> optimizer = BetterTogether(
        ...     metric=metric,
        ...     p=GEPA(metric=metric, auto="medium"),
        ...     w=BootstrapFinetune(metric=metric)
        ... )
        >>>
        >>> student.set_lm(lm)
        >>> compiled = optimizer.compile(
        ...     student,
        ...     trainset=trainset,
        ...     valset=valset,
        ...     strategy="p -> w"
        ... )

    You can pass optimizer-specific arguments to each optimizer's ``compile()`` method using
    ``optimizer_compile_args``. This allows you to customize each optimizer's behavior:

        >>> from dspy.teleprompt import MIPROv2
        >>>
        >>> # Use MIPROv2 for prompt optimization with custom parameters
        >>> optimizer = BetterTogether(
        ...     metric=metric,
        ...     p=MIPROv2(metric=metric),
        ...     w=BootstrapFinetune(metric=metric)
        ... )
        >>>
        >>> student.set_lm(lm)
        >>> compiled = optimizer.compile(
        ...     student,
        ...     trainset=trainset,
        ...     valset=valset,
        ...     strategy="p -> w",
        ...     optimizer_compile_args={
        ...         "p": {"num_trials": 10, "max_bootstrapped_demos": 8},  # Configure MIPROv2's compile arguments
        ...     }
        ... )

    Since BetterTogether is a meta-optimizer that can run arbitrary optimizers in sequence, any
    sequence of optimizers can be combined together. The optimizer names used in the strategy string
    correspond to the keyword arguments specified in the constructor. For example, different prompt
    optimizers can be alternated multiple times (though note this is just an illustration of
    BetterTogether's flexibility, not a recommended configuration):

        >>> from dspy.teleprompt import MIPROv2, GEPA
        >>>
        >>> # Chain two optimizers three times: MIPROv2 -> GEPA -> MIPROv2
        >>> optimizer = BetterTogether(
        ...     metric=metric,
        ...     mipro=MIPROv2(metric=metric, auto="light"),
        ...     gepa=GEPA(metric=metric, auto="light")
        ... )
        >>>
        >>> student.set_lm(lm)
        >>> compiled = optimizer.compile(
        ...     student,
        ...     trainset=trainset,
        ...     valset=valset,
        ...     strategy="mipro -> gepa -> mipro"
        ... )

    Note:
        Output Attributes: The returned program includes two additional attributes:
        ``candidate_programs`` and ``flag_compilation_error_occurred``. The ``candidate_programs``
        attribute is a list of dicts, each containing 'program', 'score', and 'strategy' (e.g.,
        '', 'p', 'p -> w', 'p -> w -> p'), sorted by descending score (similar to
        ``dspy.MIPROv2.candidate_programs``). If any optimizer step fails,
        ``flag_compilation_error_occurred`` is set to True and the best program found so far is
        returned.

        Model Lifecycle Management: BetterTogether automatically manages language model lifecycle
        (launching, killing, and relaunching after fine-tuning), which are no-ops for most
        API-based LMs. This is particularly important when using weight optimizers like
        BootstrapFinetune with local providers (e.g., dspy.LocalProvider), as it handles model
        initialization and cleanup between optimization steps.
    """

    STRAT_SEP = " -> "

    def __init__(
        self,
        metric: Callable,
        **optimizers: Teleprompter,
    ):
        """Initialize BetterTogether with a metric and custom optimizers.

        Args:
            metric: Evaluation metric function for scoring programs. Should accept
                ``(example, prediction, trace=None)`` and return a numeric score (higher is better).
                This metric is used to evaluate candidate programs during optimization and is passed
                to the default optimizers if no custom optimizers are provided.
            **optimizers: Custom optimizers as keyword arguments, where keys become the optimizer
                names used in the strategy string. For example, ``p=GEPA(...), w=BootstrapFinetune(...)``
                makes 'p' and 'w' available for use in strategies like ``"p -> w"``. If not provided,
                defaults to ``p=BootstrapFewShotWithRandomSearch(metric=metric)`` and
                ``w=BootstrapFinetune(metric=metric)``. Any DSPy Teleprompter can be used.

        Example:
            >>> # Use custom optimizers
            >>> from dspy.teleprompt import GEPA, BootstrapFinetune
            >>> optimizer = BetterTogether(
            ...     metric=metric,
            ...     p=GEPA(metric=metric, auto="medium"),
            ...     w=BootstrapFinetune(metric=metric)
            ... )
            >>>
            >>> # Use default optimizers
            >>> optimizer = BetterTogether(metric=metric)
        """
        self.metric = metric

        if not optimizers:
            logger.info(
                "No optimizers provided. Using defaults: "
                "BootstrapFewShotWithRandomSearch (p) and BootstrapFinetune (w). "
                "You can use the letters p and w to specify the compile strategy. "
                "For example, to run weight optimization after prompt optimization, use strategy='p -> w'."
            )
            optimizers = {
                "p": BootstrapFewShotWithRandomSearch(metric=metric),
                "w": BootstrapFinetune(metric=metric),
            }
        for key, optimizer in optimizers.items():
            if not isinstance(optimizer, Teleprompter):
                raise TypeError(
                    f"Optimizer '{key}' must be a Teleprompter, "
                    f"got {type(optimizer).__name__}"
                )
        self.optimizers: dict[str, Teleprompter] = optimizers

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | list[Module] | None = None,
        valset: list[Example] | None = None,
        # often specified in init in other optimizers
        num_threads: int | None = None,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        seed: int | None = None,
        # specific to BetterTogether
        valset_ratio: float = 0.1,
        shuffle_trainset_between_steps: bool = True,
        strategy: str = "p -> w -> p",
        optimizer_compile_args: dict[str, dict[str, Any]] | None = None,
    ) -> Module:
        """Compile and optimize a student program using a sequence of optimization strategies.

        Executes the optimizers specified in the strategy string sequentially, evaluating each
        intermediate result and returning the best performing program.

        Args:
            student: DSPy program to optimize. All predictors must have language models assigned.
                program.set_lm(lm) can be used to assign a language model to all modules of a 
                program.
            trainset: Training examples for optimization. Each optimizer receives the full trainset
                (or a shuffled version if ``shuffle_trainset_between_steps=True``).
            teacher: Optional teacher module(s) for bootstrapping. Can be a single module or list.
                Passed to optimizers.
            valset: Validation set for evaluating optimization steps. If not provided, a portion of
                trainset is held out (controlled by ``valset_ratio``). If both ``valset`` and
                ``valset_ratio`` are None/0, no validation occurs and the latest program is returned.
            num_threads: Number of parallel evaluation threads. Default is None, which means sequential evaluation.
            max_errors: Maximum errors to tolerate during evaluation. Defaults to
                ``dspy.settings.max_errors``.
            provide_traceback: Whether to show detailed tracebacks for evaluation errors.
            seed: Random seed for reproducibility. Controls trainset shuffling and evaluation sampling.
            valset_ratio: Fraction of trainset to hold out as validation (range [0, 1)). For example,
                0.1 holds out 10%. Set to 0 to skip validation. Default is 0.1.
            shuffle_trainset_between_steps: Whether to shuffle trainset before each optimization step.
                Helps prevent overfitting to example ordering. Default is True.
            strategy: Sequence of optimizers to apply, separated by ``" -> "``. Each element must be
                a key from the optimizers provided in ``__init__``. For example, ``"p -> w -> p"``
                applies prompt optimization, then weight optimization, then prompt optimization again.
                Default is ``"p -> w -> p"``.
            optimizer_compile_args: Optional dict mapping optimizer keys to their ``compile()``
                arguments. If trainset, valset, or teacher are provided in the dict for a specific
                optimizer, they override the defaults from BetterTogether's compile method. For example:
                ``{"p": {"num_trials": 10}, "w": {"trainset": custom_trainset}}``. This is useful to
                override the default compile arguments for specific optimizers. The ``student`` argument
                cannot be included in optimizer_compile_args; BetterTogether's compile method manages
                the student reference for all optimizers.

        Returns:
            Optimized student program with two additional attributes:

            - ``candidate_programs``: List of dicts with 'program', 'score', and 'strategy' keys,
              sorted by score (best first). Contains all evaluated programs including the baseline.
            - ``flag_compilation_error_occurred``: Boolean indicating if any optimization step failed.

        Raises:
            ValueError: If trainset is empty, valset_ratio not in [0, 1), strategy is empty or
                contains invalid optimizer keys, or optimizer_compile_args contains invalid arguments.
            TypeError: If optimizer_compile_args contains a 'student' key (not allowed).

        Example:
            >>> optimizer = BetterTogether(
            ...     metric=metric,
            ...     p=GEPA(metric=metric),
            ...     w=BootstrapFinetune(metric=metric)
            ... )
            >>> student.set_lm(lm)
            >>> compiled = optimizer.compile(
            ...     student,
            ...     trainset=trainset,
            ...     valset=valset,
            ...     strategy="p -> w"
            ... )
            >>> print(f"Best score: {compiled.candidate_programs[0]['score']}")
        """
        logger.info(f"\n{BOLD}==> BETTERTOGETHER COMPILATION STARTED <=={ENDC}")
        logger.info(f"{BLUE}Strategy:{ENDC} {strategy}")
        logger.info(f"{BLUE}Trainset size:{ENDC} {len(trainset)}")
        logger.info(f"{BLUE}Validation ratio:{ENDC} {valset_ratio if valset is None else 'using provided valset'}")

        student, teacher = self._prepare_student_and_teacher(student, teacher)
        trainset, valset = self._prepare_trainset_and_valset(trainset, valset, valset_ratio)
        effective_max_errors = max_errors if max_errors is not None else dspy.settings.max_errors
        parsed_strategy = self._prepare_strategy(strategy)
        optimizer_compile_args = self._prepare_optimizer_compile_args(optimizer_compile_args, teacher)

        student = self._run_strategies(
            student,
            trainset,
            teacher,
            valset,
            num_threads,
            effective_max_errors,
            provide_traceback,
            seed,
            parsed_strategy,
            shuffle_trainset_between_steps,
            optimizer_compile_args,
        )

        logger.info(f"\n{BOLD}{GREEN}==> BETTERTOGETHER COMPILATION COMPLETE <=={ENDC}")
        logger.info(f"{GREEN}Best score achieved:{ENDC} {student.candidate_programs[0]['score']}")
        logger.info(f"{GREEN}Best strategy:{ENDC} {student.candidate_programs[0]['strategy'] or 'original (no optimization)'}")

        student._compiled = True
        return student

    def _prepare_student_and_teacher(
        self, student: Module, teacher: Module | list[Module] | None
    ) -> tuple[Module, list[Module] | None]:
        student = prepare_student(student)
        all_predictors_have_lms(student)

        if not teacher:
            return student, None
        teacher = [teacher] if not isinstance(teacher, list) else teacher
        teacher = [prepare_teacher(student, t) for t in teacher]

        return student, teacher

    def _prepare_trainset_and_valset(
        self, trainset: list[Example], valset: list[Example] | None, valset_ratio: float
    ) -> tuple[list[Example], list[Example] | None]:
        if not trainset:
            raise ValueError("trainset cannot be empty")

        if valset_ratio < 0 or valset_ratio >= 1:
            raise ValueError(f"valset_ratio must be in range [0, 1), got {valset_ratio}")

        trainset = trainset[:]  # shallow copy to avoid modifying the original

        if valset:
            logger.info(f"{BLUE}Using provided validation set ({len(valset)} examples). Ignoring valset_ratio.{ENDC}")
            return trainset, valset

        if valset_ratio == 0:
            logger.info(f"{YELLOW}No validation set provided and valset_ratio=0. No validation set created.{ENDC}")
            return trainset, None

        logger.info(f"{BLUE}Sampling {valset_ratio:.1%} of trainset as validation set.{ENDC}")
        num_val_examples = int(valset_ratio * len(trainset))
        valset = trainset[:num_val_examples]
        trainset = trainset[num_val_examples:]
        logger.info(f"{BLUE}Created validation set: {len(valset)} examples. Training set: {len(trainset)} examples.{ENDC}")

        return trainset, valset

    def _prepare_strategy(self, strategy: str) -> list[str]:
        if not strategy or not strategy.strip():
            raise ValueError("strategy cannot be empty")

        parsed_strategy = strategy.split(self.STRAT_SEP)

        invalid_steps = [s for s in parsed_strategy if s not in self.optimizers]
        if invalid_steps:
            raise ValueError(
                f"Strategy contains invalid optimizer keys: {invalid_steps}. "
                f"Valid keys are: {list(self.optimizers.keys())}"
            )

        return parsed_strategy

    def _prepare_optimizer_compile_args(
        self,
        optimizer_compile_args: dict[str, dict[str, Any]] | None,
        teacher: list[Module] | None,
    ) -> dict[str, dict[str, Any]]:
        logger.info(f"{BLUE}Validating optimizer compile arguments...{ENDC}")

        # Validate user-provided compile args
        if not optimizer_compile_args:
            return {}

        for optimizer_key, compile_args in optimizer_compile_args.items():
            if optimizer_key not in self.optimizers:
                raise ValueError(
                    f"Invalid optimizer key '{optimizer_key}'. "
                    f"Valid keys are: {list(self.optimizers.keys())}"
                )
            optimizer = self.optimizers[optimizer_key]
            self._validate_compile_args(optimizer, optimizer_key, compile_args)

            # Special checks
            if optimizer.__class__.__name__ == "GEPA":
                # GEPA accepts a teacher argument, but raises an error if it's set.
                if teacher is not None:
                    raise ValueError(
                        "GEPA does not accept a teacher argument. Please remove the teacher argument."
                    )

        return optimizer_compile_args

    def _validate_compile_args(
        self, optimizer: Teleprompter, optimizer_key: str, compile_args: dict[str, Any]
    ) -> None:
        if "student" in compile_args:
            raise ValueError(
                f"'student' is not allowed in optimizer_compile_args for optimizer '{optimizer_key}'. "
                "The same student is used throughout compilation."
            )

        valid_params = inspect.signature(optimizer.compile).parameters
        invalid_args = set(compile_args.keys()) - set(valid_params.keys())
        if invalid_args:
            raise ValueError(
                f"Invalid compile arguments for optimizer '{optimizer_key}': {sorted(invalid_args)}. "
                f"{optimizer.__class__.__name__}.compile() accepts: {list(valid_params.keys())}"
            )

    def _run_strategies(
        self,
        student: Module,
        trainset: list[Example],
        teacher: list[Module] | None,
        valset: list[Example] | None,
        num_threads: int | None,
        effective_max_errors: int | None,
        provide_traceback: bool | None,
        seed: int | None,
        parsed_strategy: list[str],
        shuffle_trainset_between_steps: bool,
        optimizer_args: dict[str, dict[str, Any]],
    ) -> Module:
        rng = random.Random(seed)
        candidate_programs = []
        flag_lms_launched = False
        flag_compilation_error_occurred = False

        # Evaluate original program
        logger.info(f"\n{BOLD}==> BASELINE EVALUATION <=={ENDC}")
        logger.info("Evaluating original program (no optimization applied)")

        launch_lms(student)
        flag_lms_launched = True

        score = self._evaluate_on_valset(student, valset, rng, num_threads, effective_max_errors, provide_traceback)
        self._add_candidate(candidate_programs, student, strategy="", score=score)
        logger.info(f"{YELLOW}Baseline score:{ENDC} {score}")

        # Apply each optimization step
        for ind, step_code in enumerate(parsed_strategy):
            current_strategy = self.STRAT_SEP.join(parsed_strategy[:ind + 1])
            optimizer = self.optimizers[step_code]

            logger.info(f"\n{BOLD}==> STEP {ind + 1}/{len(parsed_strategy)}: {optimizer.__class__.__name__.upper()} <=={ENDC}")
            logger.info(f"{BLUE}Current strategy:{ENDC} '{current_strategy}'")
            logger.info(f"{BLUE}Optimizer:{ENDC} {optimizer.__class__.__name__}")

            try:
                if shuffle_trainset_between_steps:
                    logger.info(f"{BLUE}Shuffling trainset...{ENDC}")
                    rng.shuffle(trainset)

                # Run optimizer, evaluate, and record results
                compile_args = optimizer_args.get(step_code, {})
                student, score, is_new_best, lms_relaunched = self._run_and_evaluate_step(
                    optimizer, student, teacher, trainset, valset, compile_args,
                    candidate_programs, current_strategy, rng,
                    num_threads, effective_max_errors, provide_traceback
                )

                if lms_relaunched:
                    flag_lms_launched = True

                # Log score
                if is_new_best:
                    logger.info(f"{GREEN}New best score!{ENDC} {score} (strategy: '{current_strategy}')")
                else:
                    logger.info(f"{YELLOW}Score after optimization:{ENDC} {score}")

            except Exception as e:
                flag_compilation_error_occurred = True
                logger.error(
                    f"{YELLOW}Step {ind + 1}/{len(parsed_strategy)} failed with error: {type(e).__name__}: {e}{ENDC}"
                )
                logger.error(
                    f"{YELLOW}Stopping optimization early. Returning best program found so far from {len(candidate_programs)} candidate(s).{ENDC}"
                )
                logger.error(f"{YELLOW}Traceback:{ENDC}", exc_info=True)
                break

        # Cleanup and finalize
        if flag_lms_launched:
            kill_lms(student)

        # Sort candidates by score (best first), with earlier programs winning ties
        candidate_programs_with_idx = [(i, cp) for i, cp in enumerate(candidate_programs)]
        candidate_programs_with_idx.sort(
            key=lambda x: (x[1]["score"] if x[1]["score"] is not None else float("-inf"), -x[0]),
            reverse=True
        )
        candidate_programs = [cp for _, cp in candidate_programs_with_idx]

        # Select best program based on valset availability
        if valset is None or len(valset) == 0:
            # No valset: return the latest program (last in original order)
            best_program = candidate_programs_with_idx[-1][1]
        else:
            # Valset provided: return highest score (first after sorting)
            best_program = candidate_programs[0]

        # Attach sorted candidate programs and error flag to the best program
        best_student = best_program["program"]
        best_student.candidate_programs = candidate_programs
        best_student.flag_compilation_error_occurred = flag_compilation_error_occurred

        logger.info(f"\n{BOLD}==> OPTIMIZATION SUMMARY <=={ENDC}")
        logger.info(f"{GREEN}Best score:{ENDC} {best_program['score']}")
        strategy_display = best_program["strategy"] if best_program["strategy"] else "original (no optimization)"
        logger.info(f"{GREEN}Best strategy:{ENDC} {strategy_display}")
        logger.info(f"{BLUE}Total candidates evaluated:{ENDC} {len(candidate_programs)}")

        return best_student

    def _run_and_evaluate_step(
        self,
        optimizer: Teleprompter,
        student: Module,
        teacher: list[Module] | None,
        trainset: list[Example],
        valset: list[Example] | None,
        compile_args: dict[str, Any],
        candidate_programs: list,
        current_strategy: str,
        rng: random.Random,
        num_threads: int | None,
        effective_max_errors: int | None,
        provide_traceback: bool | None,
    ) -> tuple[Module, float | None, bool, bool]:
        """Run optimizer, evaluate result, and record candidate program.

        Returns:
            student: Optimized student program
            score: Validation score (None if no valset)
            is_new_best: Whether this score is the best so far
            lms_relaunched: Whether LMs were relaunched (for flag tracking)
        """
        # Save LMs before optimization
        pred_lms_before = [pred.lm for pred in student.predictors()]

        # Run optimizer
        student._compiled = False
        logger.info(f"{BLUE}Running {optimizer.__class__.__name__} with {len(trainset)} training examples...{ENDC}")

        # Build compile args with standard parameters, filtering to only what the optimizer accepts
        potential_args = {"trainset": trainset, "teacher": teacher, "valset": valset, **compile_args}
        sig = inspect.signature(optimizer.compile)
        accepted_params = set(sig.parameters.keys())
        filtered_compile_args = {k: v for k, v in potential_args.items() if k in accepted_params}

        student = optimizer.compile(student, **filtered_compile_args)

        # Restore LMs if optimizer incorrectly reset them
        # TODO: Some optimizers like BootstrapFewShotWithRandomSearch reset predictor LMs during compilation,
        # which breaks weight optimizers as they require each predictor to have an LM. We should ensure that
        # all optimizers respect the assigned LMs of programs and do not override them. Until then, we restore
        # the original LMs here.
        if not all_predictors_have_lms(student):
            logger.warning(
                f"{YELLOW}Warning: {optimizer.__class__.__name__} incorrectly reset predictor LMs. "
                f"Restoring to original LMs.{ENDC}"
            )
            for pred, lm in zip(student.predictors(), pred_lms_before, strict=False):
                pred.lm = lm

        # Relaunch LMs if models changed (e.g., after fine-tuning)
        # Weight optimizers create new model instances and kill the old ones before returning.
        # We detect this via model name changes and relaunch the new models.
        # Note: launch() and kill() are no-ops for most API-based LMs; these mainly affect local
        # LMs that need launch or clean up routines.
        lms_relaunched = False
        if self._models_changed(student, pred_lms_before):
            launch_lms(student)
            lms_relaunched = True

        # Evaluate optimized program
        score = self._evaluate_on_valset(student, valset, rng, num_threads, effective_max_errors, provide_traceback)
        self._add_candidate(candidate_programs, student, current_strategy, score)

        # Check if this is the best score so far
        valid_scores = [cp["score"] for cp in candidate_programs if cp["score"] is not None]
        best_score_so_far = max(valid_scores) if valid_scores else float("-inf")
        is_new_best = score is not None and score >= best_score_so_far

        return student, score, is_new_best, lms_relaunched

    def _models_changed(self, student: Module, pred_lms_before: list) -> bool:
        """Check if model names changed after optimization (e.g., fine-tuning)."""
        pred_lms_after = [pred.lm for pred in student.predictors()]
        model_names_before = [lm.model if lm else None for lm in pred_lms_before]
        model_names_after = [lm.model if lm else None for lm in pred_lms_after]
        return model_names_before != model_names_after

    def _add_candidate(
        self,
        candidate_programs: list,
        student: Module,
        strategy: str,
        score: float,
    ) -> None:
        """Add a candidate program to the list."""
        candidate_programs.append({
            "score": score,
            "program": student.deepcopy(),
            "strategy": strategy,
        })

    def _evaluate_on_valset(
        self,
        program: Module,
        valset: list[Example] | None,
        rng: random.Random,
        num_threads: int | None,
        effective_max_errors: int | None,
        provide_traceback: bool | None,
    ) -> float | None:
        if valset is None or len(valset) == 0:
            logger.info(f"{YELLOW}No validation set provided. Skipping evaluation.{ENDC}")
            return None

        logger.info(f"{BLUE}Evaluating on {len(valset)} validation examples...{ENDC}")
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=num_threads,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
            provide_traceback=provide_traceback,
        )
        eval_result = eval_candidate_program(len(valset), valset, program, evaluate, rng)
        return eval_result.score
