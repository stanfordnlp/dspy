import inspect
import logging
import random
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, Union

from gepa import GEPAResult
from gepa.core.adapter import ProposalFn
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector

from dspy.clients.lm import LM
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter, DSPyTrace, PredictorFeedbackFn, ScoreWithFeedback
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)

AUTO_RUN_SETTINGS = {
    "light": {"n": 6},
    "medium": {"n": 12},
    "heavy": {"n": 18},
}

@experimental(version="3.0.0")
class GEPAFeedbackMetric(Protocol):
    def __call__(
        gold: Example,
        pred: Prediction,
        trace: Optional["DSPyTrace"],
        pred_name: str | None,
        pred_trace: Optional["DSPyTrace"],
    ) -> Union[float, "ScoreWithFeedback"]:
        """
        This function is called with the following arguments:
        - gold: The gold example.
        - pred: The predicted output.
        - trace: Optional. The trace of the program's execution.
        - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which 
            the feedback is being requested.
        - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

        Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
        feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
        and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
        If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str) corresponding 
        to the predictor.
        If not available at the predictor level, the metric can also return a text feedback at the program level
        (using just the gold, pred and trace).
        If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: 
        f"This trajectory got a score of {score}."
        """
        ...

@experimental(version="3.0.0")
@dataclass(frozen=True)
class DspyGEPAResult:
    """
    Additional data related to the GEPA run.

    Fields:
    - candidates: list of proposed candidates (component_name -> component_text)
    - parents: lineage info; for each candidate i, parents[i] is a list of parent indices or None
    - val_aggregate_scores: per-candidate aggregate score on the validation set (higher is better)
    - val_subscores: per-candidate per-instance scores on the validation set (len == num_val_instances)
    - per_val_instance_best_candidates: for each val instance t, a set of candidate indices achieving the best score on t
    - discovery_eval_counts: Budget (number of metric calls / rollouts) consumed up to the discovery of each candidate

    - total_metric_calls: total number of metric calls made across the run
    - num_full_val_evals: number of full validation evaluations performed
    - log_dir: where artifacts were written (if any)
    - seed: RNG seed for reproducibility (if known)

    - best_idx: candidate index with the highest val_aggregate_scores
    - best_candidate: the program text mapping for best_idx
    """
    # Data about the proposed candidates
    candidates: list[Module]
    parents: list[list[int | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[list[float]]
    per_val_instance_best_candidates: list[set[int]]
    discovery_eval_counts: list[int]

    # Optional data
    best_outputs_valset: list[list[tuple[int, list[Prediction]]]] | None = None

    # Optimization metadata
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    log_dir: str | None = None
    seed: int | None = None

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda i: scores[i])

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]

    @property
    def highest_score_achieved_per_val_task(self) -> list[float]:
        return [
            self.val_subscores[list(self.per_val_instance_best_candidates[val_idx])[0]][val_idx]
            for val_idx in range(len(self.val_subscores[0]))
        ]

    def to_dict(self) -> dict[str, Any]:
        cands = [
            {k: v for k, v in cand.items()}
            for cand in self.candidates
        ]

        return dict(
            candidates=cands,
            parents=self.parents,
            val_aggregate_scores=self.val_aggregate_scores,
            best_outputs_valset=self.best_outputs_valset,
            val_subscores=self.val_subscores,
            per_val_instance_best_candidates=[list(s) for s in self.per_val_instance_best_candidates],
            discovery_eval_counts=self.discovery_eval_counts,
            total_metric_calls=self.total_metric_calls,
            num_full_val_evals=self.num_full_val_evals,
            log_dir=self.log_dir,
            seed=self.seed,
            best_idx=self.best_idx,
        )

    @staticmethod
    def from_gepa_result(gepa_result: "GEPAResult", adapter: "DspyAdapter") -> "DspyGEPAResult":
        return DspyGEPAResult(
            candidates=[adapter.build_program(c) for c in gepa_result.candidates],
            parents=gepa_result.parents,
            val_aggregate_scores=gepa_result.val_aggregate_scores,
            best_outputs_valset=gepa_result.best_outputs_valset,
            val_subscores=gepa_result.val_subscores,
            per_val_instance_best_candidates=gepa_result.per_val_instance_best_candidates,
            discovery_eval_counts=gepa_result.discovery_eval_counts,
            total_metric_calls=gepa_result.total_metric_calls,
            num_full_val_evals=gepa_result.num_full_val_evals,
            log_dir=gepa_result.run_dir,
            seed=gepa_result.seed,
        )

@experimental(version="3.0.0")
class GEPA(Teleprompter):
    """
    GEPA is an evolutionary optimizer, which uses reflection to evolve text components
    of complex systems. GEPA is proposed in the paper [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457).
    The GEPA optimization engine is provided by the `gepa` package, available from [https://github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa).

    GEPA captures full traces of the DSPy module's execution, identifies the parts of the trace
    corresponding to a specific predictor, and reflects on the behaviour of the predictor to
    propose a new instruction for the predictor. GEPA allows users to provide textual feedback
    to the optimizer, which is used to guide the evolution of the predictor. The textual feedback
    can be provided at the granularity of individual predictors, or at the level of the entire system's
    execution.

    To provide feedback to the GEPA optimizer, implement a metric as follows:
    ```
    def metric(
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[DSPyTrace] = None,
    ) -> float | ScoreWithFeedback:
        \"""
        This function is called with the following arguments:
        - gold: The gold example.
        - pred: The predicted output.
        - trace: Optional. The trace of the program's execution.
        - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which 
            the feedback is being requested.
        - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.

        Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain
        feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`
        and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.
        If available at the predictor level, the metric should return {'score': float, 'feedback': str} corresponding 
        to the predictor.
        If not available at the predictor level, the metric can also return a text feedback at the program level
        (using just the gold, pred and trace).
        If no feedback is returned, GEPA will use a simple text feedback consisting of just the score: 
        f"This trajectory got a score of {score}."
        \"""
        ...
    ```

    GEPA can also be used as a batch inference-time search strategy, by passing `valset=trainset, track_stats=True, track_best_outputs=True`, and using the
    `detailed_results` attribute of the optimized program (returned by `compile`) to get the Pareto frontier of the batch. `optimized_program.detailed_results.best_outputs_valset` will contain the best outputs for each task in the batch.

    Example:
    ```
    gepa = GEPA(metric=metric, track_stats=True)
    batch_of_tasks = [dspy.Example(...) for task in tasks]
    new_prog = gepa.compile(student, trainset=trainset, valset=batch_of_tasks)
    pareto_frontier = new_prog.detailed_results.val_aggregate_scores
    # pareto_frontier is a list of scores, one for each task in the batch.
    ```

    Args:
        metric: The metric function to use for feedback and evaluation.
        auto: The auto budget to use for the run. Options: "light", "medium", "heavy".
        max_full_evals: The maximum number of full evaluations to perform.
        max_metric_calls: The maximum number of metric calls to perform.
        reflection_minibatch_size: The number of examples to use for reflection in a single GEPA step. Default is 3.
        candidate_selection_strategy: The strategy to use for candidate selection. Default is "pareto", 
            which stochastically selects candidates from the Pareto frontier of all validation scores. 
            Options: "pareto", "current_best".
        reflection_lm: The language model to use for reflection. Required parameter. GEPA benefits from 
            a strong reflection model. Consider using `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)` 
            for optimal performance.
        skip_perfect_score: Whether to skip examples with perfect scores during reflection. Default is True.
        instruction_proposer: Optional custom instruction proposer implementing GEPA's ProposalFn protocol.
            **Default: None (recommended for most users)** - Uses GEPA's proven instruction proposer from 
            the [GEPA library](https://github.com/gepa-ai/gepa), which implements the 
            [`ProposalFn`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py). This default 
            proposer is highly capable and was validated across diverse experiments reported in the GEPA 
            paper and tutorials.

            See documentation on custom instruction proposers 
            [here](https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/#custom-instruction-proposers).
            
            **Advanced Feature**: Only needed for specialized scenarios:
            - **Multi-modal handling**: Processing dspy.Image inputs alongside textual information
            - **Nuanced control over constraints**: Fine-grained control over instruction length, format, 
              and structural requirements beyond standard feedback mechanisms
            - **Domain-specific knowledge injection**: Specialized terminology or context that cannot be 
              provided through feedback_func alone
            - **Provider-specific prompting**: Optimizations for specific LLM providers (OpenAI, Anthropic) 
              with unique formatting preferences
            - **Coupled component updates**: Coordinated updates of multiple components together rather 
              than independent optimization
            - **External knowledge integration**: Runtime access to databases, APIs, or knowledge bases
            
            The default proposer handles the vast majority of use cases effectively. Use 
            MultiModalInstructionProposer() from dspy.teleprompt.gepa.instruction_proposal for visual 
            content or implement custom ProposalFn for highly specialized requirements.
            
            Note: When both instruction_proposer and reflection_lm are set, the instruction_proposer is called 
            in the reflection_lm context. However, reflection_lm is optional when using a custom instruction_proposer. 
            Custom instruction proposers can invoke their own LLMs if needed.
        component_selector: Custom component selector implementing the ReflectionComponentSelector protocol,
            or a string specifying a built-in selector strategy. Controls which components (predictors) are selected 
            for optimization at each iteration. Defaults to 'round_robin' strategy which cycles through components 
            one at a time. Available string options: 'round_robin' (cycles through components sequentially), 
            'all' (selects all components for simultaneous optimization). Custom selectors can implement strategies 
            using LLM-driven selection logic based on optimization state and trajectories. 
            See [gepa component selectors](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/component_selector.py) 
            for available built-in selectors and the ReflectionComponentSelector protocol for implementing custom selectors.
        add_format_failure_as_feedback: Whether to add format failures as feedback. Default is False.
        use_merge: Whether to use merge-based optimization. Default is True.
        max_merge_invocations: The maximum number of merge invocations to perform. Default is 5.
        num_threads: The number of threads to use for evaluation with `Evaluate`. Optional.
        failure_score: The score to assign to failed examples. Default is 0.0.
        perfect_score: The maximum score achievable by the metric. Default is 1.0. Used by GEPA 
            to determine if all examples in a minibatch are perfect.
        log_dir: The directory to save the logs. GEPA saves elaborate logs, along with all candidate 
            programs, in this directory. Running GEPA with the same `log_dir` will resume the run 
            from the last checkpoint.
        track_stats: Whether to return detailed results and all proposed programs in the `detailed_results` 
            attribute of the optimized program. Default is False.
        use_wandb: Whether to use wandb for logging. Default is False.
        wandb_api_key: The API key to use for wandb. If not provided, wandb will use the API key 
            from the environment variable `WANDB_API_KEY`.
        wandb_init_kwargs: Additional keyword arguments to pass to `wandb.init`.
        track_best_outputs: Whether to track the best outputs on the validation set. track_stats must 
            be True if track_best_outputs is True. The optimized program's `detailed_results.best_outputs_valset` 
            will contain the best outputs for each task in the validation set.
        warn_on_score_mismatch: GEPA (currently) expects the metric to return the same module-level score when 
            called with and without the pred_name. This flag (defaults to True) determines whether a warning is 
            raised if a mismatch in module-level and predictor-level score is detected.
        seed: The random seed to use for reproducibility. Default is 0.
        gepa_kwargs: (Optional) provide additional kwargs to be passed to [gepa.optimize](https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py) method
        
    Note:
        Budget Configuration: Exactly one of `auto`, `max_full_evals`, or `max_metric_calls` must be provided.
        The `auto` parameter provides preset configurations: "light" for quick experimentation, "medium" for
        balanced optimization, and "heavy" for thorough optimization.
        
        Reflection Configuration: The `reflection_lm` parameter is required and should be a strong language model.
        GEPA performs best with models like `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)`.
        The reflection process analyzes failed examples to generate feedback for program improvement.
        
        Merge Configuration: GEPA can merge successful program variants using `use_merge=True`.
        The `max_merge_invocations` parameter controls how many merge attempts are made during optimization.
        
        Evaluation Configuration: Use `num_threads` to parallelize evaluation. The `failure_score` and 
        `perfect_score` parameters help GEPA understand your metric's range and optimize accordingly.
        
        Logging Configuration: Set `log_dir` to save detailed logs and enable checkpoint resuming.
        Use `track_stats=True` to access detailed optimization results via the `detailed_results` attribute.
        Enable `use_wandb=True` for experiment tracking and visualization.
        
        Reproducibility: Set `seed` to ensure consistent results across runs with the same configuration.
    """
    def __init__(
        self,
        metric: GEPAFeedbackMetric,
        *,
        # Budget configuration
        auto: Literal["light", "medium", "heavy"] | None = None,
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        # Reflection configuration
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        reflection_lm: LM | None = None,
        skip_perfect_score: bool = True,
        add_format_failure_as_feedback: bool = False,
        instruction_proposer: "ProposalFn | None" = None,
        component_selector: "ReflectionComponentSelector | str" = "round_robin",
        # Merge-based configuration
        use_merge: bool = True,
        max_merge_invocations: int | None = 5,
        # Evaluation configuration
        num_threads: int | None = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        # Logging
        log_dir: str = None,
        track_stats: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        track_best_outputs: bool = False,
        warn_on_score_mismatch: bool = True,
        use_mlflow: bool = False,
        # Reproducibility
        seed: int | None = 0,
        # GEPA passthrough kwargs
        gepa_kwargs: dict | None = None
    ):
        try:
            inspect.signature(metric).bind(None, None, None, None, None)
        except TypeError as e:
            raise TypeError(
                "GEPA metric must accept five arguments: (gold, pred, trace, pred_name, pred_trace). "
                "See https://dspy.ai/api/optimizers/GEPA for details."
            ) from e

        self.metric_fn = metric

        # Budget configuration
        assert (
            (max_metric_calls is not None) +
            (max_full_evals is not None) +
            (auto is not None)
            == 1
        ), (
            "Exactly one of max_metric_calls, max_full_evals, auto must be set. "
            f"You set max_metric_calls={max_metric_calls}, "
            f"max_full_evals={max_full_evals}, "
            f"auto={auto}."
        )
        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls

        # Reflection configuration
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy

        assert reflection_lm is not None or instruction_proposer is not None, (
            "GEPA requires a reflection language model, or custom instruction proposer to be provided. "
            "Typically, you can use `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)` to get a good reflection model. "
            "Reflection LM is used by GEPA to reflect on the behavior of the program and propose new instructions, and will benefit from a strong model. "
        )

        self.reflection_lm = reflection_lm
        self.skip_perfect_score = skip_perfect_score
        self.add_format_failure_as_feedback = add_format_failure_as_feedback

        # Merge-based configuration
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations

        # Evaluation Configuration
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.perfect_score = perfect_score

        # Logging configuration
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.use_mlflow = use_mlflow

        if track_best_outputs:
            assert track_stats, "track_stats must be True if track_best_outputs is True."
        self.track_best_outputs = track_best_outputs

        # Reproducibility
        self.seed = seed

        self.custom_instruction_proposer = instruction_proposer
        self.component_selector = component_selector
        self.gepa_kwargs = gepa_kwargs or {}

    def auto_budget(self, num_preds, num_candidates, valset_size: int, minibatch_size: int = 35, full_eval_steps: int = 5) -> int:
        import numpy as np
        num_trials = int(max(2 * (num_preds * 2) * np.log2(num_candidates), 1.5 * num_candidates))
        if num_trials < 0 or valset_size < 0 or minibatch_size < 0:
            raise ValueError("num_trials, valset_size, and minibatch_size must be >= 0.")
        if full_eval_steps < 1:
            raise ValueError("full_eval_steps must be >= 1.")

        V = valset_size
        N = num_trials
        M = minibatch_size
        m = full_eval_steps

        # Initial full evaluation on the default program
        total = V

        # Assume upto 5 trials for bootstrapping each candidate
        total += num_candidates * 5

        # N minibatch evaluations
        total += N * M
        if N == 0:
            return total  # no periodic/full evals inside the loop
        # Periodic full evals occur when trial_num % (m+1) == 0, where trial_num runs 2..N+1
        periodic_fulls = (N + 1) // (m) + 1
        # If 1 <= N < m, the code triggers one final full eval at the end
        extra_final = 1 if N < m else 0

        total += (periodic_fulls + extra_final) * V
        return total

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
    ) -> Module:
        """
        GEPA uses the trainset to perform reflective updates to the prompt, but uses the valset for tracking Pareto scores.
        If no valset is provided, GEPA will use the trainset for both.

        Parameters:
        - student: The student module to optimize.
        - trainset: The training set to use for reflective updates.
        - valset: The validation set to use for tracking Pareto scores. If not provided, GEPA will use the trainset for both.
        """
        from gepa import GEPAResult, optimize

        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter, LoggerAdapter

        assert trainset is not None and len(trainset) > 0, "Trainset must be provided and non-empty"
        assert teacher is None, "Teacher is not supported in DspyGEPA yet."

        if self.auto is not None:
            self.max_metric_calls = self.auto_budget(
                num_preds=len(student.predictors()),
                num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],
                valset_size=len(valset) if valset is not None else len(trainset),
            )
        elif self.max_full_evals is not None:
            self.max_metric_calls = self.max_full_evals * (len(trainset) + (len(valset) if valset is not None else 0))
        else:
            assert self.max_metric_calls is not None, "Either auto, max_full_evals, or max_metric_calls must be set."

        logger.info(f"Running GEPA for approx {self.max_metric_calls} metric calls of the program. This amounts to {self.max_metric_calls / len(trainset) if valset is None else self.max_metric_calls / (len(trainset) + len(valset)):.2f} full evals on the {'train' if valset is None else 'train+val'} set.")

        if valset is None:
            logger.warning("No valset provided; Using trainset as valset. This is useful as an inference-time scaling strategy where you want GEPA to find the best solutions for the provided tasks in the trainset, as it makes GEPA overfit prompts to the provided trainset. In order to ensure generalization and perform well on unseen tasks, please provide separate trainset and valset. Provide the smallest valset that is just large enough to match the downstream task distribution, while keeping trainset as large as possible.")
        valset = valset or trainset
        logger.info(f"Using {len(valset)} examples for tracking Pareto scores. You can consider using a smaller sample of the valset to allow GEPA to explore more diverse solutions within the same budget. GEPA requires you to provide the smallest valset that is just large enough to match your downstream task distribution, while providing as large trainset as possible.")

        rng = random.Random(self.seed)

        def feedback_fn_creator(pred_name: str, predictor) -> "PredictorFeedbackFn":
            def feedback_fn(
                predictor_output: dict[str, Any],
                predictor_inputs: dict[str, Any],
                module_inputs: Example,
                module_outputs: Prediction,
                captured_trace: "DSPyTrace",
            ) -> "ScoreWithFeedback":
                trace_for_pred = [(predictor, predictor_inputs, predictor_output)]
                o = self.metric_fn(
                    module_inputs,
                    module_outputs,
                    captured_trace,
                    pred_name,
                    trace_for_pred,
                )
                if hasattr(o, "feedback"):
                    if o["feedback"] is None:
                        o["feedback"] = f"This trajectory got a score of {o['score']}."
                    return o
                else:
                    return dict(score=o, feedback=f"This trajectory got a score of {o}.")
            return feedback_fn

        feedback_map = {
            k: feedback_fn_creator(k, v)
            for k, v in student.named_predictors()
        }

        # Build the DSPy adapter that encapsulates evaluation, trace capture, feedback extraction, and instruction proposal
        adapter = DspyAdapter(
            student_module=student,
            metric_fn=self.metric_fn,
            feedback_map=feedback_map,
            failure_score=self.failure_score,
            num_threads=self.num_threads,
            add_format_failure_as_feedback=self.add_format_failure_as_feedback,
            rng=rng,
            reflection_lm=self.reflection_lm,
            custom_instruction_proposer=self.custom_instruction_proposer,
            warn_on_score_mismatch=self.warn_on_score_mismatch
        )

        # Instantiate GEPA with the simpler adapter-based API
        base_program = {name: pred.signature.instructions for name, pred in student.named_predictors()}
        gepa_result: GEPAResult = optimize(
            seed_candidate=base_program,
            trainset=trainset,
            valset=valset,
            adapter=adapter,

            # Reflection-based configuration
            reflection_lm=(lambda x: self.reflection_lm(x)[0]) if self.reflection_lm is not None else None,
            candidate_selection_strategy=self.candidate_selection_strategy,
            skip_perfect_score=self.skip_perfect_score,
            reflection_minibatch_size=self.reflection_minibatch_size,
            module_selector=self.component_selector,

            perfect_score=self.perfect_score,

            # Merge-based configuration
            use_merge=self.use_merge,
            max_merge_invocations=self.max_merge_invocations,

            # Budget
            max_metric_calls=self.max_metric_calls,

            # Logging
            logger=LoggerAdapter(logger),
            run_dir=self.log_dir,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            wandb_init_kwargs=self.wandb_init_kwargs,
            use_mlflow=self.use_mlflow,
            track_best_outputs=self.track_best_outputs,
            display_progress_bar=True,
            raise_on_exception=True,

            # Reproducibility
            seed=self.seed,
            **self.gepa_kwargs
        )

        new_prog = adapter.build_program(gepa_result.best_candidate)

        if self.track_stats:
            dspy_gepa_result = DspyGEPAResult.from_gepa_result(gepa_result, adapter)
            new_prog.detailed_results = dspy_gepa_result

        return new_prog
