"""
AdaEvolve: DSPy optimizer powered by skydiscover's adaptive multi-island evolutionary search.

Thin adapter that delegates ALL algorithm logic to skydiscover's AdaEvolveController —
including multi-island adaptive search, UCB island selection, paradigm breakthroughs,
migration, mode-aware prompting, and error retry.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Callable

import dspy
from dspy.primitives import Example, Module
from dspy.teleprompt.teleprompt import Teleprompter

from skydiscover.config import (
    AdaEvolveDatabaseConfig,
    Config,
    ContextBuilderConfig,
    EvaluatorConfig,
    LLMConfig,
    SearchConfig,
)
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput

from .adapter import (
    DSPY_SYSTEM_MESSAGE,
    TempFiles,
    _extract_per_example_scores,
    best_program_to_module,
    build_dspy_program,
    create_eval_file,
    make_dspy_eval_fn,
    make_llm_config,
    register_eval,
    run_async,
    seed_database,
    select_best_on_valset,
    select_pareto_subset,
)

logger = logging.getLogger(__name__)


class AdaEvolve(Teleprompter):
    """
    Evolutionary prompt optimizer using skydiscover's AdaEvolve algorithm.

    Wraps skydiscover's ``AdaEvolveController`` with thin DSPy adapters for
    LLM calls and evaluation. All optimization logic — island management, UCB
    selection, adaptive search intensity, paradigm breakthroughs, migration —
    runs inside skydiscover.

    Example::

        optimizer = dspy.AdaEvolve(
            metric=my_metric,
            reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0),
            max_iterations=30,
        )
        optimized = optimizer.compile(MyModule(), trainset=train, valset=val)
    """

    def __init__(
        self,
        metric: Callable,
        *,
        reflection_lm: dspy.LM,
        # Budget
        max_iterations: int = 30,
        # AdaEvolve config
        num_islands: int | None = None,
        population_size: int = 20,
        use_paradigm_breakthrough: bool | None = None,
        use_adaptive_search: bool = True,
        use_ucb_selection: bool = True,
        use_migration: bool = True,
        # Multi-objective: select a difficulty-stratified subset of
        # training examples as Pareto objectives (q000, q001, ...)
        # like frontier_cs's per-problem metrics.  NSGA-II then
        # balances performance across individual examples instead of
        # optimizing a single average.  Set pareto_size to enable;
        # the subset is chosen by seed-scoring the trainset and
        # over-sampling hard (unsolved) examples.
        pareto_size: int = 0,
        pareto_objectives_weight: float = 0.2,
        # Evaluation
        num_threads: int | None = None,
        feedback_sample_size: int = 50,
        # Reproducibility
        seed: int = 0,
    ):
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.use_adaptive_search = use_adaptive_search
        self.use_ucb_selection = use_ucb_selection
        self.use_migration = use_migration
        self.pareto_size = pareto_size
        self.pareto_objectives_weight = pareto_objectives_weight
        self.num_threads = num_threads
        self.feedback_sample_size = feedback_sample_size
        self.seed = seed

        # Auto-configure for budget: at low iteration counts, multi-island
        # search wastes budget.  Default to 1 island when ≤15 iterations.
        if num_islands is None:
            self.num_islands = 1 if max_iterations <= 15 else 2
        else:
            self.num_islands = num_islands

        # Paradigm breakthrough is always on — it's the primary mechanism
        # for escaping local optima on both short and long runs.
        if use_paradigm_breakthrough is None:
            self.use_paradigm_breakthrough = True
        else:
            self.use_paradigm_breakthrough = use_paradigm_breakthrough

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        valset: list[Example] | None = None,
    ) -> Module:
        # Evolution scores on trainset with feedback from trainset.
        # If a separate valset is provided, the top-k candidates are
        # re-evaluated on valset after evolution to pick the one that
        # generalises best.
        has_valset = valset is not None
        if not has_valset:
            valset = trainset

        # 1. Determine Pareto subset if multi-objective is enabled.
        # We run the seed program on the trainset first to get per-example
        # scores, then select a difficulty-stratified subset: over-sample
        # hard (unsolved) examples where prompts actually differ.
        pareto_indices: list[int] | None = None
        pareto_objectives: list[str] = []

        if self.pareto_size > 0:
            logger.info(
                f"Running seed evaluation to select Pareto subset "
                f"({self.pareto_size} examples from {len(trainset)})..."
            )
            seed_program = build_dspy_program(student, {})
            seed_evaluator = dspy.Evaluate(
                devset=trainset, metric=self.metric,
                num_threads=self.num_threads or 4,
            )
            seed_result = seed_evaluator(seed_program)
            seed_scores = _extract_per_example_scores(seed_result.results)

            pareto_indices = select_pareto_subset(
                trainset, seed_scores,
                pareto_size=self.pareto_size, seed=self.seed,
            )
            pareto_objectives = [f"q{i:03d}" for i in range(len(pareto_indices))]

        # 2. Register evaluation function — scored on trainset with feedback.
        # When pareto_indices is set, per-example scores for the selected
        # subset are added as q000, q001, ... like frontier_cs's p00, p01.
        eval_fn = make_dspy_eval_fn(
            student, self.metric, trainset, self.num_threads,
            pareto_indices=pareto_indices,
        )
        pred_names = [name for name, _ in student.named_predictors()]
        register_eval("current", eval_fn, predictor_names=pred_names)

        with TempFiles() as tmp:
            # 2. Create temp evaluation file (bridges file-based eval to registry)
            eval_file = tmp.add(create_eval_file())

            # 3. Build skydiscover Config
            llm_cfg = make_llm_config(self.reflection_lm, "reflection")

            db_config = AdaEvolveDatabaseConfig(
                population_size=self.population_size,
                num_islands=self.num_islands,
                use_paradigm_breakthrough=self.use_paradigm_breakthrough,
                use_adaptive_search=self.use_adaptive_search,
                use_ucb_selection=self.use_ucb_selection,
                use_migration=self.use_migration,
                diversity_strategy="text",
                use_dynamic_islands=True,
                migration_interval=max(self.max_iterations // 4, 5),
                enable_error_retry=True,
                max_error_retries=2,
                # Novelty pressure: protect diverse prompt styles in the
                # archive even if their fitness is lower — prompt space is
                # highly non-convex and novel phrasings can unlock better
                # solutions when refined.
                fitness_weight=0.7,
                novelty_weight=0.3,
                # Paradigm breakthrough tuned for prompt optimization:
                # longer window (prompts improve slower than code) and
                # slightly more aggressive trigger.
                paradigm_window_size=20,
                paradigm_improvement_threshold=0.10,
                # Multi-objective Pareto: balance across example groups
                pareto_objectives=pareto_objectives,
                pareto_objectives_weight=(
                    self.pareto_objectives_weight if pareto_objectives else 0.0
                ),
                fitness_key="combined_score",
            )

            # Generous timeout: the split eval runs val (N examples) +
            # train sample (feedback_sample_size examples) per candidate.
            eval_timeout = max(
                1800,
                (len(valset) + self.feedback_sample_size) * 3,
            )

            config = Config(
                max_iterations=self.max_iterations,
                language="text",
                file_suffix=".json",
                diff_based_generation=False,
                llm=LLMConfig(
                    models=[llm_cfg],
                    evaluator_models=[llm_cfg],
                    guide_models=[llm_cfg],
                ),
                evaluator=EvaluatorConfig(
                    evaluation_file=eval_file,
                    file_suffix=".json",
                    cascade_evaluation=False,
                    timeout=eval_timeout,
                ),
                search=SearchConfig(
                    type="adaevolve",
                    database=db_config,
                ),
                context_builder=ContextBuilderConfig(
                    system_message=DSPY_SYSTEM_MESSAGE,
                ),
            )

            # 4. Create database and seed (on trainset)
            database = AdaEvolveDatabase("adaevolve", db_config)
            database.language = "text"
            seed_prog = seed_database(
                database, student, self.metric, trainset, self.num_threads,
                pareto_indices=pareto_indices,
            )

            logger.info(
                f"AdaEvolve: {self.max_iterations} iterations, "
                f"{self.num_islands} islands, "
                f"paradigm_breakthrough={self.use_paradigm_breakthrough}, "
                f"seed score={seed_prog.metrics.get('combined_score', 0):.4f}"
            )

            # 5. Create controller (uses our LLM backend + eval file via config)
            controller_input = DiscoveryControllerInput(
                config=config,
                evaluation_file=eval_file,
                database=database,
                file_suffix=".json",
                output_dir=tempfile.mkdtemp(prefix="dspy_adaevolve_"),
            )
            controller = AdaEvolveController(controller_input)

            # 6. Run the full optimization loop
            run_async(controller.run_discovery(0, self.max_iterations))

            # 7. Select best program
            if has_valset:
                # Re-evaluate top candidates on valset to avoid overfitting
                logger.info(
                    f"Re-evaluating top 5 candidates on valset "
                    f"({len(valset)} examples)..."
                )
                return select_best_on_valset(
                    database, student, self.metric, valset,
                    self.num_threads, top_k=5,
                )
            return best_program_to_module(database, student)
