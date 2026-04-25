"""
EvoX: DSPy optimizer with co-evolution of search strategies.

Thin adapter that delegates ALL algorithm logic to skydiscover's CoEvolutionController —
including stagnation detection, search strategy generation, strategy validation,
hot-swap with migration, fallback on error, and solution scoring via LogWindowScorer.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Callable

import dspy
from dspy.primitives import Example, Module
from dspy.teleprompt.teleprompt import Teleprompter

from skydiscover.config import (
    Config,
    ContextBuilderConfig,
    DatabaseConfig,
    EvaluatorConfig,
    EvoxDatabaseConfig,
    LLMConfig,
    SearchConfig,
)
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.evox.controller import CoEvolutionController
from skydiscover.search.evox.utils.search_scorer import LogWindowScorer
from skydiscover.search.registry import create_database
from skydiscover.search.utils.discovery_utils import load_database_from_file

from .adapter import (
    DSPY_SYSTEM_MESSAGE,
    TempFiles,
    best_program_to_module,
    create_eval_file,
    make_dspy_eval_fn,
    make_llm_config,
    register_eval,
    run_async,
    seed_database,
    select_best_on_valset,
)

logger = logging.getLogger(__name__)


def _ensure_program_score_property():
    """Add a ``.score`` property to Program if missing.

    Co-evolution generates Python search strategies that access
    ``program.score``, but skydiscover's Program dataclass stores
    scores in ``program.metrics["combined_score"]``.  This bridge
    makes both access patterns work.
    """
    if not isinstance(getattr(Program, "score", None), property):
        Program.score = property(
            lambda self: self.metrics.get("combined_score", 0.0)
        )


class _DSPyCoEvolutionController(CoEvolutionController):
    """CoEvolutionController with programmatic search controller setup.

    Overrides ``_init_search_evolution_controller`` to create the search
    controller from in-memory config rather than file-based ``setup_search``,
    so we can inject our DSPy LLM backend into the search-side LLM pool.
    """

    def __init__(
        self,
        controller_input: DiscoveryControllerInput,
        *,
        guide_llm_cfg,
        search_eval_file: str,
        initial_strategy_path: str,
    ):
        # Stash before super().__init__ calls _init_search_evolution_controller
        self._guide_llm_cfg = guide_llm_cfg
        self._search_eval_file = search_eval_file
        self._initial_strategy_path = initial_strategy_path
        super().__init__(controller_input)

    def _init_search_evolution_controller(self) -> None:
        """Create search controller programmatically (bypass setup_search)."""
        # Read initial search strategy code
        with open(self._initial_strategy_path) as f:
            self._search_initial_code = f.read()

        # Build search controller config (search strategies are Python code)
        search_config = Config(
            language="python",
            file_suffix=".py",
            diff_based_generation=False,
            max_iterations=1,
            llm=LLMConfig(
                models=[self._guide_llm_cfg],
                evaluator_models=[self._guide_llm_cfg],
                guide_models=[self._guide_llm_cfg],
            ),
            evaluator=EvaluatorConfig(
                evaluation_file=self._search_eval_file,
                file_suffix=".py",
                cascade_evaluation=False,
            ),
            context_builder=ContextBuilderConfig(template="evox"),
            search=SearchConfig(type="topk"),
        )

        search_db = create_database("topk", DatabaseConfig())
        search_input = DiscoveryControllerInput(
            config=search_config,
            evaluation_file=self._search_eval_file,
            database=search_db,
            file_suffix=".py",
            output_dir=tempfile.mkdtemp(prefix="dspy_evox_search_"),
        )

        self.search_controller = DiscoveryController(search_input)
        self.search_scorer = LogWindowScorer()
        self._active_search_algorithm_code = self._search_initial_code

        db_cfg = self.config.search.database
        self._log_coevolution_setup(db_cfg)
        self._init_search_tracking()


class EvoX(Teleprompter):
    """
    Co-evolutionary prompt optimizer: evolves both prompts AND the search strategy.

    Wraps skydiscover's ``CoEvolutionController`` with thin DSPy adapters.
    When prompt optimization stagnates, EvoX uses a guide LM to generate a new
    ``ProgramDatabase`` class that controls parent selection. The new strategy is
    validated, hot-swapped in, and scored by how much improvement it drives.

    Example::

        optimizer = dspy.EvoX(
            metric=my_metric,
            reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0),
            guide_lm=dspy.LM("openai/gpt-4o", temperature=0.7),
            max_iterations=50,
        )
        optimized = optimizer.compile(MyModule(), trainset=train, valset=val)
    """

    def __init__(
        self,
        metric: Callable,
        *,
        reflection_lm: dspy.LM,
        guide_lm: dspy.LM | None = None,
        # Budget
        max_iterations: int = 50,
        # Co-evolution
        switch_ratio: float = 0.10,
        improvement_threshold: float = 0.01,
        # Evaluation
        num_threads: int | None = None,
        # Reproducibility
        seed: int = 0,
    ):
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.guide_lm = guide_lm or reflection_lm
        self.max_iterations = max_iterations
        self.switch_ratio = switch_ratio
        self.improvement_threshold = improvement_threshold
        self.num_threads = num_threads
        self.seed = seed

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

        # Bridge: make Program.score accessible for generated search strategies
        _ensure_program_score_property()

        # 1. Register evaluation function — scored on trainset with feedback
        eval_fn = make_dspy_eval_fn(
            student, self.metric, trainset, self.num_threads,
        )
        pred_names = [name for name, _ in student.named_predictors()]
        register_eval("current", eval_fn, predictor_names=pred_names)

        with TempFiles() as tmp:
            # 2. Create temp evaluation file for solution-side
            eval_file = tmp.add(create_eval_file())

            # 3. EvoxDatabaseConfig auto-sets database_file_path, evaluation_file,
            # and config_path to skydiscover's built-in defaults via __post_init__
            db_config = EvoxDatabaseConfig()

            # 4. Load initial search strategy and create solution database from it
            solution_db = self._load_initial_database(db_config.database_file_path)
            solution_db.language = "text"

            # 5. Seed solution database (on trainset)
            seed_prog = seed_database(
                solution_db, student, self.metric, trainset, self.num_threads,
            )

            # 6. Build config for the solution side
            llm_cfg = make_llm_config(self.reflection_lm, "reflection")
            guide_llm_cfg = make_llm_config(self.guide_lm, "guide")

            eval_timeout = max(1800, len(trainset) * 3)

            config = Config(
                max_iterations=self.max_iterations,
                language="text",
                file_suffix=".json",
                diff_based_generation=False,
                llm=LLMConfig(
                    models=[llm_cfg],
                    evaluator_models=[llm_cfg],
                    guide_models=[guide_llm_cfg],
                ),
                evaluator=EvaluatorConfig(
                    evaluation_file=eval_file,
                    file_suffix=".json",
                    cascade_evaluation=False,
                    timeout=eval_timeout,
                ),
                search=SearchConfig(
                    type="evox",
                    database=db_config,
                ),
                context_builder=ContextBuilderConfig(
                    system_message=DSPY_SYSTEM_MESSAGE,
                ),
            )

            # 7. Apply co-evolution thresholds
            CoEvolutionController.DEFAULT_SWITCH_RATIO = self.switch_ratio
            CoEvolutionController.DEFAULT_IMPROVEMENT_THRESHOLD = self.improvement_threshold

            output_dir = tempfile.mkdtemp(prefix="dspy_evox_")

            logger.info(
                f"EvoX: {self.max_iterations} iterations, "
                f"switch_ratio={self.switch_ratio}, "
                f"seed score={seed_prog.metrics.get('combined_score', 0):.4f}"
            )

            # 8. Create controller with DSPy-adapted search controller
            controller_input = DiscoveryControllerInput(
                config=config,
                evaluation_file=eval_file,
                database=solution_db,
                file_suffix=".json",
                output_dir=output_dir,
            )
            controller = _DSPyCoEvolutionController(
                controller_input,
                guide_llm_cfg=guide_llm_cfg,
                search_eval_file=db_config.evaluation_file,
                initial_strategy_path=db_config.database_file_path,
            )

            # 9. Run the full co-evolution loop
            run_async(controller.run_discovery(0, self.max_iterations))

            # 10. Select best program
            final_db = controller.database
            if has_valset:
                logger.info(
                    f"Re-evaluating top 5 candidates on valset "
                    f"({len(valset)} examples)..."
                )
                return select_best_on_valset(
                    final_db, student, self.metric, valset,
                    self.num_threads, top_k=5,
                )
            return best_program_to_module(final_db, student)

    @staticmethod
    def _load_initial_database(strategy_path: str) -> ProgramDatabase:
        """Load the initial search strategy and create a solution database."""
        db_class, prog_class = load_database_from_file(strategy_path)
        db = db_class("evox", EvoxDatabaseConfig())
        db._program_class = prog_class
        return db
