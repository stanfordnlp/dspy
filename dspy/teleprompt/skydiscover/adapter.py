"""
Thin adapters bridging DSPy objects to skydiscover's controller interfaces.

Provides:
- DSPyLLMBackend: wraps dspy.LM as skydiscover's LLMInterface
- Evaluation registry: bridges in-memory DSPy eval to file-based skydiscover eval
- Conversion utilities: DSPy Module ↔ skydiscover Program
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import tempfile
import textwrap
import uuid
from typing import Any, Callable, Dict, List

import dspy
from dspy.primitives import Example, Module

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Backend Adapter
# ---------------------------------------------------------------------------


class DSPyLLMBackend(LLMInterface):
    """Wraps a dspy.LM as a skydiscover LLMInterface backend."""

    def __init__(self, cfg_or_lm):
        if isinstance(cfg_or_lm, dspy.LM):
            self.lm = cfg_or_lm
        else:
            # Called via LLMModelConfig.init_client(cfg)
            self.lm = cfg_or_lm._dspy_lm

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        dspy_messages = [{"role": "system", "content": system_message}]
        for m in messages:
            if isinstance(m, dict):
                dspy_messages.append(m)
            else:
                dspy_messages.append({"role": "user", "content": str(m)})

        responses = self.lm(messages=dspy_messages)

        text = ""
        if responses:
            r = responses[0]
            text = r if isinstance(r, str) else r.get("text", str(r))

        return LLMResponse(text=text)


def make_llm_config(dspy_lm: dspy.LM, name: str = "dspy") -> LLMModelConfig:
    """Create an LLMModelConfig backed by a dspy.LM."""
    cfg = LLMModelConfig(
        name=name,
        init_client=lambda c: DSPyLLMBackend(c),
        weight=1.0,
    )
    cfg._dspy_lm = dspy_lm  # stashed for init_client closure
    return cfg


# ---------------------------------------------------------------------------
# Evaluation Registry
# ---------------------------------------------------------------------------

_EVAL_REGISTRY: Dict[str, Any] = {}


def register_eval(key: str, fn: Callable, predictor_names: list[str] | None = None):
    """Register an evaluation function and predictor names (called from compile())."""
    _EVAL_REGISTRY[key] = fn
    if predictor_names is not None:
        _EVAL_REGISTRY["predictor_names"] = predictor_names


def get_eval_fn() -> Callable:
    """Get the current evaluation function (called from the generated eval file)."""
    return _EVAL_REGISTRY["current"]


def get_predictor_names() -> list[str]:
    """Get the predictor names (called from the generated eval file for fallback parsing)."""
    return _EVAL_REGISTRY.get("predictor_names", [])


def create_eval_file() -> str:
    """Write a temp evaluation file that delegates to the registry.

    The generated file returns an ``EvaluationResult`` so that
    skydiscover's evaluator preserves the ``artifacts`` dict (which
    carries per-example feedback to the context builder).
    """
    code = textwrap.dedent("""\
        import json
        import re
        from dspy.teleprompt.skydiscover.adapter import get_eval_fn, get_predictor_names
        from skydiscover.evaluation.evaluation_result import EvaluationResult

        def _parse_candidate(solution):
            # 1. Try direct JSON parse
            try:
                return json.loads(solution)
            except (json.JSONDecodeError, ValueError):
                pass
            # 2. Try to find a JSON object embedded in the text
            match = re.search(r'\\{[^{}]*\\}', solution)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    pass
            # 3. Fallback: treat entire text as instruction for all predictors
            names = get_predictor_names()
            if names:
                return {name: solution.strip() for name in names}
            return None

        def evaluate(program_path):
            with open(program_path) as f:
                solution = f.read()
            candidate = _parse_candidate(solution)
            if candidate is None:
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "error": "unparseable"},
                )
            result = get_eval_fn()(candidate)
            artifacts = result.pop("artifacts", {})
            return EvaluationResult(metrics=result, artifacts=artifacts)
    """)
    fd, path = tempfile.mkstemp(suffix=".py", prefix="dspy_eval_")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    return path


# Maximum number of failure feedbacks to include in artifacts.
# Keeps the context builder prompt bounded while still giving the
# mutation LLM concrete examples of what went wrong.
_MAX_FEEDBACK_EXAMPLES = 10


def _collect_feedback(results) -> str | None:
    """Extract failure feedback strings from dspy.Evaluate results.

    *results* is a list of ``(example, prediction, metric_output)``
    tuples.  We gather ``.feedback`` strings from failed examples
    (up to ``_MAX_FEEDBACK_EXAMPLES``).
    """
    parts: list[str] = []
    for _example, _prediction, metric_output in results:
        if len(parts) >= _MAX_FEEDBACK_EXAMPLES:
            break
        m_score = None
        m_feedback = None
        if hasattr(metric_output, "score"):
            m_score = metric_output.score
            m_feedback = getattr(metric_output, "feedback", None)
        elif isinstance(metric_output, (int, float, bool)):
            m_score = metric_output
        if m_score is not None and not m_score and m_feedback:
            parts.append(str(m_feedback))
    return "\n\n".join(parts) if parts else None


def _extract_per_example_scores(results) -> List[float]:
    """Extract a per-example score list from dspy.Evaluate results."""
    scores: List[float] = []
    for _example, _prediction, metric_output in results:
        m_score = 0.0
        if hasattr(metric_output, "score"):
            m_score = float(bool(metric_output.score))
        elif isinstance(metric_output, (int, float, bool)):
            m_score = float(bool(metric_output))
        scores.append(m_score)
    return scores


def select_pareto_subset(
    dataset: list[Example],
    seed_scores: List[float],
    pareto_size: int = 30,
    seed: int = 42,
) -> List[int]:
    """Select a difficulty-stratified subset of examples for Pareto objectives.

    Inspired by frontier_cs's per-problem Pareto and GEPA's stratified
    merge sampling, but adapted for prompt optimization:

    1. Stratify examples into solved (seed got right) and unsolved (wrong).
    2. Over-sample unsolved examples — these are where prompts actually
       differ and where Pareto pressure matters most.
    3. Include some solved examples to detect regressions.

    Returns indices into *dataset* for the selected subset.
    """
    import random as _random

    rng = _random.Random(seed)

    solved = [i for i, s in enumerate(seed_scores) if s > 0.5]
    unsolved = [i for i, s in enumerate(seed_scores) if s <= 0.5]

    # Allocate ~70% to unsolved (where improvement happens),
    # ~30% to solved (to catch regressions).
    n_unsolved = min(len(unsolved), int(pareto_size * 0.7))
    n_solved = min(len(solved), pareto_size - n_unsolved)
    # If one bucket is too small, fill from the other.
    n_unsolved = min(len(unsolved), pareto_size - n_solved)

    selected = rng.sample(unsolved, n_unsolved) + rng.sample(solved, n_solved)
    rng.shuffle(selected)

    logger.info(
        f"Pareto subset: {len(selected)} examples selected "
        f"({n_unsolved} unsolved + {n_solved} solved) "
        f"from {len(dataset)} total"
    )
    return selected


def make_dspy_eval_fn(
    student: Module,
    metric: Callable,
    dataset: list[Example],
    num_threads: int | None,
    pareto_indices: List[int] | None = None,
) -> Callable:
    """Create an evaluation function for DSPy instruction candidates.

    Returns a dict with ``combined_score`` (float, 0-1) and an
    ``artifacts`` dict.  If the metric returns objects with a
    ``.feedback`` attribute (e.g. ``dspy.Prediction(score=...,
    feedback=...)``), failure feedbacks are collected and placed in
    ``artifacts["feedback"]`` so that skydiscover's context builder
    can show them to the mutation LLM.

    When *pareto_indices* is provided, per-example scores are added
    for the selected examples as ``q000``, ``q001``, ... following
    the same pattern as frontier_cs's ``p00``, ``p01``, ... .
    Only the subset at the given indices becomes a Pareto objective,
    keeping dimensionality manageable for NSGA-II while focusing on
    the most informative (typically hardest) examples.
    """

    def eval_fn(candidate: dict) -> dict:
        program = build_dspy_program(student, candidate)
        evaluator = dspy.Evaluate(
            devset=dataset,
            metric=metric,
            num_threads=num_threads or 4,
        )
        result = evaluator(program)
        score = result.score / 100.0  # dspy.Evaluate returns 0-100

        metrics: dict[str, Any] = {"combined_score": score}

        # Per-example Pareto objectives for the selected subset.
        if pareto_indices is not None:
            all_scores = _extract_per_example_scores(result.results)
            for q_idx, ex_idx in enumerate(pareto_indices):
                if ex_idx < len(all_scores):
                    metrics[f"q{q_idx:03d}"] = all_scores[ex_idx]

        artifacts: dict[str, Any] = {}
        feedback = _collect_feedback(result.results)
        if feedback:
            artifacts["feedback"] = feedback

        metrics["artifacts"] = artifacts
        return metrics

    return eval_fn


def make_dspy_eval_fn_with_feedback_split(
    student: Module,
    metric: Callable,
    trainset: list[Example],
    valset: list[Example],
    num_threads: int | None,
    feedback_sample_size: int = 50,
) -> Callable:
    """Evaluation that scores on *valset* but collects feedback from *trainset*.

    This prevents overfitting: the mutation LLM sees concrete failure
    examples from trainset (what to fix), but candidates are ranked by
    valset accuracy (what generalises).  The ``combined_score`` in the
    returned dict is the valset score, while ``artifacts["feedback"]``
    contains failure analysis from a random sample of trainset.

    Only *feedback_sample_size* training examples are evaluated for
    feedback each call (default 50), since we only need
    ``_MAX_FEEDBACK_EXAMPLES`` failures.  This keeps the per-iteration
    cost at roughly ``len(valset) + feedback_sample_size`` instead of
    ``len(valset) + len(trainset)``.
    """
    import random as _random

    _rng = _random.Random(42)

    def eval_fn(candidate: dict) -> dict:
        program = build_dspy_program(student, candidate)
        threads = num_threads or 4

        # Score on valset (this determines fitness / ranking)
        val_evaluator = dspy.Evaluate(
            devset=valset, metric=metric, num_threads=threads,
        )
        val_result = val_evaluator(program)
        score = val_result.score / 100.0

        # Collect feedback from a sample of trainset (guides mutations).
        # We only need ~10 failure feedbacks, so 50 examples is plenty.
        sample = _rng.sample(trainset, min(feedback_sample_size, len(trainset)))
        train_evaluator = dspy.Evaluate(
            devset=sample, metric=metric, num_threads=threads,
        )
        train_result = train_evaluator(program)

        artifacts: dict[str, Any] = {}
        feedback = _collect_feedback(train_result.results)
        if feedback:
            artifacts["feedback"] = feedback

        return {"combined_score": score, "artifacts": artifacts}

    return eval_fn


# ---------------------------------------------------------------------------
# System Message
# ---------------------------------------------------------------------------

DSPY_SYSTEM_MESSAGE = textwrap.dedent("""\
    You are an expert tasked with iteratively improving a prompt instruction.
    Your goal is to maximize the COMBINED SCORE.

    CRITICAL FORMAT REQUIREMENT:
    The solution is a JSON object mapping predictor names to instruction strings.
    You MUST output valid JSON in a ```text code block. Example:

    ```text
    {"classify": "Your improved instruction here. Be specific and clear."}
    ```

    Do NOT output markdown, explanations, or anything other than the JSON object
    inside the code block. The JSON keys must match the original predictor names exactly.
""")


# ---------------------------------------------------------------------------
# DSPy ↔ Program Conversion
# ---------------------------------------------------------------------------


def extract_seed_candidate(student: Module) -> dict[str, str]:
    """Extract current instructions from a DSPy module."""
    return {
        name: pred.signature.instructions
        for name, pred in student.named_predictors()
    }


def build_dspy_program(student: Module, candidate: dict[str, str]) -> Module:
    """Build a DSPy module with new instructions from a candidate dict."""
    program = copy.deepcopy(student)
    for name, pred in program.named_predictors():
        if name in candidate:
            pred.signature = pred.signature.with_instructions(candidate[name])
    return program


def candidate_to_solution(candidate: dict[str, str]) -> str:
    """Serialize a candidate dict to a JSON solution string."""
    return json.dumps(candidate)


def solution_to_candidate(solution: str) -> dict[str, str]:
    """Deserialize a JSON solution string to a candidate dict."""
    return json.loads(solution)


def seed_database(
    db,
    student: Module,
    metric: Callable,
    valset: list[Example],
    num_threads: int | None,
    pareto_indices: List[int] | None = None,
) -> Program:
    """Evaluate the seed candidate and add it to the database."""
    candidate = extract_seed_candidate(student)
    solution = candidate_to_solution(candidate)

    eval_fn = make_dspy_eval_fn(
        student, metric, valset, num_threads,
        pareto_indices=pareto_indices,
    )
    result = eval_fn(candidate)
    artifacts = result.pop("artifacts", {})

    seed = Program(
        id=str(uuid.uuid4()),
        solution=solution,
        language="text",
        metrics=result,
        artifacts=artifacts,
        iteration_found=0,
    )
    db.add(seed, iteration=0)
    return seed


def best_program_to_module(db, student: Module) -> Module:
    """Convert the best program in the database to a compiled DSPy module."""
    best = db.get_best_program()
    if not best:
        return copy.deepcopy(student)

    candidate = solution_to_candidate(best.solution)
    result = build_dspy_program(student, candidate)
    result._compiled = True
    return result


def select_best_on_valset(
    db,
    student: Module,
    metric: Callable,
    valset: list[Example],
    num_threads: int | None,
    top_k: int = 5,
) -> Module:
    """Re-evaluate top-k candidates (by training fitness) on a held-out valset.

    During evolution the fitness function scores on trainset.  This function
    retrieves the *top_k* programs from the database (ranked by training
    score), re-evaluates each on *valset*, and returns the candidate that
    generalises best.  This prevents the optimizer from selecting a prompt
    that over-fits the training examples.

    If the database is empty or all candidates fail, returns a deep copy of
    the original *student* module.
    """
    top_programs = db.get_top_programs(n=top_k)
    if not top_programs:
        return copy.deepcopy(student)

    val_eval_fn = make_dspy_eval_fn(student, metric, valset, num_threads)

    best_val_score = -1.0
    best_candidate = None
    best_train_score = None

    for prog in top_programs:
        candidate = solution_to_candidate(prog.solution)
        train_score = prog.metrics.get("combined_score", 0)
        val_result = val_eval_fn(candidate)
        val_score = val_result.get("combined_score", 0)

        logger.info(
            f"  Candidate {prog.id[:8]}: "
            f"train={train_score:.4f}, val={val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_candidate = candidate
            best_train_score = train_score

    if best_candidate is None:
        return copy.deepcopy(student)

    logger.info(
        f"Selected candidate: train={best_train_score:.4f}, "
        f"val={best_val_score:.4f}"
    )

    result = build_dspy_program(student, best_candidate)
    result._compiled = True
    return result


# ---------------------------------------------------------------------------
# Async Runner
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async coroutine from sync code, handling nested event loops."""
    try:
        asyncio.get_running_loop()
        # Already in an event loop (e.g. Jupyter notebook)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Temp file cleanup helper
# ---------------------------------------------------------------------------


class TempFiles:
    """Context manager that tracks and cleans up temp files."""

    def __init__(self):
        self.paths: list[str] = []

    def add(self, path: str) -> str:
        self.paths.append(path)
        return path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for p in self.paths:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass
