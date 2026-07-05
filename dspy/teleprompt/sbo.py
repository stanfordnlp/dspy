"""
Semantic Bundle Optimization (SBO) for DSPy.

Based on: "Semantic Bundle Methods: Rigorous Prompt Optimization via Discrete-Continuous Relaxation"

SBO addresses limit cycles and catastrophic forgetting in greedy prompt optimization by maintaining
a "bundle" of historical critiques and using them to construct a cutting-plane model of the objective.
"""

import logging
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import dspy
from dspy.clients.lm import LM
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


@dataclass
class BundleEntry:
    """A single entry in the optimization bundle."""
    prompt: dict[str, str]  # Component name -> instruction text
    loss: float  # F̃(p_i) - smoothed/robust loss
    critique: str  # Textual critique c_i
    iteration: int  # When this was added
    self_score: float = 0.0  # s_i^0 = S(p_i, p_i, c_i)
    lambda_value: float = 1.0  # λ_i stored with this critique
    kind: str = "standard"  # standard, null_refinement, lite, etc.
    exact_cut_signature: Optional[str] = None  # Canonical prompt signature for exact null self-cuts
    exact_cut_loss: Optional[float] = None  # Exact self-cut target loss, if applicable
    exact_cut_center_signature: Optional[str] = None  # Center signature where the null cut was generated


@dataclass
class SBOResult:
    """Results from SBO optimization."""
    best_program: Module
    bundle: list[BundleEntry]
    best_idx: int
    val_scores: list[float]
    total_iterations: int
    num_serious_steps: int
    num_null_steps: int
    trace: dict[str, Any]


class SBOProgramCallError(Exception):
    """Wrap a program-call error with any LM calls captured before failure."""

    def __init__(self, original_error: Exception, actual_task_lm_calls: list[dict[str, Any]]):
        super().__init__(str(original_error))
        self.original_error = original_error
        self.actual_task_lm_calls = actual_task_lm_calls


@experimental(version="3.1.0")
class SemanticBundleOptimization(Teleprompter):
    """
    Semantic Bundle Optimization (SBO) - A rigorous prompt optimization framework.

    Unlike greedy methods (GEPA, OPRO) that only consider the latest critique, SBO maintains
    a bundle of historical constraints to prevent limit cycles and catastrophic forgetting.

    Key Components:
    - **Judge**: Semantic inner product Ŝ_J(p, p_ref, c) - scores alignment with critique
    - **Proposer**: Generates candidate variations addressing critique
    - **Verifier**: Filters candidates by cumulative violation against full bundle
    - **Serious/Null Steps**: Rigorous acceptance criterion for stability

    Args:
        metric: Evaluation function taking (example, prediction, trace) -> float or ScoreWithFeedback
        judge_lm: Language model for the semantic judge (default: main LM)
        proposer_lm: Language model for generating candidate prompts (default: main LM)
        critic_lm: Language model for generating critiques (default: main LM)
        num_candidates: Number of candidates to generate per iteration (N in paper)
        num_judge_samples: Number of judge samples for Monte Carlo averaging (J in paper)
        descent_param: Descent parameter m ∈ (0,1) for serious step test
        lambda_init: Initial sensitivity parameter λ_0
        lambda_min: Minimum allowed λ
        lambda_max: Maximum allowed λ
        lambda_gamma: EMA smoothing factor for λ updates
        tau_margin: Margin parameter for slack formulation in verifier
        max_iterations: Maximum optimization iterations
        max_null_steps: Maximum consecutive null steps before termination
        temperature: Temperature for LM calls
        judge_temperature: Sampling temperature for judge calls
        num_eval_samples: Number of stochastic program samples per evaluation example
        eval_temperature: Sampling temperature for robust loss program evaluations
        eval_cache: Whether robust loss program evaluations may use LM cache
        parse_failure_retries: Number of fresh-rollout retries after adapter parse errors
        parse_retry_temperature: Optional temperature for parse-error retry attempts
        max_critique_examples: Maximum evidence examples included in critique prompts
        max_critique_field_chars: Optional per-field character cap for critique prompts
        track_stats: Whether to track detailed statistics
    """

    def __init__(
        self,
        metric: Callable,
        judge_lm: Optional[LM] = None,
        proposer_lm: Optional[LM] = None,
        critic_lm: Optional[LM] = None,
        num_candidates: int = 5,
        num_judge_samples: int = 3,
        descent_param: float = 0.1,
        lambda_init: float = 1.0,
        lambda_min: float = 0.1,
        lambda_max: float = 10.0,
        lambda_gamma: float = 0.3,
        tau_margin: float = 0.5,
        bundle_size: int = 10,
        active_bundle_size: int = 3,
        watchlist_size: int = 2,
        active_tau_margin: float = 0.0,
        watchlist_tau_margin: float = 0.0,
        active_violation_tolerance: float = 0.0,
        watchlist_violation_tolerance: float = 0.0,
        lambda_stability_epsilon: float = 1e-6,
        tau_stop: float = 0.0,
        max_iterations: int = 50,
        max_null_steps: int = 5,
        temperature: float = 0.7,
        judge_temperature: float = 0.7,
        num_eval_samples: int = 1,
        eval_temperature: float = 0.7,
        eval_cache: bool = False,
        parse_failure_retries: int = 0,
        parse_retry_temperature: Optional[float] = None,
        max_critique_examples: int = 3,
        max_critique_field_chars: Optional[int] = None,
        max_bundle_critique_chars: Optional[int] = 900,
        stop_on_no_improving_candidate: bool = True,
        enable_exact_null_cuts: bool = True,
        track_stats: bool = True,
    ):
        super().__init__()
        self.metric = metric
        self.judge_lm = judge_lm
        self.proposer_lm = proposer_lm
        self.critic_lm = critic_lm
        self.num_candidates = num_candidates
        self.num_judge_samples = num_judge_samples
        self.descent_param = descent_param
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_gamma = lambda_gamma
        self.tau_margin = tau_margin
        self.bundle_size = max(1, bundle_size)
        self.active_bundle_size = max(1, active_bundle_size)
        self.watchlist_size = max(0, watchlist_size)
        self.active_tau_margin = active_tau_margin
        self.watchlist_tau_margin = watchlist_tau_margin
        self.active_violation_tolerance = active_violation_tolerance
        self.watchlist_violation_tolerance = watchlist_violation_tolerance
        self.lambda_stability_epsilon = lambda_stability_epsilon
        self.tau_stop = tau_stop
        self.max_iterations = max_iterations
        self.max_null_steps = max_null_steps
        self.temperature = temperature
        self.judge_temperature = judge_temperature
        self.num_eval_samples = max(1, num_eval_samples)
        self.eval_temperature = eval_temperature
        self.eval_cache = eval_cache
        self.parse_failure_retries = max(0, parse_failure_retries)
        self.parse_retry_temperature = parse_retry_temperature
        self.max_critique_examples = max(1, max_critique_examples)
        self.max_critique_field_chars = max_critique_field_chars
        self.max_bundle_critique_chars = max_bundle_critique_chars
        self.stop_on_no_improving_candidate = stop_on_no_improving_candidate
        self.enable_exact_null_cuts = enable_exact_null_cuts
        self.track_stats = track_stats

        self.result: Optional[SBOResult] = None
        self.trace: dict[str, Any] = {}

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
        **kwargs
    ) -> Module:
        """
        Optimize the student program using Semantic Bundle Optimization.

        Args:
            student: Program to optimize
            trainset: Training examples (used for critique generation)
            teacher: Unused (SBO doesn't use teacher)
            valset: Validation examples for evaluating candidates

        Returns:
            Optimized program
        """
        # Setup LMs with defaults
        judge_lm = self.judge_lm or dspy.settings.lm
        proposer_lm = self.proposer_lm or dspy.settings.lm
        critic_lm = self.critic_lm or dspy.settings.lm

        if valset is None or len(valset) == 0:
            raise ValueError("SBO requires a validation set for robust loss estimation")
        critique_examples = valset

        logger.info(f"Starting SBO optimization with {len(trainset)} train, {len(valset)} val examples")
        self.trace = self._new_trace(trainset, valset, judge_lm, proposer_lm, critic_lm)

        # Initialize bundle with original program
        original_prompts = self._extract_prompts(student)
        logger.info(f"\n{'='*60}")
        logger.info(f"INITIAL PROMPTS EXTRACTED:")
        for pred_name, prompt_text in original_prompts.items():
            logger.info(f"  [{pred_name}]: {repr(prompt_text)}")
        logger.info(f"{'='*60}\n")

        original_loss = self._evaluate_program(student, valset, eval_name="initial_val")
        logger.info(f"Original loss on valset: {original_loss:.4f}")

        initial_critique = self._generate_critique(
            student,
            critique_examples,
            original_prompts,
            critic_lm,
            trace_context=self.trace["initial"],
        )
        logger.info(f"\nINITIAL CRITIQUE:")
        logger.info(f"  {initial_critique}")
        logger.info(f"")
        initial_self_score_trace: dict[str, Any] = {}
        initial_self_score = self._compute_semantic_score(
            original_prompts,
            original_prompts,
            initial_critique,
            judge_lm,
            trace_context=initial_self_score_trace,
        )
        self.trace["initial"].update({
            "prompts": original_prompts,
            "loss": original_loss,
            "critique": initial_critique,
            "self_score": initial_self_score,
            "self_score_trace": initial_self_score_trace,
        })

        bundle = [BundleEntry(
            prompt=original_prompts,
            loss=original_loss,
            critique=initial_critique,
            iteration=0,
            self_score=initial_self_score,
            lambda_value=self.lambda_init,
        )]

        # Initialize center and sensitivity
        center_program = student.deepcopy()
        center_idx = 0
        lambda_current = self.lambda_init

        num_serious = 0
        num_null = 0
        consecutive_null = 0

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"SBO Iteration {iteration}/{self.max_iterations}")
            logger.info(f"Center loss: {bundle[center_idx].loss:.4f}, λ: {lambda_current:.3f}")
            logger.info(f"{'='*60}")

            old_center_entry = bundle[center_idx]
            old_center_loss = old_center_entry.loss
            active_bundle, watchlist_bundle = self._select_active_and_watchlist(bundle, center_idx)
            logger.info(f"\nActive bundle for candidate generation:")
            for entry_idx, entry in active_bundle:
                logger.info(f"  A bundle[{entry_idx}] loss={entry.loss:.4f}: {entry.critique}")
            if watchlist_bundle:
                logger.info(f"Watchlist bundle:")
                for entry_idx, entry in watchlist_bundle:
                    logger.info(f"  W bundle[{entry_idx}] loss={entry.loss:.4f}: {entry.critique}")

            iteration_trace = {
                "iteration": iteration,
                "center_idx": center_idx,
                "center_loss": old_center_loss,
                "lambda_start": lambda_current,
                "center_prompt": old_center_entry.prompt,
                "center_critique": old_center_entry.critique,
                "active_bundle": self._bundle_refs_for_trace(active_bundle),
                "watchlist_bundle": self._bundle_refs_for_trace(watchlist_bundle),
            }

            candidates = self._generate_candidates(
                center_program,
                old_center_entry.critique,
                proposer_lm,
                trace_context=iteration_trace,
                active_bundle=active_bundle,
                watchlist_bundle=watchlist_bundle,
            )

            logger.info(f"\nGenerated {len(candidates)} candidate programs")
            # Log all candidates to see what was generated
            for i, candidate in enumerate(candidates):
                cand_prompts = self._extract_prompts(candidate)
                logger.info(f"  Candidate {i+1} prompts:")
                for pred_name, prompt_text in cand_prompts.items():
                    logger.info(f"    [{pred_name}]: {repr(prompt_text)}")
                    # Check if this candidate is different from center
                    if prompt_text == bundle[center_idx].prompt.get(pred_name):
                        logger.warning(f"    ⚠ Candidate {i+1} [{pred_name}] is IDENTICAL to center!")

            # Stage 2: Score and evaluate the full finite candidate set C_k.
            candidate_records = self._score_candidate_batch(
                candidates,
                active_bundle,
                watchlist_bundle,
                lambda_current,
                judge_lm,
                trace_context=iteration_trace,
            )

            for record in candidate_records:
                candidate_loss = self._evaluate_program(
                    record["program"],
                    valset,
                    eval_name=f"iteration_{iteration}_candidate_{record['candidate_index']}_val",
                )
                predicted_improvement = old_center_loss - record["model_value"]
                positive_predicted_improvement = max(0.0, predicted_improvement)
                actual_improvement = old_center_loss - candidate_loss
                serious_threshold = self.descent_param * positive_predicted_improvement
                record.update({
                    "loss": candidate_loss,
                    "predicted_improvement": predicted_improvement,
                    "positive_predicted_improvement": positive_predicted_improvement,
                    "actual_improvement": actual_improvement,
                    "serious_step_threshold": serious_threshold,
                    "in_serious_set": (
                        actual_improvement > 0
                        and actual_improvement >= serious_threshold
                    ),
                })

            serious_records = [record for record in candidate_records if record["in_serious_set"]]
            if serious_records:
                best_record = min(
                    serious_records,
                    key=lambda record: (record["loss"], record["model_value"], record["candidate_index"]),
                )
                selection_reason = "lowest_loss_in_serious_set"
            else:
                best_record = min(
                    candidate_records,
                    key=lambda record: (record["loss"], record["model_value"], record["candidate_index"]),
                )
                selection_reason = "lowest_loss_for_null_refinement"

            best_candidate = best_record["program"]
            best_candidate_prompts = best_record["prompts"]
            candidate_loss = best_record["loss"]
            model_value = best_record["model_value"]
            model_trace = best_record["model_value_trace"]
            violation_trace = {
                "active": best_record["active_violation_trace"],
                "watchlist": best_record["watchlist_violation_trace"],
            }
            active_violation = best_record["active_violation"]
            watchlist_violation = best_record["watchlist_violation"]
            predicted_improvement = best_record["predicted_improvement"]
            positive_predicted_improvement = best_record["positive_predicted_improvement"]
            actual_improvement = best_record["actual_improvement"]
            serious_threshold = best_record["serious_step_threshold"]
            can_serious_step = bool(serious_records)

            for record in candidate_records:
                record["selection_role"] = (
                    "serious_trial"
                    if record is best_record and can_serious_step
                    else "null_refinement_trial"
                    if record is best_record
                    else "not_selected"
                )

            trace_candidate_records = [
                self._candidate_record_for_trace(record)
                for record in candidate_records
            ]
            iteration_trace["candidate_records"] = trace_candidate_records
            if "verifier" in iteration_trace:
                iteration_trace["verifier"]["selected_candidate_index"] = best_record["candidate_index"]
                iteration_trace["verifier"]["selected_model_value"] = model_value
                iteration_trace["verifier"]["selected_loss"] = candidate_loss
                iteration_trace["verifier"]["selection_reason"] = selection_reason
                by_idx = {record["candidate_index"]: record for record in trace_candidate_records}
                for candidate_trace in iteration_trace["verifier"].get("candidates", []):
                    candidate_trace.update(by_idx.get(candidate_trace["candidate_index"], {}))

            iteration_trace.update({
                "selected_candidate_prompts": best_candidate_prompts,
                "selected_candidate_index": best_record["candidate_index"],
                "selection_reason": selection_reason,
                "candidate_loss": candidate_loss,
                "model_value": model_value,
                "model_value_trace": model_trace,
                "violation_trace": violation_trace,
                "active_violation": active_violation,
                "watchlist_violation": watchlist_violation,
                "active_violation_tolerance": self.active_violation_tolerance,
                "watchlist_violation_tolerance": self.watchlist_violation_tolerance,
                "predicted_improvement": predicted_improvement,
                "positive_predicted_improvement": positive_predicted_improvement,
                "actual_improvement": actual_improvement,
                "serious_step_threshold": serious_threshold,
                "serious_set_candidate_indices": [
                    record["candidate_index"] for record in serious_records
                ],
            })

            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration} RESULTS:")
            logger.info(f"  Center loss:           {old_center_loss:.4f}")
            logger.info(f"  Candidate loss:        {candidate_loss:.4f}")
            logger.info(f"  Model value:           {model_value:.4f}")
            logger.info(f"  Predicted improvement: {predicted_improvement:.4f}")
            logger.info(f"  Positive predicted:    {positive_predicted_improvement:.4f}")
            logger.info(f"  Actual improvement:    {actual_improvement:.4f}")
            logger.info(f"  Serious threshold:     {serious_threshold:.4f}")
            logger.info(f"  Active violation:      {active_violation:.4f} (diagnostic)")
            logger.info(f"  Watchlist violation:   {watchlist_violation:.4f} (diagnostic)")
            logger.info(f"  Serious set size:      {len(serious_records)}")
            logger.info(f"\nBest candidate selected:")
            for pred_name, prompt_text in best_candidate_prompts.items():
                # Check if changed
                is_same = (prompt_text == bundle[center_idx].prompt.get(pred_name, ""))
                status = "[UNCHANGED]" if is_same else "[MODIFIED]"
                logger.info(f"  [{pred_name}] {status}: {repr(prompt_text)}")
            logger.info(f"{'='*60}\n")

            # Descent test: Serious vs Null step
            if can_serious_step:
                # SERIOUS STEP - Accept candidate as new center
                logger.info("✓ SERIOUS STEP: Accepting candidate as new center")

                center_program = best_candidate
                center_idx = len(bundle)  # Will be the new bundle entry
                num_serious += 1
                consecutive_null = 0

                # Update sensitivity λ
                lambda_before = lambda_current
                lambda_update_trace: dict[str, Any] = {}
                semantic_score = self._compute_semantic_score(
                    best_candidate_prompts,
                    old_center_entry.prompt,
                    old_center_entry.critique,
                    judge_lm,
                    trace_context=lambda_update_trace,
                )
                centered_semantic_score = semantic_score - old_center_entry.self_score

                lambda_obs = actual_improvement / (
                    abs(centered_semantic_score) + self.lambda_stability_epsilon
                )
                lambda_current = max(
                    self.lambda_min,
                    min(
                        self.lambda_max,
                        (1 - self.lambda_gamma) * lambda_current + self.lambda_gamma * lambda_obs
                    )
                )
                logger.info(f"Updated λ: {lambda_current:.3f} (observed: {lambda_obs:.3f})")

                # Generate critique for new center
                critique_trace: dict[str, Any] = {}
                critique = self._generate_critique(
                    best_candidate,
                    critique_examples,
                    best_candidate_prompts,
                    critic_lm,
                    trace_context=critique_trace,
                )
                iteration_trace["step_type"] = "serious"
                iteration_trace["lambda_update"] = {
                    "lambda_before": lambda_before,
                    "lambda_after": lambda_current,
                    "lambda_observed": lambda_obs,
                    "semantic_score": semantic_score,
                    "old_center_self_score": old_center_entry.self_score,
                    "centered_semantic_score": centered_semantic_score,
                    "lambda_stability_epsilon": self.lambda_stability_epsilon,
                    "semantic_score_trace": lambda_update_trace,
                }
                iteration_trace["critique_generation"] = critique_trace

            else:
                # NULL STEP - Reject candidate, refine model
                logger.info("✗ NULL STEP: Refinement only (candidate rejected)")
                num_null += 1
                consecutive_null += 1

                # Generate critique explaining why candidate failed
                critique_trace = {}
                critique = self._generate_failure_critique(
                    best_candidate,
                    valset,
                    best_candidate_prompts,
                    target_loss=old_center_loss,
                    critic_lm=critic_lm,
                    trace_context=critique_trace,
                )
                iteration_trace["step_type"] = "null"
                any_improving_candidate = any(
                    record["actual_improvement"] > 0 for record in candidate_records
                )
                iteration_trace["null_reason"] = (
                    "no_candidate_improved_validation_loss"
                    if not any_improving_candidate
                    else "insufficient_model_confirmed_improvement"
                )
                iteration_trace["critique_generation"] = critique_trace

            # Add to bundle (both serious and null steps)
            new_self_score_trace: dict[str, Any] = {}
            new_self_score = self._compute_semantic_score(
                best_candidate_prompts,
                best_candidate_prompts,
                critique,
                judge_lm,
                trace_context=new_self_score_trace,
            )
            entry_kind = "standard" if can_serious_step else "null_self_cut"
            exact_cut_signature = (
                self._prompt_signature(best_candidate_prompts)
                if self.enable_exact_null_cuts and not can_serious_step
                else None
            )
            exact_cut_loss = (
                candidate_loss
                if self.enable_exact_null_cuts and not can_serious_step
                else None
            )
            exact_cut_center_signature = (
                self._prompt_signature(old_center_entry.prompt)
                if self.enable_exact_null_cuts and not can_serious_step
                else None
            )
            bundle.append(BundleEntry(
                prompt=best_candidate_prompts,
                loss=candidate_loss,
                critique=critique,
                iteration=iteration,
                self_score=new_self_score,
                lambda_value=lambda_current,
                kind=entry_kind,
                exact_cut_signature=exact_cut_signature,
                exact_cut_loss=exact_cut_loss,
                exact_cut_center_signature=exact_cut_center_signature,
            ))
            if can_serious_step:
                center_idx = len(bundle) - 1
            bundle, center_idx = self._prune_bundle(bundle, center_idx)
            iteration_trace["lambda_end"] = lambda_current
            iteration_trace["bundle_entry_added"] = {
                "bundle_idx": center_idx if can_serious_step else self._find_bundle_entry_index(bundle, iteration),
                "loss": candidate_loss,
                "critique": critique,
                "prompt": best_candidate_prompts,
                "self_score": new_self_score,
                "self_score_trace": new_self_score_trace,
                "lambda_value": lambda_current,
                "kind": entry_kind,
                "exact_cut_signature": exact_cut_signature,
                "exact_cut_loss": exact_cut_loss,
                "exact_cut_center_signature": exact_cut_center_signature,
            }
            iteration_trace["bundle_size_after_prune"] = len(bundle)
            iteration_trace["center_idx_after_prune"] = center_idx
            iteration_trace["num_serious_steps_so_far"] = num_serious
            iteration_trace["num_null_steps_so_far"] = num_null
            iteration_trace["consecutive_null_steps"] = consecutive_null
            self._record_iteration(iteration_trace)

            # Termination check
            if consecutive_null >= self.max_null_steps:
                logger.info(f"Terminating: {self.max_null_steps} consecutive null steps")
                break

            # Algorithm 1 stops when no generated candidate improves validation loss.
            if (
                self.stop_on_no_improving_candidate
                and all(record["actual_improvement"] <= 0 for record in candidate_records)
            ):
                logger.info("Terminating: no candidate improved validation loss")
                break

        # Return best program from bundle
        best_idx = min(range(len(bundle)), key=lambda i: bundle[i].loss)
        best_program = self._build_program_from_prompts(student, bundle[best_idx].prompt)

        val_scores = [b.loss for b in bundle]

        self.result = SBOResult(
            best_program=best_program,
            bundle=bundle,
            best_idx=best_idx,
            val_scores=val_scores,
            total_iterations=iteration,
            num_serious_steps=num_serious,
            num_null_steps=num_null,
            trace=self._finalize_trace(bundle, best_idx, iteration, num_serious, num_null),
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"SBO Optimization Complete")
        logger.info(f"Best loss: {bundle[best_idx].loss:.4f} (iteration {bundle[best_idx].iteration})")
        logger.info(f"Serious steps: {num_serious}, Null steps: {num_null}")
        logger.info(f"{'='*60}\n")

        return best_program

    def _extract_prompts(self, program: Module) -> dict[str, str]:
        """Extract instruction prompts from all predictors in the program."""
        prompts = {}
        for pred_name, pred in program.named_predictors():
            if hasattr(pred, 'signature') and hasattr(pred.signature, 'instructions'):
                prompts[pred_name] = pred.signature.instructions or ""
        return prompts

    def _build_program_from_prompts(self, template: Module, prompts: dict[str, str]) -> Module:
        """Build a program by setting prompts in a template."""
        program = template.deepcopy()
        for pred_name, instruction in prompts.items():
            for name, pred in program.named_predictors():
                if name == pred_name:
                    if hasattr(pred, 'signature'):
                        # IMPORTANT: with_instructions() returns a NEW signature, must reassign!
                        new_signature = pred.signature.with_instructions(instruction)
                        pred.signature = new_signature
                        logger.debug(f"Applied instruction to {pred_name}: {repr(instruction)}")
        return program

    def _evaluate_program(self, program: Module, examples: list[Example], eval_name: str = "evaluation") -> float:
        """Evaluate program on examples using the metric (robust loss estimation)."""
        total_loss = 0.0
        total_observations = 0
        eval_trace = {
            "name": eval_name,
            "num_examples": len(examples),
            "num_eval_samples": self.num_eval_samples,
            "eval_temperature": self.eval_temperature,
            "eval_cache": self.eval_cache,
            "examples": [],
        }
        logger.info(
            f"Evaluating program on {len(examples)} examples "
            f"× {self.num_eval_samples} stochastic sample(s)..."
        )
        for idx, ex in enumerate(examples):
            example_trace = {
                "example_idx": idx,
                "inputs": self._safe_serialize(ex.inputs()),
                "labels": self._safe_serialize(ex.labels()),
                "samples": [],
            }
            for sample_idx in range(self.num_eval_samples):
                rollout_id = self._fresh_rollout_id() if self.eval_temperature and self.eval_temperature > 0 else None
                sample_trace = {"sample_idx": sample_idx, "rollout_id": rollout_id}
                try:
                    logger.debug(f"  Example {idx+1}/{len(examples)} sample {sample_idx+1}: Running program...")
                    eval_lm = self._evaluation_lm(rollout_id)
                    pred, task_lm_calls = self._run_program_with_lm_trace(program, ex.inputs(), eval_lm)
                    logger.debug(f"  Example {idx+1}/{len(examples)} sample {sample_idx+1}: Computing metric...")
                    score = self.metric(ex, pred, None)
                    # Convert score to loss (assuming metric returns [0,1] with 1=best)
                    loss = 1.0 - float(score)
                    total_loss += loss
                    total_observations += 1
                    sample_trace.update({
                        "actual_task_lm_calls": task_lm_calls,
                        "prediction": self._safe_serialize(pred),
                        "score": float(score),
                        "loss": loss,
                    })
                    logger.debug(
                        f"  Example {idx+1}/{len(examples)} sample {sample_idx+1}: "
                        f"score={score:.3f}, loss={loss:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"Evaluation error on example {idx+1}, sample {sample_idx+1}: {e}")
                    logger.debug(f"  Example inputs: {ex.inputs()}")
                    total_loss += 1.0  # Max penalty
                    total_observations += 1
                    task_lm_calls = e.actual_task_lm_calls if isinstance(e, SBOProgramCallError) else []
                    sample_trace.update({
                        "actual_task_lm_calls": task_lm_calls,
                        "error": repr(e.original_error if isinstance(e, SBOProgramCallError) else e),
                        "loss": 1.0,
                    })
                example_trace["samples"].append(sample_trace)
            eval_trace["examples"].append(example_trace)
        avg_loss = total_loss / total_observations if total_observations else 1.0
        eval_trace["avg_loss"] = avg_loss
        eval_trace["total_observations"] = total_observations
        self._record_evaluation(eval_trace)
        logger.info(f"Program evaluation complete: avg_loss={avg_loss:.4f}")
        return avg_loss

    def _generate_critique(
        self,
        program: Module,
        examples: list[Example],
        prompts: dict[str, str],
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate critique identifying weaknesses in the program."""
        # Sample a few failure cases
        failures = []
        skipped_parse_failures = []
        instruction_prompt_text = self._format_instruction_prompt(prompts)
        output_fields = self._program_output_fields(program)
        sampled_examples = random.sample(examples, min(self.max_critique_examples, len(examples)))
        for ex in sampled_examples:
            try:
                pred, task_lm_calls = self._run_program_with_lm_trace(program, ex.inputs())
                score = self.metric(ex, pred, None)
                if score < 0.8:  # Consider as failure
                    failures.append(
                        self._build_failure_record(
                            example=ex,
                            prediction=pred,
                            score=float(score),
                            prompts=prompts,
                            example_idx=len(failures) + 1,
                            output_fields=output_fields,
                            actual_task_lm_calls=task_lm_calls,
                    )
                )
            except Exception as e:
                original_error = e.original_error if isinstance(e, SBOProgramCallError) else e
                task_lm_calls = e.actual_task_lm_calls if isinstance(e, SBOProgramCallError) else []
                error_record = self._build_error_record(
                    example=ex,
                    error=original_error,
                    prompts=prompts,
                    example_idx=len(failures) + len(skipped_parse_failures) + 1,
                    output_fields=output_fields,
                    actual_task_lm_calls=task_lm_calls,
                )
                if self._is_parse_error(original_error):
                    skipped_parse_failures.append(error_record)
                else:
                    failures.append(error_record)

        if not failures:
            if skipped_parse_failures:
                critique = (
                    "Program calls failed to parse after retry attempts; no semantic critique was generated "
                    "from malformed outputs. Tighten the instruction so the model follows the required output schema."
                )
            else:
                critique = "The prompt is performing well on the given examples."
            if trace_context is not None:
                trace_context["critique_generation"] = {
                    "type": "standard",
                    "instruction_prompt_text": instruction_prompt_text,
                    "prompt_text": instruction_prompt_text,
                    "prompt_text_kind": "instruction_only",
                    "example_prompt_texts": [],
                    "sampled_examples": len(sampled_examples),
                    "failures": [],
                    "skipped_parse_failures": skipped_parse_failures,
                    "critique_prompt": None,
                    "response": critique,
                }
            return critique

        # Format critique generation prompt. The reusable instruction is separate
        # from example-specific runtime inputs; feedback is separate from the
        # task prompt snapshot so the trace mirrors the actual task prompt.
        task_prompt_texts = [f["task_prompt_text"] for f in failures]
        critique_task_prompt_texts = [
            self._format_task_prompt_snapshot(
                example_idx=f["example_idx"],
                instruction_prompt_text=instruction_prompt_text,
                inputs=f["inputs"],
                max_field_chars=self.max_critique_field_chars,
            )
            for f in failures
        ]
        feedback_texts = [f["feedback_text"] for f in failures]
        task_prompt_evidence_text = self._format_numbered_blocks(critique_task_prompt_texts)
        feedback_evidence_text = "\n\n".join(feedback_texts)

        critique_prompt = f"""You are an expert prompt engineer. Analyze the following instruction template and failure-specific prompt snapshots to identify the single most critical weakness.

Current Instruction Template:
{instruction_prompt_text}

Task Prompt Snapshots Sent To The Answering LM:
{task_prompt_evidence_text}

Failure Feedback For Those Snapshots:
{feedback_evidence_text}

Instructions:
1. Analyze why the prompt failed on these examples
2. Formulate a specific, actionable critique (e.g., "The prompt is too vague about output formatting")
3. Do NOT suggest a new prompt. Only state the critique.

Critique:"""

        with dspy.context(lm=lm, temperature=self.temperature):
            response = lm(critique_prompt)

        # LM returns a list of completions, get the first one
        critique = (response[0] if isinstance(response, list) else response).strip()
        if trace_context is not None:
            trace_context["critique_generation"] = {
                "type": "standard",
                "instruction_prompt_text": instruction_prompt_text,
                "prompt_text": task_prompt_evidence_text,
                "prompt_text_kind": "critic_evidence_task_prompt_snapshots",
                "critic_evidence_prompt_text": task_prompt_evidence_text,
                "task_prompt_texts": task_prompt_texts,
                "critique_task_prompt_texts": critique_task_prompt_texts,
                "example_prompt_texts": task_prompt_texts,
                "feedback_texts": feedback_texts,
                "failures": failures,
                "skipped_parse_failures": skipped_parse_failures,
                "critique_prompt": critique_prompt,
                "critic_prompt_text": critique_prompt,
                "critique_evidence_limits": {
                    "max_critique_examples": self.max_critique_examples,
                    "max_critique_field_chars": self.max_critique_field_chars,
                },
                "raw_response": self._safe_serialize(response),
                "response": critique,
                "temperature": self.temperature,
            }
        return critique

    def _generate_failure_critique(
        self,
        program: Module,
        examples: list[Example],
        prompts: dict[str, str],
        target_loss: float,
        critic_lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate critique explaining why a candidate failed to improve."""
        current_loss = self._evaluate_program(program, examples, eval_name="null_step_failure_critique_candidate")

        instruction_prompt_text = self._format_instruction_prompt(prompts)
        output_fields = self._program_output_fields(program)
        sampled_examples = random.sample(examples, min(self.max_critique_examples, len(examples)))
        evidence_records = []
        skipped_parse_failures = []
        for ex in sampled_examples:
            try:
                pred, task_lm_calls = self._run_program_with_lm_trace(program, ex.inputs())
                score = self.metric(ex, pred, None)
                evidence_records.append(
                    self._build_failure_record(
                        example=ex,
                        prediction=pred,
                        score=float(score),
                        prompts=prompts,
                        example_idx=len(evidence_records) + 1,
                        output_fields=output_fields,
                        actual_task_lm_calls=task_lm_calls,
                    )
                )
            except Exception as e:
                original_error = e.original_error if isinstance(e, SBOProgramCallError) else e
                task_lm_calls = e.actual_task_lm_calls if isinstance(e, SBOProgramCallError) else []
                error_record = self._build_error_record(
                    example=ex,
                    error=original_error,
                    prompts=prompts,
                    example_idx=len(evidence_records) + len(skipped_parse_failures) + 1,
                    output_fields=output_fields,
                    actual_task_lm_calls=task_lm_calls,
                )
                if self._is_parse_error(original_error):
                    skipped_parse_failures.append(error_record)
                else:
                    evidence_records.append(error_record)

        if not evidence_records and skipped_parse_failures:
            critique = (
                "The candidate produced outputs that failed to parse after retry attempts, so no semantic "
                "failure critique was generated. Tighten the instruction so every response follows the required "
                "output schema exactly."
            )
            if trace_context is not None:
                trace_context["failure_critique_generation"] = {
                    "instruction_prompt_text": instruction_prompt_text,
                    "prompt_text": instruction_prompt_text,
                    "prompt_text_kind": "instruction_only_parse_failures_skipped",
                    "task_prompt_texts": [],
                    "critique_task_prompt_texts": [],
                    "example_prompt_texts": [],
                    "feedback_texts": [],
                    "examples": [],
                    "skipped_parse_failures": skipped_parse_failures,
                    "current_loss": current_loss,
                    "target_loss": target_loss,
                    "critique_prompt": None,
                    "critic_prompt_text": None,
                    "response": critique,
                    "temperature": self.temperature,
                }
            return critique

        task_prompt_texts = [record["task_prompt_text"] for record in evidence_records]
        critique_task_prompt_texts = [
            self._format_task_prompt_snapshot(
                example_idx=record["example_idx"],
                instruction_prompt_text=instruction_prompt_text,
                inputs=record.get("inputs", {}),
                max_field_chars=self.max_critique_field_chars,
            )
            for record in evidence_records
        ]
        feedback_texts = [record["feedback_text"] for record in evidence_records]
        task_prompt_evidence_text = (
            self._format_numbered_blocks(critique_task_prompt_texts)
            if critique_task_prompt_texts
            else "(No examples available.)"
        )
        feedback_evidence_text = "\n\n".join(feedback_texts) if feedback_texts else "(No feedback available.)"

        critique_prompt = f"""You are an expert prompt engineer. A candidate prompt was tested but failed to improve performance.

Candidate Instruction Template:
{instruction_prompt_text}

Candidate Task Prompt Snapshots Sent To The Answering LM:
{task_prompt_evidence_text}

Failure Feedback For Those Snapshots:
{feedback_evidence_text}

Current Loss: {current_loss:.4f}
Target Loss: {target_loss:.4f}
(Lower is better. Candidate should achieve loss < {target_loss:.4f} but didn't)

Analyze why this candidate failed to improve performance. Provide a specific, actionable critique.

Critique:"""

        with dspy.context(lm=critic_lm, temperature=self.temperature):
            response = critic_lm(critique_prompt)

        # LM returns a list of completions, get the first one
        critique = (response[0] if isinstance(response, list) else response).strip()
        if trace_context is not None:
            trace_context["failure_critique_generation"] = {
                "instruction_prompt_text": instruction_prompt_text,
                "prompt_text": task_prompt_evidence_text,
                "prompt_text_kind": "critic_evidence_task_prompt_snapshots",
                "critic_evidence_prompt_text": task_prompt_evidence_text,
                "task_prompt_texts": task_prompt_texts,
                "critique_task_prompt_texts": critique_task_prompt_texts,
                "example_prompt_texts": task_prompt_texts,
                "feedback_texts": feedback_texts,
                "examples": evidence_records,
                "skipped_parse_failures": skipped_parse_failures,
                "critique_evidence_limits": {
                    "max_critique_examples": self.max_critique_examples,
                    "max_critique_field_chars": self.max_critique_field_chars,
                },
                "current_loss": current_loss,
                "target_loss": target_loss,
                "critique_prompt": critique_prompt,
                "critic_prompt_text": critique_prompt,
                "raw_response": self._safe_serialize(response),
                "response": critique,
                "temperature": self.temperature,
            }
        return critique

    def _generate_candidates(
        self,
        center_program: Module,
        critique: str,
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
        active_bundle: Optional[list[tuple[int, BundleEntry]]] = None,
        watchlist_bundle: Optional[list[tuple[int, BundleEntry]]] = None,
    ) -> list[Module]:
        """Generate N candidate variations addressing the critique (Proposer)."""
        center_prompts = self._extract_prompts(center_program)
        prompt_text = self._format_instruction_prompt(center_prompts)
        active_bundle = active_bundle or []
        watchlist_bundle = watchlist_bundle or []
        active_text = self._format_bundle_for_proposer(active_bundle) if active_bundle else f"A0: {critique}"
        watchlist_text = self._format_bundle_for_proposer(watchlist_bundle) if watchlist_bundle else "(none)"

        proposer_prompt = f"""You are an intelligent local prompt proposer. Generate {self.num_candidates} variations of the prompt that address the active critique bundle while preserving the original task intent and avoiding regressions on the watchlist.

Current Prompt:
{prompt_text}

Active Critique Bundle:
{active_text}

Watchlist of Non-Active Critiques:
{watchlist_text}

Constraints:
1. Local edits only: Do not rewrite the entire prompt from scratch. Keep structure similar.
2. Focus: Address the active critiques directly, prioritizing high-loss or recent critiques. Use the watchlist only as a regression check.
3. Diversity: Generate {self.num_candidates} distinct variations

Output Format: Return exactly {self.num_candidates} candidate prompts. Each candidate should start with "CANDIDATE N:" on its own line, followed by the improved prompt text.

Example format:
CANDIDATE 1:
Your first improved prompt text here
CANDIDATE 2:
Your second improved prompt text here
CANDIDATE 3:
Your third improved prompt text here

Candidates:"""

        with dspy.context(lm=lm, temperature=self.temperature):
            response = lm(proposer_prompt)

        # LM returns a list of completions, get the first one
        response_text = response[0] if isinstance(response, list) else response

        logger.info(f"\n{'='*60}")
        logger.info(f"PROPOSER RAW RESPONSE:")
        logger.info(f"{response_text}")
        logger.info(f"{'='*60}\n")

        candidates = []
        parsed_candidates = []
        candidate_texts, parse_warnings = self._parse_candidate_response(response_text)

        for candidate_text in candidate_texts[: self.num_candidates]:
            logger.info(f"Parsed candidate {len(candidates)+1}: {repr(candidate_text)}")
            new_prompts = {k: candidate_text for k in center_prompts.keys()}
            parsed_candidates.append({
                "candidate_index": len(candidates),
                "candidate_text": candidate_text,
                "prompts": new_prompts,
            })
            candidates.append(self._build_program_from_prompts(center_program, new_prompts))

        # If parsing failed completely, try to extract at least some variations
        if len(candidates) == 0:
            logger.warning("Failed to parse any candidates from LM response, using center program")
            parsed_candidates.append({
                "candidate_index": 0,
                "candidate_text": None,
                "prompts": center_prompts,
                "fallback": "center_program",
            })
            candidates.append(center_program.deepcopy())
        
        # Fill remaining slots with center program copies if needed
        while len(candidates) < self.num_candidates:
            logger.warning(f"Only parsed {len(candidates)} candidates, padding with center program")
            parsed_candidates.append({
                "candidate_index": len(candidates),
                "candidate_text": None,
                "prompts": center_prompts,
                "fallback": "center_program_padding",
            })
            candidates.append(center_program.deepcopy())

        if trace_context is not None:
            trace_context["proposer"] = {
                "num_candidates_requested": self.num_candidates,
                "center_prompts": center_prompts,
                "instruction_prompt_text": prompt_text,
                "prompt_text": prompt_text,
                "runtime_inputs_attached": False,
                "critique": critique,
                "active_bundle": self._bundle_refs_for_trace(active_bundle),
                "watchlist_bundle": self._bundle_refs_for_trace(watchlist_bundle),
                "active_bundle_text": active_text,
                "watchlist_text": watchlist_text,
                "proposer_prompt": proposer_prompt,
                "raw_response": response_text,
                "temperature": self.temperature,
                "parse_warnings": parse_warnings,
                "parsed_candidates": parsed_candidates[:self.num_candidates],
            }
        return candidates[:self.num_candidates]

    def _parse_candidate_response(self, response_text: str) -> tuple[list[str], list[str]]:
        """Parse proposer output while avoiding accidental matches like 'CANDIDATE -3'."""
        warnings = []
        texts = []

        candidate_pattern = re.compile(
            r"(?ms)^\s*CANDIDATE\s+(\d+)\s*:\s*(.*?)(?=^\s*CANDIDATE\s+\d+\s*:|\Z)"
        )
        for match in candidate_pattern.finditer(response_text):
            body = match.group(2).strip()
            if body:
                texts.append(body)

        if texts:
            if re.search(r"(?m)^\s*CANDIDATE\s+-\d+\s*:", response_text):
                warnings.append("ignored_negative_candidate_marker")
            return texts, warnings

        numbered_pattern = re.compile(
            r"(?ms)^\s*(\d+)[\.\)]\s+(.*?)(?=^\s*\d+[\.\)]\s+|\Z)"
        )
        for match in numbered_pattern.finditer(response_text):
            body = match.group(2).strip()
            if body:
                texts.append(body)

        if texts:
            warnings.append("parsed_numbered_list_fallback")
            return texts, warnings

        warnings.append("no_candidates_parsed")
        return [], warnings

    def _format_instruction_prompt(self, prompts: dict[str, str]) -> str:
        """Format predictor instructions without any example-specific variables."""
        if len(prompts) == 1:
            return next(iter(prompts.values()))
        return "\n\n".join([f"Predictor `{k}` instruction:\n{v}" for k, v in prompts.items()])

    def _build_failure_record(
        self,
        *,
        example: Example,
        prediction: Any,
        score: float,
        prompts: dict[str, str],
        example_idx: int,
        output_fields: set[str] | None = None,
        actual_task_lm_calls: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Build a structured failure record with runtime variables attached."""
        inputs = self._example_fields(example.inputs())
        expected = self._filter_fields(self._example_fields(example.labels()), output_fields)
        predicted = self._prediction_fields(prediction)
        task_prompt_text = self._format_task_prompt_snapshot(
            example_idx=example_idx,
            instruction_prompt_text=self._format_instruction_prompt(prompts),
            inputs=inputs,
        )
        feedback_text = self._format_failure_feedback(
            example_idx=example_idx,
            expected=expected,
            predicted=predicted,
            score=score,
        )
        return {
            "example_idx": example_idx,
            "inputs": inputs,
            "expected": expected,
            "predicted": predicted,
            "score": score,
            "task_prompt_text": task_prompt_text,
            "prompt_text": task_prompt_text,
            "feedback_text": feedback_text,
            "actual_task_lm_calls": actual_task_lm_calls or [],
        }

    def _build_error_record(
        self,
        *,
        example: Example,
        error: Exception,
        prompts: dict[str, str],
        example_idx: int,
        output_fields: set[str] | None = None,
        actual_task_lm_calls: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Build a structured evidence record when the program call itself fails."""
        inputs = self._example_fields(example.inputs())
        expected = self._filter_fields(self._example_fields(example.labels()), output_fields)
        task_prompt_text = self._format_task_prompt_snapshot(
            example_idx=example_idx,
            instruction_prompt_text=self._format_instruction_prompt(prompts),
            inputs=inputs,
        )
        feedback_text = f"""Example {example_idx} Feedback:
Expected Outputs:
{self._format_fields(expected)}

Program Error:
{repr(error)}

Score: 0.00"""
        return {
            "example_idx": example_idx,
            "inputs": inputs,
            "expected": expected,
            "predicted": None,
            "score": 0.0,
            "error": repr(error),
            "task_prompt_text": task_prompt_text,
            "prompt_text": task_prompt_text,
            "feedback_text": feedback_text,
            "actual_task_lm_calls": actual_task_lm_calls or [],
        }

    def _format_task_prompt_snapshot(
        self,
        *,
        example_idx: int,
        instruction_prompt_text: str,
        inputs: dict[str, Any],
        max_field_chars: Optional[int] = None,
    ) -> str:
        """Format the task prompt shape: instruction plus runtime inputs only."""
        return f"""{instruction_prompt_text}

{self._format_fields(inputs, max_value_chars=max_field_chars)}"""

    def _format_numbered_blocks(self, blocks: list[str]) -> str:
        return "\n\n".join([f"Example {idx + 1}:\n{block}" for idx, block in enumerate(blocks)])

    def _format_failure_feedback(
        self,
        *,
        example_idx: int,
        expected: dict[str, Any],
        predicted: Any,
        score: float,
    ) -> str:
        """Format labels/predictions as critic feedback, not as task prompt text."""
        return f"""Example {example_idx} Feedback:
Expected Outputs:
{self._format_fields(expected)}

Model Prediction:
{self._format_fields(predicted)}

Score: {score:.2f}"""

    def _format_fields(self, fields: Any, max_value_chars: Optional[int] = None) -> str:
        if isinstance(fields, dict):
            if not fields:
                return "(none)"
            return "\n".join([
                f"{key}: {self._clip_for_prompt(self._safe_serialize(value), max_value_chars)!r}"
                for key, value in fields.items()
            ])
        return repr(self._clip_for_prompt(self._safe_serialize(fields), max_value_chars))

    def _clip_for_prompt(self, value: Any, max_chars: Optional[int]) -> Any:
        if max_chars is None:
            return value
        text = repr(value)
        if len(text) <= max_chars:
            return value
        return text[:max_chars] + f"... [truncated for critic prompt; original length={len(text)} chars]"

    def _example_fields(self, example: Example) -> dict[str, Any]:
        if hasattr(example, "toDict"):
            return self._safe_serialize(example.toDict())
        return self._safe_serialize(dict(example))

    def _prediction_fields(self, prediction: Any) -> Any:
        if hasattr(prediction, "toDict"):
            return self._safe_serialize(prediction.toDict())
        if isinstance(prediction, dict):
            return self._safe_serialize(prediction)
        try:
            return self._safe_serialize(dict(prediction))
        except Exception:
            return self._safe_serialize(prediction)

    def _program_output_fields(self, program: Module) -> set[str]:
        output_fields = set()
        for _, pred in program.named_predictors():
            if hasattr(pred, "signature") and hasattr(pred.signature, "output_fields"):
                output_fields.update(pred.signature.output_fields.keys())
        return output_fields

    def _filter_fields(self, fields: dict[str, Any], allowed_fields: set[str] | None) -> dict[str, Any]:
        if not allowed_fields:
            return fields
        filtered = {key: value for key, value in fields.items() if key in allowed_fields}
        return filtered or fields

    def _compute_semantic_score(
        self,
        candidate_prompts: dict[str, str],
        reference_prompts: dict[str, str],
        critique: str,
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Compute smoothed semantic score Ŝ_J(p, p_ref, c).

        Returns value in [-1, 1] indicating alignment with critique.
        """
        scores = []
        sample_details = []

        for sample_idx in range(self.num_judge_samples):
            score, detail = self._judge_single(
                candidate_prompts,
                reference_prompts,
                critique,
                lm,
                sample_idx=sample_idx,
                return_detail=True,
            )
            scores.append(score)
            sample_details.append(detail)

        mean_score = sum(scores) / len(scores)
        if trace_context is not None:
            trace_context.update({
                "candidate_prompts": candidate_prompts,
                "reference_prompts": reference_prompts,
                "critique": critique,
                "num_judge_samples": self.num_judge_samples,
                "judge_temperature": self.judge_temperature,
                "scores": scores,
                "mean_score": mean_score,
                "samples": sample_details,
            })
        return mean_score

    def _judge_single(
        self,
        candidate_prompts: dict[str, str],
        reference_prompts: dict[str, str],
        critique: str,
        lm: LM,
        sample_idx: int = 0,
        return_detail: bool = False,
    ) -> float | tuple[float, dict[str, Any]]:
        """Single judge evaluation (returns [-1, 1])."""
        candidate_text = "\n\n".join([f"{k}: {v}" for k, v in candidate_prompts.items()])
        reference_text = "\n\n".join([f"{k}: {v}" for k, v in reference_prompts.items()])

        judge_prompt = f"""You are an objective optimization judge. Quantify how well the Candidate Prompt addresses the Critique relative to the Reference Prompt.

Reference Prompt:
{reference_text}

Critique: {critique}

Candidate Prompt:
{candidate_text}

Scoring Rubric:
• +1.0 (Strong Alignment): Candidate completely resolves the issue
• +0.5 (Weak Alignment): Candidate partially fixes the issue
• 0.0 (Orthogonal): Candidate ignores the critique
• -0.5 (Weak Regression): Candidate slightly worsens the issue
• -1.0 (Strong Regression): Candidate explicitly violates the critique

Output ONLY a single number between -1.0 and 1.0. No explanation.

Score:"""

        rollout_id = self._fresh_rollout_id() if self.judge_temperature and self.judge_temperature > 0 else None
        with dspy.context(lm=lm, temperature=self.judge_temperature):
            response = lm(
                judge_prompt,
                temperature=self.judge_temperature,
                rollout_id=rollout_id,
                cache=False,
            )

        # LM returns a list of completions, get the first one
        response_text = (response[0] if isinstance(response, list) else response).strip()

        # Parse score
        try:
            score = float(response_text)
            parsed_score = max(-1.0, min(1.0, score))
        except:
            logger.warning(f"Failed to parse judge score: {response_text}")
            parsed_score = 0.0

        detail = {
            "sample_idx": sample_idx,
            "rollout_id": rollout_id,
            "judge_prompt": judge_prompt,
            "raw_response": self._safe_serialize(response),
            "response_text": response_text,
            "score": parsed_score,
            "temperature": self.judge_temperature,
            "cache": False,
        }
        if return_detail:
            return parsed_score, detail
        return parsed_score

    def _score_candidate_batch(
        self,
        candidates: list[Module],
        active_bundle: list[tuple[int, BundleEntry]],
        watchlist_bundle: list[tuple[int, BundleEntry]],
        lambda_current: float,
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Score every generated candidate against the active bundle model."""
        logger.info(f"\n{'='*60}")
        logger.info(
            f"VERIFIER: Scoring {len(candidates)} candidates against "
            f"{len(active_bundle)} active and {len(watchlist_bundle)} watchlist entries"
        )
        logger.info(
            f"  (This requires about "
            f"{len(candidates) * (len(active_bundle) + len(watchlist_bundle)) * self.num_judge_samples} "
            "judge LM calls)"
        )
        logger.info(f"{'='*60}")

        verifier_trace = {
            "lambda_current": lambda_current,
            "tau_margin": self.tau_margin,
            "active_tau_margin": self.active_tau_margin,
            "watchlist_tau_margin": self.watchlist_tau_margin,
            "num_candidates": len(candidates),
            "active_bundle_size": len(active_bundle),
            "watchlist_size": len(watchlist_bundle),
            "active_bundle": self._bundle_refs_for_trace(active_bundle),
            "watchlist_bundle": self._bundle_refs_for_trace(watchlist_bundle),
            "selection_policy": "paper_loss_first_candidate_set",
            "candidates": [],
        }

        records = []
        for idx, candidate in enumerate(candidates):
            candidate_prompts = self._extract_prompts(candidate)
            model_trace: dict[str, Any] = {}
            model_value = self._compute_model_value(
                candidate_prompts,
                active_bundle,
                lambda_current,
                lm,
                trace_context=model_trace,
            )
            active_violation_trace: dict[str, Any] = {}
            active_violation = self._compute_bundle_violation(
                candidate_prompts,
                active_bundle,
                self.active_tau_margin,
                lm,
                trace_context=active_violation_trace,
            )
            watchlist_violation_trace: dict[str, Any] = {}
            watchlist_violation = self._compute_bundle_violation(
                candidate_prompts,
                watchlist_bundle,
                self.watchlist_tau_margin,
                lm,
                trace_context=watchlist_violation_trace,
            )
            total_violation = active_violation + watchlist_violation
            record = {
                "candidate_index": idx,
                "program": candidate,
                "prompts": candidate_prompts,
                "model_value": model_value,
                "model_value_trace": model_trace,
                "active_violation": active_violation,
                "watchlist_violation": watchlist_violation,
                "total_violation": total_violation,
                "active_violation_trace": active_violation_trace,
                "watchlist_violation_trace": watchlist_violation_trace,
                # Keep the historical key name for existing trace readers/tests.
                "bundle_scores": active_violation_trace.get("entries", []),
                "active_scores": active_violation_trace.get("entries", []),
                "watchlist_scores": watchlist_violation_trace.get("entries", []),
            }
            records.append(record)
            verifier_trace["candidates"].append(self._candidate_record_for_trace(record))
            logger.info(
                f"  Candidate {idx+1}: model={model_value:.4f}, "
                f"active_violation={active_violation:.4f}, watchlist_violation={watchlist_violation:.4f}"
            )

        semantic_ranked = sorted(records, key=lambda item: (item["model_value"], item["total_violation"]))
        for rank, record in enumerate(semantic_ranked, start=1):
            record["semantic_rank"] = rank

        if trace_context is not None:
            trace_context["verifier"] = verifier_trace
        return records

    def _candidate_record_for_trace(self, record: dict[str, Any]) -> dict[str, Any]:
        """Remove non-serializable candidate program objects from a candidate record."""
        return {
            key: value
            for key, value in record.items()
            if key != "program"
        }

    def _select_best_candidate(
        self,
        candidates: list[Module],
        active_bundle: list[tuple[int, BundleEntry]],
        watchlist_bundle: list[tuple[int, BundleEntry]],
        lambda_current: float,
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> tuple[Module, dict[str, str]]:
        """
        Select best candidate by approximately minimizing the active semantic model.

        Returns: (best_program, best_prompts)
        """
        logger.info(f"\n{'='*60}")
        logger.info(
            f"VERIFIER: Evaluating {len(candidates)} candidates against "
            f"{len(active_bundle)} active and {len(watchlist_bundle)} watchlist entries"
        )
        logger.info(
            f"  (This requires about "
            f"{len(candidates) * (len(active_bundle) + len(watchlist_bundle)) * self.num_judge_samples} "
            "judge LM calls)"
        )
        logger.info(f"{'='*60}")

        best_model_value = float('inf')
        best_total_violation = float('inf')
        best_candidate = candidates[0]
        best_prompts = self._extract_prompts(candidates[0])
        verifier_trace = {
            "lambda_current": lambda_current,
            "tau_margin": self.tau_margin,
            "active_tau_margin": self.active_tau_margin,
            "watchlist_tau_margin": self.watchlist_tau_margin,
            "num_candidates": len(candidates),
            "active_bundle_size": len(active_bundle),
            "watchlist_size": len(watchlist_bundle),
            "active_bundle": self._bundle_refs_for_trace(active_bundle),
            "watchlist_bundle": self._bundle_refs_for_trace(watchlist_bundle),
            "candidates": [],
        }

        for idx, candidate in enumerate(candidates):
            candidate_prompts = self._extract_prompts(candidate)

            # Compute the paper's active max-envelope model and diagnostic violations.
            model_trace: dict[str, Any] = {}
            model_value = self._compute_model_value(
                candidate_prompts,
                active_bundle,
                lambda_current,
                lm,
                trace_context=model_trace,
            )
            active_violation_trace: dict[str, Any] = {}
            active_violation = self._compute_bundle_violation(
                candidate_prompts,
                active_bundle,
                self.active_tau_margin,
                lm,
                trace_context=active_violation_trace,
            )
            watchlist_violation_trace: dict[str, Any] = {}
            watchlist_violation = self._compute_bundle_violation(
                candidate_prompts,
                watchlist_bundle,
                self.watchlist_tau_margin,
                lm,
                trace_context=watchlist_violation_trace,
            )
            total_violation = active_violation + watchlist_violation
            candidate_trace = {
                "candidate_index": idx,
                "prompts": candidate_prompts,
                "model_value": model_value,
                "model_value_trace": model_trace,
                "active_violation": active_violation,
                "watchlist_violation": watchlist_violation,
                "total_violation": total_violation,
                # Keep the historical key name for existing trace readers/tests.
                "bundle_scores": active_violation_trace.get("entries", []),
                "active_scores": active_violation_trace.get("entries", []),
                "watchlist_scores": watchlist_violation_trace.get("entries", []),
            }

            logger.info(
                f"  Candidate {idx+1}: model={model_value:.4f}, "
                f"active_violation={active_violation:.4f}, watchlist_violation={watchlist_violation:.4f}"
            )
            for item in candidate_trace["active_scores"]:
                logger.info(
                    f"    vs active bundle[{item['bundle_idx']}]: "
                    f"centered={item['centered_score']:.3f}, violation={item['violation']:.3f}"
                )

            is_better = (
                model_value < best_model_value
                or (model_value == best_model_value and total_violation < best_total_violation)
            )
            if is_better:
                best_model_value = model_value
                best_total_violation = total_violation
                best_candidate = candidate
                best_prompts = candidate_prompts
            verifier_trace["candidates"].append(candidate_trace)

        logger.info(
            f"\n✓ Selected candidate with model value: {best_model_value:.4f} "
            f"(total violation: {best_total_violation:.4f})"
        )
        logger.info(f"{'='*60}\n")
        verifier_trace["selected_candidate_index"] = next(
            (
                item["candidate_index"]
                for item in verifier_trace["candidates"]
                if item["prompts"] == best_prompts
            ),
            None,
        )
        verifier_trace["selected_model_value"] = best_model_value
        verifier_trace["selected_violation"] = best_total_violation
        if trace_context is not None:
            trace_context["verifier"] = verifier_trace
        return best_candidate, best_prompts

    def _compute_model_value(
        self,
        prompts: dict[str, str],
        bundle: list[tuple[int, BundleEntry]] | list[BundleEntry],
        lambda_current: float,
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Compute M_k(p) = max_i {F̃_i - λ_i · [Ŝ_J(p,p_i,c_i)-s_i^0]}.
        """
        model_value = float('-inf')
        terms = []

        normalized_bundle = self._normalize_bundle_refs(bundle)
        if not normalized_bundle:
            model_value = float("inf")

        for entry_idx, entry in normalized_bundle:
            score_trace: dict[str, Any] = {}
            score = self._compute_semantic_score(
                prompts,
                entry.prompt,
                entry.critique,
                lm,
                trace_context=score_trace,
            )
            centered_score = score - entry.self_score
            lambda_value = entry.lambda_value if entry.lambda_value is not None else lambda_current
            value = entry.loss - lambda_value * centered_score
            exact_cut_applied = False
            if (
                self.enable_exact_null_cuts
                and entry.exact_cut_signature is not None
                and entry.exact_cut_loss is not None
                and self._prompt_signature(prompts) == entry.exact_cut_signature
            ):
                value = max(value, entry.exact_cut_loss)
                exact_cut_applied = True
            model_value = max(model_value, value)
            terms.append({
                "bundle_idx": entry_idx,
                "bundle_iteration": entry.iteration,
                "bundle_loss": entry.loss,
                "semantic_score": score,
                "self_score": entry.self_score,
                "centered_score": centered_score,
                "lambda_value": lambda_value,
                "term_value": value,
                "exact_cut_applied": exact_cut_applied,
                "exact_cut_loss": entry.exact_cut_loss,
                "semantic_score_trace": score_trace,
            })

        if trace_context is not None:
            trace_context.update({
                "candidate_prompts": prompts,
                "lambda_current": lambda_current,
                "model_kind": "centered_active_bundle_max_envelope",
                "terms": terms,
                "model_value": model_value,
            })
        return model_value

    def _compute_bundle_violation(
        self,
        prompts: dict[str, str],
        bundle: list[tuple[int, BundleEntry]] | list[BundleEntry],
        margin: float,
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> float:
        """Compute sum_i max(0, tau - centered_score_i) for a bundle subset."""
        total = 0.0
        entries = []
        for entry_idx, entry in self._normalize_bundle_refs(bundle):
            score_trace: dict[str, Any] = {}
            raw_score = self._compute_semantic_score(
                prompts,
                entry.prompt,
                entry.critique,
                lm,
                trace_context=score_trace,
            )
            centered_score = raw_score - entry.self_score
            violation = max(0.0, margin - centered_score)
            total += violation
            entries.append({
                "bundle_idx": entry_idx,
                "bundle_iteration": entry.iteration,
                "bundle_loss": entry.loss,
                "critique": entry.critique,
                "raw_score": raw_score,
                "self_score": entry.self_score,
                "centered_score": centered_score,
                "margin": margin,
                "violation": violation,
                "semantic_score_trace": score_trace,
            })
        if trace_context is not None:
            trace_context.update({
                "candidate_prompts": prompts,
                "margin": margin,
                "total_violation": total,
                "entries": entries,
            })
        return total

    def _select_active_and_watchlist(
        self,
        bundle: list[BundleEntry],
        center_idx: int,
    ) -> tuple[list[tuple[int, BundleEntry]], list[tuple[int, BundleEntry]]]:
        """Select a compact active bundle and watchlist from stored critiques."""
        indexed = list(enumerate(bundle))
        center = [(center_idx, bundle[center_idx])] if 0 <= center_idx < len(bundle) else []
        remaining = [(idx, entry) for idx, entry in indexed if idx != center_idx]
        # Prefer high-loss and recent critiques for active constraints.
        remaining_sorted = sorted(
            remaining,
            key=lambda item: (item[1].loss, item[1].iteration),
            reverse=True,
        )
        active = (center + remaining_sorted)[: self.active_bundle_size]
        active_ids = {idx for idx, _ in active}
        watchlist_pool = [(idx, entry) for idx, entry in remaining_sorted if idx not in active_ids]
        watchlist = watchlist_pool[: self.watchlist_size]
        return active, watchlist

    def _normalize_bundle_refs(
        self,
        bundle: list[tuple[int, BundleEntry]] | list[BundleEntry],
    ) -> list[tuple[int, BundleEntry]]:
        if not bundle:
            return []
        first = bundle[0]
        if isinstance(first, tuple):
            return bundle  # type: ignore[return-value]
        return list(enumerate(bundle))  # type: ignore[arg-type]

    def _bundle_refs_for_trace(self, bundle: list[tuple[int, BundleEntry]]) -> list[dict[str, Any]]:
        return [
            {
                "bundle_idx": idx,
                "iteration": entry.iteration,
                "loss": entry.loss,
                "critique": entry.critique,
                "self_score": entry.self_score,
                "lambda_value": entry.lambda_value,
                "kind": entry.kind,
                "prompt": entry.prompt,
                "exact_cut_signature": entry.exact_cut_signature,
                "exact_cut_loss": entry.exact_cut_loss,
                "exact_cut_center_signature": entry.exact_cut_center_signature,
            }
            for idx, entry in bundle
        ]

    def _format_bundle_for_proposer(self, bundle: list[tuple[int, BundleEntry]]) -> str:
        if not bundle:
            return "(none)"
        lines = []
        for local_idx, (entry_idx, entry) in enumerate(bundle, start=1):
            critique = self._clip_for_prompt(entry.critique, self.max_bundle_critique_chars)
            lines.append(
                f"{local_idx}. id=bundle[{entry_idx}], iteration={entry.iteration}, "
                f"loss={entry.loss:.4f}, critique={critique}"
            )
        return "\n".join(lines)

    def _prune_bundle(
        self,
        bundle: list[BundleEntry],
        center_idx: int,
    ) -> tuple[list[BundleEntry], int]:
        """Keep the center plus recent/high-loss entries up to bundle_size."""
        if len(bundle) <= self.bundle_size:
            return bundle, center_idx
        keep = {center_idx}
        candidates = [
            (idx, entry)
            for idx, entry in enumerate(bundle)
            if idx != center_idx
        ]
        candidates = sorted(candidates, key=lambda item: (item[1].iteration, item[1].loss), reverse=True)
        keep.update(idx for idx, _ in candidates[: self.bundle_size - 1])
        kept_pairs = [(idx, entry) for idx, entry in enumerate(bundle) if idx in keep]
        new_center_idx = next(new_idx for new_idx, (old_idx, _) in enumerate(kept_pairs) if old_idx == center_idx)
        return [entry for _, entry in kept_pairs], new_center_idx

    def _find_bundle_entry_index(self, bundle: list[BundleEntry], iteration: int) -> Optional[int]:
        for idx, entry in enumerate(bundle):
            if entry.iteration == iteration:
                return idx
        return None

    def _new_trace(
        self,
        trainset: list[Example],
        valset: list[Example],
        judge_lm: LM,
        proposer_lm: LM,
        critic_lm: LM,
    ) -> dict[str, Any]:
        """Create an SBO trace container."""
        return {
            "trace_version": 1,
            "algorithm": "sbo",
            "config": {
                "num_candidates": self.num_candidates,
                "num_judge_samples": self.num_judge_samples,
                "descent_param": self.descent_param,
                "lambda_init": self.lambda_init,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
                "lambda_gamma": self.lambda_gamma,
                "tau_margin": self.tau_margin,
                "bundle_size": self.bundle_size,
                "active_bundle_size": self.active_bundle_size,
                "watchlist_size": self.watchlist_size,
                "active_tau_margin": self.active_tau_margin,
                "watchlist_tau_margin": self.watchlist_tau_margin,
                "active_violation_tolerance": self.active_violation_tolerance,
                "watchlist_violation_tolerance": self.watchlist_violation_tolerance,
                "lambda_stability_epsilon": self.lambda_stability_epsilon,
                "tau_stop": self.tau_stop,
                "max_iterations": self.max_iterations,
                "max_null_steps": self.max_null_steps,
                "temperature": self.temperature,
                "judge_temperature": self.judge_temperature,
                "num_eval_samples": self.num_eval_samples,
                "eval_temperature": self.eval_temperature,
                "eval_cache": self.eval_cache,
                "parse_failure_retries": self.parse_failure_retries,
                "parse_retry_temperature": self.parse_retry_temperature,
                "max_critique_examples": self.max_critique_examples,
                "max_critique_field_chars": self.max_critique_field_chars,
                "max_bundle_critique_chars": self.max_bundle_critique_chars,
                "stop_on_no_improving_candidate": self.stop_on_no_improving_candidate,
                "enable_exact_null_cuts": self.enable_exact_null_cuts,
                "track_stats": self.track_stats,
            },
            "dataset": {
                "train_size": len(trainset),
                "val_size": len(valset),
                "input_fields": self._infer_input_fields(trainset or valset),
                "critique_examples_source": "valset",
            },
            "lms": {
                "judge": self._lm_trace(judge_lm),
                "proposer": self._lm_trace(proposer_lm),
                "critic": self._lm_trace(critic_lm),
            },
            "initial": {},
            "evaluations": [],
            "iterations": [],
            "final": {},
        }

    def _record_evaluation(self, eval_trace: dict[str, Any]) -> None:
        if self.track_stats and self.trace:
            self.trace.setdefault("evaluations", []).append(eval_trace)

    def _record_iteration(self, iteration_trace: dict[str, Any]) -> None:
        if self.track_stats and self.trace:
            self.trace.setdefault("iterations", []).append(iteration_trace)

    def _finalize_trace(
        self,
        bundle: list[BundleEntry],
        best_idx: int,
        total_iterations: int,
        num_serious: int,
        num_null: int,
    ) -> dict[str, Any]:
        if not self.trace:
            return {}
        self.trace["final"] = {
            "best_idx": best_idx,
            "best_iteration": bundle[best_idx].iteration,
            "best_loss": bundle[best_idx].loss,
            "best_prompt": bundle[best_idx].prompt,
            "total_iterations": total_iterations,
            "num_serious_steps": num_serious,
            "num_null_steps": num_null,
            "bundle": [
                {
                    "bundle_idx": idx,
                    "iteration": entry.iteration,
                    "loss": entry.loss,
                    "critique": entry.critique,
                    "self_score": entry.self_score,
                    "lambda_value": entry.lambda_value,
                    "kind": entry.kind,
                    "prompt": entry.prompt,
                    "exact_cut_signature": entry.exact_cut_signature,
                    "exact_cut_loss": entry.exact_cut_loss,
                    "exact_cut_center_signature": entry.exact_cut_center_signature,
                }
                for idx, entry in enumerate(bundle)
            ],
        }
        return self._safe_serialize(self.trace)

    def _evaluation_lm(self, rollout_id: int | None) -> LM:
        """Return an LM configured for one stochastic robust-loss sample."""
        lm = dspy.settings.lm
        if hasattr(lm, "copy"):
            return lm.copy(
                rollout_id=rollout_id,
                temperature=self.eval_temperature,
                cache=self.eval_cache,
            )
        return lm

    def _run_program_with_lm_trace(
        self,
        program: Module,
        inputs: Example | dict[str, Any],
        lm: Optional[LM] = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Run a DSPy program and return actual answering-LM history entries."""
        base_lm = lm or dspy.settings.lm
        all_task_lm_calls = []
        last_error = None

        for attempt_idx in range(self.parse_failure_retries + 1):
            task_lm = self._task_lm_for_program_attempt(base_lm, attempt_idx)
            history = getattr(task_lm, "history", None)
            history_start = len(history) if isinstance(history, list) else None

            try:
                with dspy.context(lm=task_lm):
                    prediction = program(**inputs)
                task_lm_calls = self._history_entries_since(task_lm, history_start, attempt_idx=attempt_idx)
                all_task_lm_calls.extend(task_lm_calls)
                return prediction, all_task_lm_calls
            except Exception as e:
                task_lm_calls = self._history_entries_since(task_lm, history_start, attempt_idx=attempt_idx)
                all_task_lm_calls.extend(task_lm_calls)
                last_error = e
                if not self._is_parse_error(e) or attempt_idx >= self.parse_failure_retries:
                    raise SBOProgramCallError(e, all_task_lm_calls) from e
                logger.info(
                    "Program output failed to parse; retrying task LM call "
                    f"({attempt_idx + 1}/{self.parse_failure_retries})"
                )

        raise SBOProgramCallError(last_error or RuntimeError("Program call failed"), all_task_lm_calls)

    def _task_lm_for_program_attempt(self, base_lm: LM, attempt_idx: int) -> LM:
        if attempt_idx == 0 or not hasattr(base_lm, "copy"):
            return base_lm
        base_temperature = getattr(base_lm, "kwargs", {}).get("temperature")
        retry_temperature = self.parse_retry_temperature
        if retry_temperature is None:
            retry_temperature = base_temperature if base_temperature is not None else self.eval_temperature
        return base_lm.copy(
            rollout_id=self._fresh_rollout_id(),
            temperature=retry_temperature,
            cache=False,
        )

    def _history_entries_since(
        self,
        task_lm: LM,
        history_start: int | None,
        attempt_idx: int = 0,
    ) -> list[dict[str, Any]]:
        if history_start is None or not isinstance(getattr(task_lm, "history", None), list):
            return []
        new_entries = task_lm.history[history_start:]
        entries = []
        for entry in new_entries:
            serialized = self._serialize_lm_history_entry(entry)
            serialized["attempt_idx"] = attempt_idx
            serialized["is_retry"] = attempt_idx > 0
            entries.append(serialized)
        return entries

    def _is_parse_error(self, error: Exception | None) -> bool:
        if error is None:
            return False
        if isinstance(error, AdapterParseError):
            return True
        cause = getattr(error, "__cause__", None)
        context = getattr(error, "__context__", None)
        return isinstance(cause, AdapterParseError) or isinstance(context, AdapterParseError)

    def _serialize_lm_history_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": self._safe_serialize(entry.get("prompt")),
            "messages": self._safe_serialize(entry.get("messages")),
            "kwargs": self._safe_serialize(entry.get("kwargs")),
            "outputs": self._safe_serialize(entry.get("outputs")),
            "usage": self._safe_serialize(entry.get("usage")),
            "cost": self._safe_serialize(entry.get("cost")),
            "timestamp": self._safe_serialize(entry.get("timestamp")),
            "uuid": self._safe_serialize(entry.get("uuid")),
            "model": self._safe_serialize(entry.get("model")),
            "response_model": self._safe_serialize(entry.get("response_model")),
            "model_type": self._safe_serialize(entry.get("model_type")),
        }

    def _fresh_rollout_id(self) -> int:
        return random.randint(1, 2**31 - 1)

    def _lm_trace(self, lm: LM) -> dict[str, Any]:
        return {
            "model": getattr(lm, "model", None),
            "cache": getattr(lm, "cache", None),
            "kwargs": self._safe_serialize(getattr(lm, "kwargs", {})),
        }

    def _prompt_signature(self, prompts: dict[str, str]) -> str:
        """Canonical prompt signature used only for exact null self-cut matching."""
        return json.dumps(self._safe_serialize(prompts), sort_keys=True, ensure_ascii=False)

    def _infer_input_fields(self, examples: list[Example]) -> list[str]:
        if not examples:
            return []
        try:
            return list(examples[0].inputs().keys())
        except Exception:
            return []

    def _safe_serialize(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._safe_serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._safe_serialize(v) for v in value]
        try:
            return dict(value)
        except Exception:
            return repr(value)


@experimental(version="3.1.0")
class SemanticBundleOptimizationLite(SemanticBundleOptimization):
    """SBO-Lite: critique-bundle search with qualitative constraint verification."""

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
        **kwargs
    ) -> Module:
        judge_lm = self.judge_lm or dspy.settings.lm
        proposer_lm = self.proposer_lm or dspy.settings.lm
        critic_lm = self.critic_lm or dspy.settings.lm

        if valset is None or len(valset) == 0:
            raise ValueError("SBO-Lite requires a validation set for robust loss estimation")
        critique_examples = valset

        logger.info(f"Starting SBO-Lite optimization with {len(trainset)} train, {len(valset)} val examples")
        self.trace = self._new_trace(trainset, valset, judge_lm, proposer_lm, critic_lm)
        self.trace["algorithm"] = "sbo_lite"
        self.trace["config"]["lite_verifier"] = "qualitative_json"

        original_prompts = self._extract_prompts(student)
        original_loss = self._evaluate_program(student, valset, eval_name="initial_val")
        initial_critique = self._generate_critique(
            student,
            critique_examples,
            original_prompts,
            critic_lm,
            trace_context=self.trace["initial"],
        )
        self.trace["initial"].update({
            "prompts": original_prompts,
            "loss": original_loss,
            "critique": initial_critique,
        })

        bundle = [BundleEntry(
            prompt=original_prompts,
            loss=original_loss,
            critique=initial_critique,
            iteration=0,
            self_score=0.0,
            lambda_value=0.0,
            kind="lite",
        )]
        center_program = student.deepcopy()
        center_idx = 0
        num_serious = 0
        num_null = 0
        consecutive_null = 0
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            old_center_entry = bundle[center_idx]
            old_center_loss = old_center_entry.loss
            active_bundle, watchlist_bundle = self._select_active_and_watchlist(bundle, center_idx)
            iteration_trace = {
                "iteration": iteration,
                "center_idx": center_idx,
                "center_loss": old_center_loss,
                "center_prompt": old_center_entry.prompt,
                "center_critique": old_center_entry.critique,
                "active_bundle": self._bundle_refs_for_trace(active_bundle),
                "watchlist_bundle": self._bundle_refs_for_trace(watchlist_bundle),
                "lite_controller": True,
            }

            candidates = self._generate_candidates(
                center_program,
                old_center_entry.critique,
                proposer_lm,
                trace_context=iteration_trace,
                active_bundle=active_bundle,
                watchlist_bundle=watchlist_bundle,
            )

            candidate_records = []
            for candidate_idx, candidate in enumerate(candidates):
                candidate_prompts = self._extract_prompts(candidate)
                candidate_loss = self._evaluate_program(
                    candidate,
                    valset,
                    eval_name=f"lite_iteration_{iteration}_candidate_{candidate_idx}_val",
                )
                verifier_trace: dict[str, Any] = {}
                verifier_result = self._lite_verify_candidate(
                    current_prompts=old_center_entry.prompt,
                    candidate_prompts=candidate_prompts,
                    active_bundle=active_bundle,
                    watchlist_bundle=watchlist_bundle,
                    lm=judge_lm,
                    trace_context=verifier_trace,
                )
                improvement = old_center_loss - candidate_loss
                admissible = improvement > 0 and not verifier_result.get("blocking_regression", True)
                candidate_records.append({
                    "candidate_index": candidate_idx,
                    "program": candidate,
                    "prompts": candidate_prompts,
                    "loss": candidate_loss,
                    "actual_improvement": improvement,
                    "verifier": verifier_result,
                    "verifier_trace": verifier_trace,
                    "admissible": admissible,
                })

            admissible_records = [record for record in candidate_records if record["admissible"]]
            best_record = min(
                admissible_records or candidate_records,
                key=lambda record: (record["loss"], -record["actual_improvement"]),
            )
            iteration_trace["candidates"] = [
                {k: v for k, v in record.items() if k != "program"}
                for record in candidate_records
            ]
            iteration_trace["selected_candidate_index"] = best_record["candidate_index"]
            iteration_trace["candidate_loss"] = best_record["loss"]
            iteration_trace["actual_improvement"] = best_record["actual_improvement"]

            if best_record["admissible"]:
                logger.info("SBO-Lite serious step: accepting improving candidate with no blocking regression")
                center_program = best_record["program"]
                num_serious += 1
                consecutive_null = 0
                critique_trace: dict[str, Any] = {}
                critique = self._generate_critique(
                    center_program,
                    critique_examples,
                    best_record["prompts"],
                    critic_lm,
                    trace_context=critique_trace,
                )
                bundle.append(BundleEntry(
                    prompt=best_record["prompts"],
                    loss=best_record["loss"],
                    critique=critique,
                    iteration=iteration,
                    self_score=0.0,
                    lambda_value=0.0,
                    kind="lite_serious",
                ))
                center_idx = len(bundle) - 1
                iteration_trace["step_type"] = "serious"
                iteration_trace["critique_generation"] = critique_trace
            else:
                logger.info("SBO-Lite null step: no admissible improving candidate")
                num_null += 1
                consecutive_null += 1
                critique_trace = {}
                if best_record["actual_improvement"] > 0 and best_record["verifier"].get("blocking_regression", True):
                    critique = (
                        "The candidate improved validation loss but regressed on stored bundle constraints: "
                        f"{best_record['verifier'].get('summary', 'blocking regression detected')}"
                    )
                    null_reason = "blocking_regression"
                else:
                    critique = self._generate_failure_critique(
                        best_record["program"],
                        valset,
                        best_record["prompts"],
                        target_loss=old_center_loss,
                        critic_lm=critic_lm,
                        trace_context=critique_trace,
                    )
                    null_reason = "no_admissible_improvement"
                bundle.append(BundleEntry(
                    prompt=best_record["prompts"],
                    loss=best_record["loss"],
                    critique=critique,
                    iteration=iteration,
                    self_score=0.0,
                    lambda_value=0.0,
                    kind="lite_null_refinement",
                ))
                iteration_trace["step_type"] = "null"
                iteration_trace["null_reason"] = null_reason
                iteration_trace["critique_generation"] = critique_trace

            bundle, center_idx = self._prune_bundle(bundle, center_idx)
            iteration_trace["bundle_size_after_prune"] = len(bundle)
            iteration_trace["center_idx_after_prune"] = center_idx
            iteration_trace["num_serious_steps_so_far"] = num_serious
            iteration_trace["num_null_steps_so_far"] = num_null
            iteration_trace["consecutive_null_steps"] = consecutive_null
            self._record_iteration(iteration_trace)

            if consecutive_null >= self.max_null_steps:
                logger.info(f"Terminating SBO-Lite: {self.max_null_steps} consecutive null steps")
                break
            if (
                self.stop_on_no_improving_candidate
                and not admissible_records
                and all(record["actual_improvement"] <= 0 for record in candidate_records)
            ):
                logger.info("Terminating SBO-Lite: no candidate improved validation loss")
                break

        best_idx = min(range(len(bundle)), key=lambda i: bundle[i].loss)
        best_program = self._build_program_from_prompts(student, bundle[best_idx].prompt)
        self.result = SBOResult(
            best_program=best_program,
            bundle=bundle,
            best_idx=best_idx,
            val_scores=[b.loss for b in bundle],
            total_iterations=iteration,
            num_serious_steps=num_serious,
            num_null_steps=num_null,
            trace=self._finalize_trace(bundle, best_idx, iteration, num_serious, num_null),
        )
        return best_program

    def _lite_verify_candidate(
        self,
        *,
        current_prompts: dict[str, str],
        candidate_prompts: dict[str, str],
        active_bundle: list[tuple[int, BundleEntry]],
        watchlist_bundle: list[tuple[int, BundleEntry]],
        lm: LM,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        current_text = self._format_instruction_prompt(current_prompts)
        candidate_text = self._format_instruction_prompt(candidate_prompts)
        active_text = self._format_bundle_for_lite_verifier(active_bundle, prefix="A")
        watchlist_text = self._format_bundle_for_lite_verifier(watchlist_bundle, prefix="W")
        verifier_prompt = f"""You are a constraint verifier for prompt optimization. You will be given a Current Prompt, a Candidate Prompt, an Active Critique Bundle, and a Watchlist of Non-Active Critiques.

Task:
Evaluate the candidate against all listed critiques in one pass. For each critique, decide whether the candidate resolves, ignores, or regresses on that critique relative to the current prompt. Then give an overall acceptability judgment.

Labels:
- resolved: the candidate clearly fixes the critique.
- unclear: the candidate is unrelated to the critique or only partially addresses it.
- regressed: the candidate makes the critique worse or reintroduces the failure mode.

Current Prompt:
{current_text}

Candidate Prompt:
{candidate_text}

Active Critique Bundle:
{active_text}

Watchlist of Non-Active Critiques:
{watchlist_text}

Output Format:
Return a compact JSON object with fields:
- active: a list of {{id, label}} objects for active critiques.
- watchlist: a list of {{id, label}} objects for watchlist critiques.
- blocking_regression: true if any active or watchlist critique is labeled regressed; otherwise false.
- summary: one short sentence identifying the main remaining failure mode.
Do not include long reasoning."""

        with dspy.context(lm=lm, temperature=self.judge_temperature):
            response = lm(
                verifier_prompt,
                temperature=self.judge_temperature,
                rollout_id=self._fresh_rollout_id() if self.judge_temperature and self.judge_temperature > 0 else None,
                cache=False,
            )
        response_text = (response[0] if isinstance(response, list) else response).strip()
        result = self._parse_lite_verifier_json(response_text)
        if trace_context is not None:
            trace_context.update({
                "current_prompts": current_prompts,
                "candidate_prompts": candidate_prompts,
                "active_bundle": self._bundle_refs_for_trace(active_bundle),
                "watchlist_bundle": self._bundle_refs_for_trace(watchlist_bundle),
                "verifier_prompt": verifier_prompt,
                "raw_response": self._safe_serialize(response),
                "response_text": response_text,
                "parsed": result,
                "temperature": self.judge_temperature,
                "cache": False,
            })
        return result

    def _parse_lite_verifier_json(self, response_text: str) -> dict[str, Any]:
        text = response_text.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        try:
            parsed = json.loads(text)
        except Exception:
            logger.warning(f"Failed to parse SBO-Lite verifier JSON: {response_text}")
            return {
                "active": [],
                "watchlist": [],
                "blocking_regression": True,
                "summary": "Verifier output could not be parsed.",
                "parse_error": True,
            }

        active = self._normalize_lite_labels(parsed.get("active", []))
        watchlist = self._normalize_lite_labels(parsed.get("watchlist", []))
        blocking = bool(parsed.get("blocking_regression", False))
        blocking = blocking or any(item["label"] == "regressed" for item in active + watchlist)
        return {
            "active": active,
            "watchlist": watchlist,
            "blocking_regression": blocking,
            "summary": str(parsed.get("summary", "")),
        }

    def _normalize_lite_labels(self, items: Any) -> list[dict[str, str]]:
        if not isinstance(items, list):
            return []
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "unclear")).strip().lower()
            if label not in {"resolved", "unclear", "regressed"}:
                label = "unclear"
            normalized.append({
                "id": str(item.get("id", "")),
                "label": label,
            })
        return normalized

    def _format_bundle_for_lite_verifier(
        self,
        bundle: list[tuple[int, BundleEntry]],
        *,
        prefix: str,
    ) -> str:
        if not bundle:
            return "(none)"
        return "\n".join(
            f"{prefix}{local_idx}: bundle[{entry_idx}], loss={entry.loss:.4f}, "
            f"critique={self._clip_for_prompt(entry.critique, self.max_bundle_critique_chars)}"
            for local_idx, (entry_idx, entry) in enumerate(bundle, start=1)
        )
