"""Semantic Bundle Optimization for a single system prompt string.

No DSPy. Task LM is called directly. Optimizer LM uses JSON mode.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _call_optimizer_lm(
    opt_lm: Any,
    *,
    call_type: str,
    system: str,
    user: str,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Call optimizer LM (JSON mode) and log the full prompt and parsed response."""
    logger.info("Optimizer LM [%s] — request", call_type)
    logger.info("  [system]\n%s", system)
    logger.info("  [user]\n%s", user)
    result = opt_lm.json(system, user, temperature=temperature)
    logger.info(
        "Optimizer LM [%s] — response\n%s",
        call_type,
        json.dumps(result, ensure_ascii=False, indent=2) if result else "(empty / parse failure)",
    )
    return result


@dataclass
class BundleEntry:
    prompt: str
    loss: float
    critique: str
    iteration: int
    self_score: float = 0.0
    lambda_value: float = 1.0
    kind: str = "standard"  # "standard" or "null_self_cut"
    exact_cut_loss: float | None = None  # exact null-step loss for self-cut


@dataclass
class SBOConfig:
    num_candidates: int = 5
    num_judge_samples: int = 3
    descent_param: float = 0.1
    lambda_init: float = 1.0
    lambda_min: float = 0.1
    lambda_max: float = 10.0
    lambda_gamma: float = 0.3
    bundle_size: int = 10
    active_bundle_size: int = 3
    watchlist_size: int = 2
    max_iterations: int = 20
    max_null_steps: int = 5
    temperature: float = 0.7
    judge_temperature: float = 0.7
    max_critique_examples: int = 3
    max_bundle_critique_chars: int = 900
    stop_on_no_improving_candidate: bool = True
    enable_exact_null_cuts: bool = True
    lambda_stability_epsilon: float = 1e-6
    include_constraint_detail: bool = True  # show per-constraint pass/fail in critique evidence
    candidate_eval_size: int | None = None  # minibatch size for F̃; None = full valset


@dataclass
class SBOResult:
    best_prompt: str
    best_loss: float
    bundle: list[BundleEntry]
    best_idx: int
    total_iterations: int
    num_serious_steps: int
    num_null_steps: int


# ─── Optimizer LM prompts ─────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
Quantify how well a Candidate Prompt addresses the Critique relative to the Reference Prompt.

Scoring rubric:
  +1.0  Strong Alignment: Candidate completely resolves the issue.
  +0.5  Weak Alignment: Candidate partially fixes the issue.
   0.0  Orthogonal: Candidate ignores the critique.
  -0.5  Weak Regression: Candidate slightly worsens the issue.
  -1.0  Strong Regression: Candidate explicitly violates the critique.

Be objective. Respond with ONLY valid JSON: {"score": <number in [-1.0, 1.0]>}\
"""

_PROPOSER_SYSTEM = """\
You are optimizing a SYSTEM-LEVEL instruction prompt — a general directive that tells a language model
how to behave across ALL tasks. The user prompts it will receive are varied and unpredictable.

Your job: generate local edits to the current system prompt that address the active critiques.

Rules:
  1. The output must be a GENERAL system-level instruction, not a task-specific prompt.
     Do NOT write prompts about specific tasks (rap verses, biographies, etc.) from the critique evidence.
     The critique evidence shows failure cases, not what to put in the system prompt.
  2. Make LOCAL edits — do NOT rewrite from scratch; preserve overall structure.
  3. Address the critiques at the level of general behavior (e.g. "follow format constraints more carefully").
  4. Each candidate must be meaningfully different from the others.
  5. Contain ONLY instruction text — no headers, no meta-commentary, no numbering.

Respond with ONLY valid JSON: {"candidates": ["text1", "text2", ...]}\
"""

_CRITIC_SYSTEM = """\
You are an expert prompt engineer. Analyze the instruction template and failure evidence \
to identify the SINGLE most critical weakness.

Steps:
  1. Analyze why the prompt failed on the given examples.
  2. Formulate a specific, actionable critique (e.g., "The prompt is too vague about output format").
  3. Do NOT propose a fix — state the critique only.

Respond with ONLY valid JSON: {"critique": "..."}\
"""

_FAILURE_CRITIC_SYSTEM = """\
You are an expert prompt engineer. A candidate prompt failed to improve performance.

Analyze why and provide a specific, actionable critique. Do NOT propose a fix.

Respond with ONLY valid JSON: {"critique": "..."}\
"""

_LITE_VERIFIER_SYSTEM = """\
You are a constraint verifier for prompt optimization.

Evaluate the candidate against all listed critiques in one pass. For each critique, decide whether
the candidate resolves, ignores, or regresses relative to the current prompt.

Labels:
  resolved  — the candidate clearly fixes the critique.
  unclear   — unrelated or only partially addresses it.
  regressed — makes the critique worse or reintroduces the failure mode.

Return a compact JSON object with these fields:
  active:               list of {id, label} for active critiques
  watchlist:            list of {id, label} for watchlist critiques
  blocking_regression:  true if ANY critique is labeled "regressed"; otherwise false
  summary:              one short sentence identifying the main remaining failure mode

Respond with ONLY valid JSON.\
"""


# ─── Optimizer ────────────────────────────────────────────────────────────────

class SBOOptimizer:
    """Semantic Bundle Optimization for a single system-prompt string.

    The task LM (e.g. 0.8B) is called directly via task_lm.task(system, user) → str.
    The optimizer LM is called via optimizer_lm.json(system, user) → dict for
    structured judge / proposer / critic calls.
    """

    def __init__(
        self,
        task_lm: Any,       # LMClient for task execution
        optimizer_lm: Any,  # LMClient for judge/proposer/critic
        metric: Any,        # Callable(IFEvalExample, response: str) -> float in [0, 1]
        config: SBOConfig | None = None,
    ):
        self.task_lm = task_lm
        self.opt_lm = optimizer_lm
        self.metric = metric
        self.config = config or SBOConfig()

    # ── Main optimization loop ────────────────────────────────────────────────

    def optimize(
        self,
        initial_prompt: str,
        trainset: list[Any],
        valset: list[Any],
    ) -> SBOResult:
        """Optimize initial_prompt using SBO. Returns SBOResult with best_prompt."""
        cfg = self.config

        current_loss = self._evaluate(initial_prompt, valset)
        logger.info("Initial loss: %.4f", current_loss)

        # Use valset for all critique generation (matches original SBO)
        critique_examples = valset
        critique = self._generate_critique(initial_prompt, critique_examples)
        logger.info("Initial critique: %s", critique)

        self_score = self._judge(initial_prompt, initial_prompt, critique)

        bundle: list[BundleEntry] = [BundleEntry(
            prompt=initial_prompt,
            loss=current_loss,
            critique=critique,
            iteration=0,
            self_score=self_score,
            lambda_value=cfg.lambda_init,
        )]
        center_idx = 0
        lambda_current = cfg.lambda_init
        num_serious = 0
        num_null = 0
        consecutive_null = 0
        iteration = 0

        for iteration in range(1, cfg.max_iterations + 1):
            logger.info("\n=== SBO Iteration %d/%d ===", iteration, cfg.max_iterations)
            logger.info("Center loss: %.4f  λ: %.3f", bundle[center_idx].loss, lambda_current)

            center_entry = bundle[center_idx]
            active, watchlist = self._select_active_and_watchlist(bundle, center_idx)

            # Stage 1: Generate candidate prompt strings
            candidate_texts = self._generate_candidates(center_entry.prompt, active, watchlist)
            if not candidate_texts:
                logger.warning("Skipping iteration %d: proposer returned no candidates.", iteration)
                num_null += 1
                consecutive_null += 1
                bundle.append(BundleEntry(
                    prompt=center_entry.prompt,
                    loss=center_entry.loss,
                    critique="Proposer failed to generate candidates; retry with different phrasing.",
                    iteration=iteration,
                    kind="null_self_cut",
                ))
                if consecutive_null >= cfg.max_null_steps:
                    break
                continue

            # Stage 2: Score each candidate (bundle model + approximate eval F̃).
            # Use a minibatch for fast candidate ranking; re-eval winner on full val.
            if cfg.candidate_eval_size and cfg.candidate_eval_size < len(valset):
                eval_batch = random.sample(valset, cfg.candidate_eval_size)
                center_approx_loss = self._evaluate(center_entry.prompt, eval_batch)
            else:
                eval_batch = valset
                center_approx_loss = center_entry.loss

            candidate_records = []
            for i, ctext in enumerate(candidate_texts):
                model_value = self._compute_model_value(ctext, active, lambda_current)
                cand_loss = self._evaluate(ctext, eval_batch)
                predicted_improvement = center_entry.loss - model_value
                actual_improvement = center_approx_loss - cand_loss
                serious_threshold = cfg.descent_param * max(0.0, predicted_improvement)
                candidate_records.append({
                    "text": ctext,
                    "model_value": model_value,
                    "loss": cand_loss,
                    "actual_improvement": actual_improvement,
                    "in_serious_set": actual_improvement > 0 and actual_improvement >= serious_threshold,
                })
                logger.info(
                    "  Candidate %d: loss=%.4f  model=%.4f  improvement=%.4f",
                    i + 1, cand_loss, model_value, actual_improvement,
                )

            # Stage 3: Select best
            serious = [r for r in candidate_records if r["in_serious_set"]]
            if serious:
                best = min(serious, key=lambda r: (r["loss"], r["model_value"]))
                step_type = "serious"
            else:
                best = min(candidate_records, key=lambda r: (r["loss"], r["model_value"]))
                step_type = "null"

            best_text = best["text"]
            best_loss = best["loss"]
            actual_improvement = best["actual_improvement"]

            # Stage 4: Serious or null step
            if step_type == "serious":
                # Re-evaluate winner on full valset for accurate center loss.
                if eval_batch is not valset:
                    best_loss = self._evaluate(best_text, valset)
                    actual_improvement = center_entry.loss - best_loss
                    logger.info("✓ SERIOUS STEP: %.4f → %.4f (full-val re-eval)", center_entry.loss, best_loss)
                else:
                    logger.info("✓ SERIOUS STEP: %.4f → %.4f", center_entry.loss, best_loss)
                num_serious += 1
                consecutive_null = 0

                # Lambda update (EMA)
                sem_score = self._judge(best_text, center_entry.prompt, center_entry.critique)
                centered = sem_score - center_entry.self_score
                lambda_obs = actual_improvement / (abs(centered) + cfg.lambda_stability_epsilon)
                lambda_current = max(
                    cfg.lambda_min,
                    min(
                        cfg.lambda_max,
                        (1 - cfg.lambda_gamma) * lambda_current + cfg.lambda_gamma * lambda_obs,
                    ),
                )
                logger.info("Updated λ: %.3f", lambda_current)

                new_critique = self._generate_critique(best_text, critique_examples)

            else:
                logger.info("✗ NULL STEP: candidate did not meet descent criterion")
                num_null += 1
                consecutive_null += 1
                new_critique = self._generate_failure_critique(best_text, critique_examples, center_entry.loss)

            # Stage 5: Compute self-score for new bundle entry and append
            new_self_score = self._judge(best_text, best_text, new_critique)
            exact_cut_loss = best_loss if step_type == "null" and cfg.enable_exact_null_cuts else None
            bundle.append(BundleEntry(
                prompt=best_text,
                loss=best_loss,
                critique=new_critique,
                iteration=iteration,
                self_score=new_self_score,
                lambda_value=lambda_current,
                kind="standard" if step_type == "serious" else "null_self_cut",
                exact_cut_loss=exact_cut_loss,
            ))
            if step_type == "serious":
                center_idx = len(bundle) - 1

            bundle, center_idx = self._prune_bundle(bundle, center_idx)

            # Termination checks
            if consecutive_null >= cfg.max_null_steps:
                logger.info("Terminating: %d consecutive null steps", cfg.max_null_steps)
                break
            if cfg.stop_on_no_improving_candidate and all(
                r["actual_improvement"] <= 0 for r in candidate_records
            ):
                logger.info("Terminating: no candidate improved validation loss")
                break

        best_idx = min(range(len(bundle)), key=lambda i: bundle[i].loss)
        logger.info(
            "\nDone. Best loss: %.4f (iteration %d). Serious: %d, Null: %d",
            bundle[best_idx].loss, bundle[best_idx].iteration, num_serious, num_null,
        )
        return SBOResult(
            best_prompt=bundle[best_idx].prompt,
            best_loss=bundle[best_idx].loss,
            bundle=bundle,
            best_idx=best_idx,
            total_iterations=iteration,
            num_serious_steps=num_serious,
            num_null_steps=num_null,
        )

    # ── Task evaluation ───────────────────────────────────────────────────────

    def _evaluate(self, system_prompt: str, examples: list[Any]) -> float:
        """Average loss (1 − accuracy) of system_prompt on examples."""
        total_loss = 0.0
        for ex in examples:
            try:
                response = self.task_lm.task(system_prompt, ex.prompt)
                s = float(self.metric(ex, response))
                total_loss += 1.0 - s
            except Exception as e:
                logger.warning("Eval error on key=%s: %s", getattr(ex, "key", "?"), e)
                total_loss += 1.0
        return total_loss / len(examples) if examples else 1.0

    # ── Judge ─────────────────────────────────────────────────────────────────

    def _judge(self, candidate: str, reference: str, critique: str) -> float:
        """Ŝ_J(candidate, reference, critique) averaged over num_judge_samples."""
        cfg = self.config
        user = f"Reference prompt:\n{reference}\n\nCritique:\n{critique}\n\nCandidate prompt:\n{candidate}"
        scores: list[float] = []
        for sample_i in range(cfg.num_judge_samples):
            result = _call_optimizer_lm(
                self.opt_lm,
                call_type=f"judge ({sample_i + 1}/{cfg.num_judge_samples})",
                system=_JUDGE_SYSTEM,
                user=user,
                temperature=cfg.judge_temperature,
            )
            try:
                s = float(result.get("score", 0.0))
                scores.append(max(-1.0, min(1.0, s)))
            except (TypeError, ValueError):
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    # ── Proposer ──────────────────────────────────────────────────────────────

    def _generate_candidates(
        self,
        current_prompt: str,
        active: list[tuple[int, BundleEntry]],
        watchlist: list[tuple[int, BundleEntry]],
    ) -> list[str]:
        cfg = self.config
        active_text = self._format_bundle(active)
        watchlist_text = self._format_bundle(watchlist) if watchlist else "(none)"
        user = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"Active critiques:\n{active_text}\n\n"
            f"Watchlist (avoid regressions):\n{watchlist_text}\n\n"
            f"Generate exactly {cfg.num_candidates} candidate variations."
        )
        result = _call_optimizer_lm(
            self.opt_lm,
            call_type="proposer",
            system=_PROPOSER_SYSTEM,
            user=user,
            temperature=cfg.temperature,
        )
        # Exclude duplicates of the current prompt — verifying an identical candidate
        # wastes LM calls and produces incoherent verifier feedback.
        candidates = [
            c.strip() for c in result.get("candidates", [])
            if isinstance(c, str) and c.strip() and c.strip() != current_prompt
        ]
        if not candidates:
            logger.warning("Proposer returned no usable candidates (parse failure or all duplicates).")
        return candidates[: cfg.num_candidates]

    # ── Critic ────────────────────────────────────────────────────────────────

    def _generate_critique(self, prompt: str, examples: list[Any]) -> str:
        cfg = self.config
        sampled = random.sample(examples, min(cfg.max_critique_examples, len(examples)))
        failures: list[str] = []
        for i, ex in enumerate(sampled, 1):
            try:
                response = self.task_lm.task(prompt, ex.prompt)
                s = float(self.metric(ex, response))
                if s < 0.8:
                    entry = f"Example {i}:\nPrompt: {ex.prompt[:400]}\nResponse: {response[:400]}\nScore: {s:.2f}"
                    if cfg.include_constraint_detail:
                        from sbo_ifeval.metrics import score_with_detail
                        _, _, breakdown = score_with_detail(ex, response)
                        detail = ", ".join(
                            f"{iid.split(':')[1]}={'PASS' if ok else 'FAIL'}"
                            for iid, ok in breakdown
                        )
                        entry += f"\nConstraints: {detail}"
                    failures.append(entry)
            except Exception as e:
                failures.append(f"Example {i}: Error — {e}")

        if not failures:
            return "The prompt is performing well on the sampled examples."

        user = f"Instruction template:\n{prompt}\n\nFailure evidence:\n" + "\n\n".join(failures)
        result = _call_optimizer_lm(
            self.opt_lm,
            call_type="critic",
            system=_CRITIC_SYSTEM,
            user=user,
            temperature=cfg.temperature,
        )
        return result.get("critique", "The prompt needs refinement on some examples.").strip()

    def _generate_failure_critique(self, prompt: str, examples: list[Any], target_loss: float) -> str:
        cfg = self.config
        sampled = random.sample(examples, min(cfg.max_critique_examples, len(examples)))
        evidence: list[str] = []
        for i, ex in enumerate(sampled, 1):
            try:
                response = self.task_lm.task(prompt, ex.prompt)
                s = float(self.metric(ex, response))
                entry = f"Example {i}: Score={s:.2f}\nResponse: {response[:200]}"
                if cfg.include_constraint_detail:
                    from sbo_ifeval.metrics import score_with_detail
                    _, _, breakdown = score_with_detail(ex, response)
                    detail = ", ".join(
                        f"{iid.split(':')[1]}={'PASS' if ok else 'FAIL'}"
                        for iid, ok in breakdown
                    )
                    entry += f"\nConstraints: {detail}"
                evidence.append(entry)
            except Exception as e:
                evidence.append(f"Example {i}: Error — {e}")

        user = (
            f"Candidate prompt:\n{prompt}\n\n"
            f"Target loss to beat: {target_loss:.4f}\n\n"
            f"Evidence:\n" + "\n\n".join(evidence)
        )
        result = _call_optimizer_lm(
            self.opt_lm,
            call_type="failure_critic",
            system=_FAILURE_CRITIC_SYSTEM,
            user=user,
            temperature=cfg.temperature,
        )
        return result.get("critique", "The candidate did not improve; revisit approach.").strip()

    # ── Bundle model ──────────────────────────────────────────────────────────

    def _compute_model_value(
        self,
        candidate: str,
        bundle: list[tuple[int, BundleEntry]],
        lambda_current: float,
    ) -> float:
        """M_k(p) = max_i { F̃_i − λ_i · [Ŝ_J(p, p_i, c_i) − s_i^0] }"""
        if not bundle:
            return float("inf")
        model_value = float("-inf")
        for _, entry in bundle:
            sem_score = self._judge(candidate, entry.prompt, entry.critique)
            centered = sem_score - entry.self_score
            lam = entry.lambda_value if entry.lambda_value is not None else lambda_current
            value = entry.loss - lam * centered
            # Exact null-step cut: if candidate is the exact same string, floor its value
            if (
                self.config.enable_exact_null_cuts
                and entry.exact_cut_loss is not None
                and candidate == entry.prompt
            ):
                value = max(value, entry.exact_cut_loss)
            model_value = max(model_value, value)
        return model_value

    # ── Bundle management ─────────────────────────────────────────────────────

    def _select_active_and_watchlist(
        self,
        bundle: list[BundleEntry],
        center_idx: int,
    ) -> tuple[list[tuple[int, BundleEntry]], list[tuple[int, BundleEntry]]]:
        cfg = self.config
        indexed = list(enumerate(bundle))
        center = [(center_idx, bundle[center_idx])]
        remaining = [(i, e) for i, e in indexed if i != center_idx]
        # Prefer high-loss and recent critiques
        remaining_sorted = sorted(remaining, key=lambda x: (x[1].loss, x[1].iteration), reverse=True)
        active = (center + remaining_sorted)[: cfg.active_bundle_size]
        active_ids = {i for i, _ in active}
        watchlist_pool = [(i, e) for i, e in remaining_sorted if i not in active_ids]
        watchlist = watchlist_pool[: cfg.watchlist_size]
        return active, watchlist

    def _prune_bundle(
        self,
        bundle: list[BundleEntry],
        center_idx: int,
    ) -> tuple[list[BundleEntry], int]:
        cfg = self.config
        if len(bundle) <= cfg.bundle_size:
            return bundle, center_idx
        keep = {center_idx}
        rest = sorted(
            [(i, e) for i, e in enumerate(bundle) if i != center_idx],
            key=lambda x: (x[1].iteration, x[1].loss),
            reverse=True,
        )
        for i, _ in rest[: cfg.bundle_size - 1]:
            keep.add(i)
        sorted_keep = sorted(keep)
        new_bundle = [bundle[i] for i in sorted_keep]
        old_to_new = {old: new for new, old in enumerate(sorted_keep)}
        return new_bundle, old_to_new[center_idx]

    def _format_bundle(self, bundle: list[tuple[int, BundleEntry]]) -> str:
        if not bundle:
            return "(none)"
        cfg = self.config
        lines = []
        for local_i, (_, entry) in enumerate(bundle, 1):
            critique = entry.critique
            if len(critique) > cfg.max_bundle_critique_chars:
                critique = critique[: cfg.max_bundle_critique_chars] + "..."
            lines.append(f"{local_i}. iter={entry.iteration}, loss={entry.loss:.4f}: {critique}")
        return "\n".join(lines)


# ─── SBO-Lite ─────────────────────────────────────────────────────────────────

@dataclass
class SBOLiteConfig:
    """Configuration for SBO-Lite.

    Much cheaper than full SBO: replaces Monte Carlo judge calls with a single
    qualitative verifier call per candidate. No lambda adaptation.
    """
    num_candidates: int = 3
    bundle_size: int = 8
    active_bundle_size: int = 3
    watchlist_size: int = 2
    max_iterations: int = 10
    max_null_steps: int = 3
    temperature: float = 0.5
    max_critique_examples: int = 2
    max_bundle_critique_chars: int = 900
    stop_on_no_improving_candidate: bool = True
    include_constraint_detail: bool = True  # show per-constraint pass/fail in critique evidence
    candidate_eval_size: int | None = None  # minibatch size for F̃; None = full valset


class SBOLiteOptimizer:
    """SBO-Lite: Semantic Bundle Optimization without the semantic judge.

    Per-iteration cost: num_candidates × 1 verifier call  (vs full SBO's
    num_candidates × active_bundle_size × num_judge_samples judge calls).

    Acceptance criterion: improvement > 0 AND verifier says no blocking regression.

    Shares the proposer, critic, bundle management, and task-evaluation logic
    with SBOOptimizer but skips all lambda/bundle-model machinery.
    """

    def __init__(
        self,
        task_lm: Any,
        optimizer_lm: Any,
        metric: Any,
        config: SBOLiteConfig | None = None,
    ):
        self.task_lm = task_lm
        self.opt_lm = optimizer_lm
        self.metric = metric
        self.config = config or SBOLiteConfig()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def optimize(
        self,
        initial_prompt: str,
        trainset: list[Any],
        valset: list[Any],
    ) -> SBOResult:
        cfg = self.config

        current_loss = self._evaluate(initial_prompt, valset)
        logger.info("Initial loss: %.4f", current_loss)

        # Use valset for all critique generation (matches original SBO/SBO-Lite)
        critique_examples = valset
        critique = self._generate_critique(initial_prompt, critique_examples)
        logger.info("Initial critique: %s", critique)

        bundle: list[BundleEntry] = [BundleEntry(
            prompt=initial_prompt,
            loss=current_loss,
            critique=critique,
            iteration=0,
            kind="lite",
        )]
        center_idx = 0
        num_serious = 0
        num_null = 0
        consecutive_null = 0
        iteration = 0

        for iteration in range(1, cfg.max_iterations + 1):
            logger.info("\n=== SBO-Lite Iteration %d/%d ===", iteration, cfg.max_iterations)
            logger.info("Center loss: %.4f", bundle[center_idx].loss)

            center_entry = bundle[center_idx]
            active, watchlist = self._select_active_and_watchlist(bundle, center_idx)

            candidate_texts = self._generate_candidates(center_entry.prompt, active, watchlist)

            if not candidate_texts:
                logger.warning("Skipping iteration %d: proposer returned no candidates.", iteration)
                num_null += 1
                consecutive_null += 1
                bundle.append(BundleEntry(
                    prompt=center_entry.prompt,
                    loss=center_entry.loss,
                    critique="Proposer failed to generate candidates; retry with different phrasing.",
                    iteration=iteration,
                    kind="lite_null",
                ))
                if consecutive_null >= cfg.max_null_steps:
                    break
                continue

            # Use a minibatch from valset for F̃ (approximate candidate scoring).
            # This keeps per-iteration cost at N × candidate_eval_size instead of
            # N × len(valset). The winner is re-evaluated on full valset after selection.
            if cfg.candidate_eval_size and cfg.candidate_eval_size < len(valset):
                eval_batch = random.sample(valset, cfg.candidate_eval_size)
                # Re-scale center loss to the same minibatch for a fair comparison.
                center_approx_loss = self._evaluate(center_entry.prompt, eval_batch)
            else:
                eval_batch = valset
                center_approx_loss = center_entry.loss

            candidate_records = []
            for i, ctext in enumerate(candidate_texts):
                cand_loss = self._evaluate(ctext, eval_batch)
                improvement = center_approx_loss - cand_loss
                # Single verifier call instead of judge calls
                verifier = self._lite_verify(center_entry.prompt, ctext, active, watchlist)
                blocking = verifier.get("blocking_regression", True)
                admissible = improvement > 0 and not blocking
                candidate_records.append({
                    "text": ctext,
                    "loss": cand_loss,
                    "actual_improvement": improvement,
                    "blocking_regression": blocking,
                    "admissible": admissible,
                    "verifier": verifier,
                    "verifier_summary": verifier.get("summary", ""),
                })
                logger.info(
                    "  Candidate %d: loss=%.4f  improvement=%.4f  blocking=%s",
                    i + 1, cand_loss, improvement, blocking,
                )

            admissible_records = [r for r in candidate_records if r["admissible"]]
            best = min(
                admissible_records or candidate_records,
                key=lambda r: (r["loss"], -r["actual_improvement"]),
            )
            step_type = "serious" if best["admissible"] else "null"
            best_text = best["text"]
            best_loss = best["loss"]

            if step_type == "serious":
                # Re-evaluate winner on full valset for an accurate center loss.
                if eval_batch is not valset:
                    best_loss = self._evaluate(best_text, valset)
                    logger.info("✓ SERIOUS STEP: %.4f → %.4f (full-val re-eval)", center_entry.loss, best_loss)
                else:
                    logger.info("✓ SERIOUS STEP: %.4f → %.4f", center_entry.loss, best_loss)
                num_serious += 1
                consecutive_null = 0
                new_critique = self._generate_critique(best_text, critique_examples)
            else:
                logger.info("✗ NULL STEP")
                num_null += 1
                consecutive_null += 1
                # Distinguish blocking regression from no-improvement null steps
                if best["actual_improvement"] > 0 and best["verifier"].get("blocking_regression", True):
                    new_critique = (
                        "The candidate improved validation loss but regressed on stored bundle constraints: "
                        f"{best['verifier'].get('summary', 'blocking regression detected')}"
                    )
                else:
                    new_critique = self._generate_failure_critique(best_text, critique_examples, center_entry.loss)

            bundle.append(BundleEntry(
                prompt=best_text,
                loss=best_loss,
                critique=new_critique,
                iteration=iteration,
                kind="lite_serious" if step_type == "serious" else "lite_null",
            ))
            if step_type == "serious":
                center_idx = len(bundle) - 1

            bundle, center_idx = self._prune_bundle(bundle, center_idx)

            if consecutive_null >= cfg.max_null_steps:
                logger.info("Terminating: %d consecutive null steps", cfg.max_null_steps)
                break
            if (
                cfg.stop_on_no_improving_candidate
                and not admissible_records
                and all(r["actual_improvement"] <= 0 for r in candidate_records)
            ):
                logger.info("Terminating: no candidate improved validation loss")
                break

        best_idx = min(range(len(bundle)), key=lambda i: bundle[i].loss)
        logger.info(
            "\nDone. Best loss: %.4f (iteration %d). Serious: %d, Null: %d",
            bundle[best_idx].loss, bundle[best_idx].iteration, num_serious, num_null,
        )
        return SBOResult(
            best_prompt=bundle[best_idx].prompt,
            best_loss=bundle[best_idx].loss,
            bundle=bundle,
            best_idx=best_idx,
            total_iterations=iteration,
            num_serious_steps=num_serious,
            num_null_steps=num_null,
        )

    # ── Task evaluation ───────────────────────────────────────────────────────

    def _evaluate(self, system_prompt: str, examples: list[Any]) -> float:
        total_loss = 0.0
        for ex in examples:
            try:
                response = self.task_lm.task(system_prompt, ex.prompt)
                s = float(self.metric(ex, response))
                total_loss += 1.0 - s
            except Exception as e:
                logger.warning("Eval error on key=%s: %s", getattr(ex, "key", "?"), e)
                total_loss += 1.0
        return total_loss / len(examples) if examples else 1.0

    # ── Lite verifier (replaces judge) ────────────────────────────────────────

    def _lite_verify(
        self,
        current_prompt: str,
        candidate_prompt: str,
        active: list[tuple[int, BundleEntry]],
        watchlist: list[tuple[int, BundleEntry]],
    ) -> dict[str, Any]:
        """Single qualitative check: does candidate resolve critiques without regressions?"""
        active_text = "\n".join(
            f"A{i+1} (iter={e.iteration}, loss={e.loss:.4f}): {e.critique[:600]}"
            for i, (_, e) in enumerate(active)
        )
        watchlist_text = "\n".join(
            f"W{i+1}: {e.critique[:400]}"
            for i, (_, e) in enumerate(watchlist)
        ) or "(none)"
        user = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"Candidate prompt:\n{candidate_prompt}\n\n"
            f"Active critiques:\n{active_text}\n\n"
            f"Watchlist critiques:\n{watchlist_text}"
        )
        result = _call_optimizer_lm(
            self.opt_lm,
            call_type="lite_verifier",
            system=_LITE_VERIFIER_SYSTEM,
            user=user,
        )
        # Recompute blocking_regression from labels — don't trust the model's boolean.
        # Models reliably produce the per-critique labels but often get the derived
        # boolean wrong (e.g. label="regressed" but blocking_regression=false).
        all_labels = [
            item.get("label", "") for item in result.get("active", [])
        ] + [
            item.get("label", "") for item in result.get("watchlist", [])
        ]
        result["blocking_regression"] = any(lbl == "regressed" for lbl in all_labels)
        return result

    # ── Proposer ──────────────────────────────────────────────────────────────

    def _generate_candidates(
        self,
        current_prompt: str,
        active: list[tuple[int, BundleEntry]],
        watchlist: list[tuple[int, BundleEntry]],
    ) -> list[str]:
        cfg = self.config
        active_text = self._format_bundle(active)
        watchlist_text = self._format_bundle(watchlist) if watchlist else "(none)"
        user = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"Active critiques:\n{active_text}\n\n"
            f"Watchlist (avoid regressions):\n{watchlist_text}\n\n"
            f"Generate exactly {cfg.num_candidates} candidate variations."
        )
        result = _call_optimizer_lm(
            self.opt_lm,
            call_type="proposer",
            system=_PROPOSER_SYSTEM,
            user=user,
            temperature=cfg.temperature,
        )
        candidates = [
            c.strip() for c in result.get("candidates", [])
            if isinstance(c, str) and c.strip() and c.strip() != current_prompt
        ]
        if not candidates:
            logger.warning("Proposer returned no usable candidates (parse failure or all duplicates).")
        return candidates[: cfg.num_candidates]

    # ── Critic ────────────────────────────────────────────────────────────────

    def _generate_critique(self, prompt: str, examples: list[Any]) -> str:
        cfg = self.config
        sampled = random.sample(examples, min(cfg.max_critique_examples, len(examples)))
        failures: list[str] = []
        for i, ex in enumerate(sampled, 1):
            try:
                response = self.task_lm.task(prompt, ex.prompt)
                s = float(self.metric(ex, response))
                if s < 0.8:
                    entry = f"Example {i}:\nPrompt: {ex.prompt[:400]}\nResponse: {response[:400]}\nScore: {s:.2f}"
                    if cfg.include_constraint_detail:
                        from sbo_ifeval.metrics import score_with_detail
                        _, _, breakdown = score_with_detail(ex, response)
                        detail = ", ".join(
                            f"{iid.split(':')[1]}={'PASS' if ok else 'FAIL'}"
                            for iid, ok in breakdown
                        )
                        entry += f"\nConstraints: {detail}"
                    failures.append(entry)
            except Exception as e:
                failures.append(f"Example {i}: Error — {e}")
        if not failures:
            return "The prompt is performing well on the sampled examples."
        user = f"Instruction template:\n{prompt}\n\nFailure evidence:\n" + "\n\n".join(failures)
        result = _call_optimizer_lm(
            self.opt_lm,
            call_type="critic",
            system=_CRITIC_SYSTEM,
            user=user,
            temperature=cfg.temperature,
        )
        return result.get("critique", "The prompt needs refinement on some examples.").strip()

    def _generate_failure_critique(self, prompt: str, examples: list[Any], target_loss: float) -> str:
        cfg = self.config
        sampled = random.sample(examples, min(cfg.max_critique_examples, len(examples)))
        evidence: list[str] = []
        for i, ex in enumerate(sampled, 1):
            try:
                response = self.task_lm.task(prompt, ex.prompt)
                s = float(self.metric(ex, response))
                entry = f"Example {i}: Score={s:.2f}\nResponse: {response[:200]}"
                if cfg.include_constraint_detail:
                    from sbo_ifeval.metrics import score_with_detail
                    _, _, breakdown = score_with_detail(ex, response)
                    detail = ", ".join(
                        f"{iid.split(':')[1]}={'PASS' if ok else 'FAIL'}"
                        for iid, ok in breakdown
                    )
                    entry += f"\nConstraints: {detail}"
                evidence.append(entry)
            except Exception as e:
                evidence.append(f"Example {i}: Error — {e}")
        user = (
            f"Candidate prompt:\n{prompt}\n\n"
            f"Target loss to beat: {target_loss:.4f}\n\n"
            f"Evidence:\n" + "\n\n".join(evidence)
        )
        result = _call_optimizer_lm(
            self.opt_lm,
            call_type="failure_critic",
            system=_FAILURE_CRITIC_SYSTEM,
            user=user,
            temperature=cfg.temperature,
        )
        return result.get("critique", "The candidate did not improve; revisit approach.").strip()

    # ── Bundle management ─────────────────────────────────────────────────────

    def _select_active_and_watchlist(
        self,
        bundle: list[BundleEntry],
        center_idx: int,
    ) -> tuple[list[tuple[int, BundleEntry]], list[tuple[int, BundleEntry]]]:
        cfg = self.config
        indexed = list(enumerate(bundle))
        center = [(center_idx, bundle[center_idx])]
        remaining = [(i, e) for i, e in indexed if i != center_idx]
        remaining_sorted = sorted(remaining, key=lambda x: (x[1].loss, x[1].iteration), reverse=True)
        active = (center + remaining_sorted)[: cfg.active_bundle_size]
        active_ids = {i for i, _ in active}
        watchlist_pool = [(i, e) for i, e in remaining_sorted if i not in active_ids]
        watchlist = watchlist_pool[: cfg.watchlist_size]
        return active, watchlist

    def _prune_bundle(
        self,
        bundle: list[BundleEntry],
        center_idx: int,
    ) -> tuple[list[BundleEntry], int]:
        cfg = self.config
        if len(bundle) <= cfg.bundle_size:
            return bundle, center_idx
        keep = {center_idx}
        rest = sorted(
            [(i, e) for i, e in enumerate(bundle) if i != center_idx],
            key=lambda x: (x[1].iteration, x[1].loss),
            reverse=True,
        )
        for i, _ in rest[: cfg.bundle_size - 1]:
            keep.add(i)
        sorted_keep = sorted(keep)
        new_bundle = [bundle[i] for i in sorted_keep]
        old_to_new = {old: new for new, old in enumerate(sorted_keep)}
        return new_bundle, old_to_new[center_idx]

    def _format_bundle(self, bundle: list[tuple[int, BundleEntry]]) -> str:
        if not bundle:
            return "(none)"
        cfg = self.config
        lines = []
        for local_i, (_, entry) in enumerate(bundle, 1):
            critique = entry.critique
            if len(critique) > cfg.max_bundle_critique_chars:
                critique = critique[: cfg.max_bundle_critique_chars] + "..."
            lines.append(f"{local_i}. iter={entry.iteration}, loss={entry.loss:.4f}: {critique}")
        return "\n".join(lines)
