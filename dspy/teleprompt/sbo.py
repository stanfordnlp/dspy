"""
Semantic Bundle Optimization (SBO) for DSPy.

Based on: "Semantic Bundle Methods: Rigorous Prompt Optimization via Discrete-Continuous Relaxation"

SBO addresses limit cycles and catastrophic forgetting in greedy prompt optimization by maintaining
a "bundle" of historical critiques and using them to construct a cutting-plane model of the objective.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import dspy
from dspy.clients.lm import LM
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)


@dataclass
class BundleEntry:
    """A single entry in the optimization bundle."""
    prompt: dict[str, str]  # Component name -> instruction text
    loss: float  # F̃(p_i) - smoothed/robust loss
    critique: str  # Textual critique c_i
    iteration: int  # When this was added


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
        max_iterations: int = 50,
        max_null_steps: int = 5,
        temperature: float = 0.7,
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
        self.max_iterations = max_iterations
        self.max_null_steps = max_null_steps
        self.temperature = temperature
        self.track_stats = track_stats

        self.result: Optional[SBOResult] = None

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

        logger.info(f"Starting SBO optimization with {len(trainset)} train, {len(valset)} val examples")

        # Initialize bundle with original program
        original_prompts = self._extract_prompts(student)
        logger.info(f"\n{'='*60}")
        logger.info(f"INITIAL PROMPTS EXTRACTED:")
        for pred_name, prompt_text in original_prompts.items():
            logger.info(f"  [{pred_name}]: {repr(prompt_text)[:200]}")
        logger.info(f"{'='*60}\n")

        original_loss = self._evaluate_program(student, valset)
        logger.info(f"Original loss on valset: {original_loss:.4f}")

        initial_critique = self._generate_critique(student, trainset, original_prompts, critic_lm)
        logger.info(f"\nINITIAL CRITIQUE:")
        logger.info(f"  {initial_critique[:500]}")
        logger.info(f"")

        bundle = [BundleEntry(
            prompt=original_prompts,
            loss=original_loss,
            critique=initial_critique,
            iteration=0
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

            # Stage 1: Generate candidates (Proposer)
            logger.info(f"\nCurrent critique for candidate generation:")
            logger.info(f"  {bundle[center_idx].critique[:300]}...")

            candidates = self._generate_candidates(
                center_program,
                bundle[center_idx].critique,
                proposer_lm
            )

            logger.info(f"\nGenerated {len(candidates)} candidate programs")
            # Log all candidates to see what was generated
            for i, candidate in enumerate(candidates):
                cand_prompts = self._extract_prompts(candidate)
                logger.info(f"  Candidate {i+1} prompts:")
                for pred_name, prompt_text in cand_prompts.items():
                    logger.info(f"    [{pred_name}]: {repr(prompt_text)[:200]}")
                    # Check if this candidate is different from center
                    if prompt_text == bundle[center_idx].prompt.get(pred_name):
                        logger.warning(f"    ⚠ Candidate {i+1} [{pred_name}] is IDENTICAL to center!")

            # Stage 2: Filter candidates (Verifier)
            best_candidate, best_candidate_prompts = self._select_best_candidate(
                candidates,
                bundle,
                lambda_current,
                judge_lm
            )

            # Evaluate robust loss of selected candidate
            candidate_loss = self._evaluate_program(best_candidate, valset)

            # Compute predicted improvement
            model_value = self._compute_model_value(
                best_candidate_prompts,
                bundle,
                lambda_current,
                judge_lm
            )
            predicted_improvement = bundle[center_idx].loss - model_value
            actual_improvement = bundle[center_idx].loss - candidate_loss

            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration} RESULTS:")
            logger.info(f"  Center loss:           {bundle[center_idx].loss:.4f}")
            logger.info(f"  Candidate loss:        {candidate_loss:.4f}")
            logger.info(f"  Model value:           {model_value:.4f}")
            logger.info(f"  Predicted improvement: {predicted_improvement:.4f}")
            logger.info(f"  Actual improvement:    {actual_improvement:.4f}")
            logger.info(f"\nBest candidate selected:")
            for pred_name, prompt_text in best_candidate_prompts.items():
                # Check if changed
                is_same = (prompt_text == bundle[center_idx].prompt.get(pred_name, ""))
                status = "[UNCHANGED]" if is_same else "[MODIFIED]"
                logger.info(f"  [{pred_name}] {status}: {repr(prompt_text)[:200]}")
            logger.info(f"{'='*60}\n")

            # Descent test: Serious vs Null step
            if actual_improvement >= self.descent_param * predicted_improvement:
                # SERIOUS STEP - Accept candidate as new center
                logger.info("✓ SERIOUS STEP: Accepting candidate as new center")

                center_program = best_candidate
                center_idx = len(bundle)  # Will be the new bundle entry
                num_serious += 1
                consecutive_null = 0

                # Update sensitivity λ
                semantic_score = self._compute_semantic_score(
                    best_candidate_prompts,
                    bundle[center_idx - 1].prompt,  # Previous center
                    bundle[center_idx - 1].critique,
                    judge_lm
                )

                if abs(semantic_score) > 1e-6:  # Avoid division by zero
                    lambda_obs = actual_improvement / abs(semantic_score)
                    lambda_current = max(
                        self.lambda_min,
                        min(
                            self.lambda_max,
                            (1 - self.lambda_gamma) * lambda_current + self.lambda_gamma * lambda_obs
                        )
                    )
                    logger.info(f"Updated λ: {lambda_current:.3f} (observed: {lambda_obs:.3f})")

                # Generate critique for new center
                critique = self._generate_critique(best_candidate, trainset, best_candidate_prompts, critic_lm)

            else:
                # NULL STEP - Reject candidate, refine model
                logger.info("✗ NULL STEP: Refinement only (candidate rejected)")
                num_null += 1
                consecutive_null += 1

                # Generate critique explaining why candidate failed
                critique = self._generate_failure_critique(
                    best_candidate,
                    trainset,
                    best_candidate_prompts,
                    target_loss=bundle[center_idx].loss,
                    critic_lm=critic_lm
                )

            # Add to bundle (both serious and null steps)
            bundle.append(BundleEntry(
                prompt=best_candidate_prompts,
                loss=candidate_loss,
                critique=critique,
                iteration=iteration
            ))

            # Termination check
            if consecutive_null >= self.max_null_steps:
                logger.info(f"Terminating: {self.max_null_steps} consecutive null steps")
                break

            # Only terminate on non-positive predicted improvement after a few iterations
            # This gives the optimizer time to build a useful bundle
            if iteration >= 3 and predicted_improvement <= 0:
                logger.info("Terminating: Non-positive predicted improvement (after iteration 3+)")
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
            num_null_steps=num_null
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
                        logger.debug(f"Applied instruction to {pred_name}: {repr(instruction)[:100]}")
        return program

    def _evaluate_program(self, program: Module, examples: list[Example]) -> float:
        """Evaluate program on examples using the metric (robust loss estimation)."""
        total_loss = 0.0
        logger.info(f"Evaluating program on {len(examples)} examples...")
        for idx, ex in enumerate(examples):
            try:
                logger.debug(f"  Example {idx+1}/{len(examples)}: Running program...")
                pred = program(**ex.inputs())
                logger.debug(f"  Example {idx+1}/{len(examples)}: Computing metric...")
                score = self.metric(ex, pred, None)
                # Convert score to loss (assuming metric returns [0,1] with 1=best)
                loss = 1.0 - float(score)
                total_loss += loss
                logger.debug(f"  Example {idx+1}/{len(examples)}: score={score:.3f}, loss={loss:.3f}")
            except Exception as e:
                logger.warning(f"Evaluation error on example {idx+1}: {e}")
                logger.debug(f"  Example inputs: {ex.inputs()}")
                total_loss += 1.0  # Max penalty
        avg_loss = total_loss / len(examples)
        logger.info(f"Program evaluation complete: avg_loss={avg_loss:.4f}")
        return avg_loss

    def _generate_critique(
        self,
        program: Module,
        examples: list[Example],
        prompts: dict[str, str],
        lm: LM
    ) -> str:
        """Generate critique identifying weaknesses in the program."""
        # Sample a few failure cases
        failures = []
        for ex in random.sample(examples, min(3, len(examples))):
            try:
                pred = program(**ex.inputs())
                score = self.metric(ex, pred, None)
                if score < 0.8:  # Consider as failure
                    failures.append({
                        "input": str(ex.inputs()),
                        "expected": str(ex.labels()),
                        "predicted": str(pred),
                        "score": float(score)
                    })
            except:
                pass

        if not failures:
            return "The prompt is performing well on the given examples."

        # Format critique generation prompt
        prompt_text = "\n\n".join([f"{k}: {v}" for k, v in prompts.items()])
        failures_text = "\n\n".join([
            f"Example {i+1}:\nInput: {f['input']}\nExpected: {f['expected']}\nPredicted: {f['predicted']}\nScore: {f['score']:.2f}"
            for i, f in enumerate(failures)
        ])

        critique_prompt = f"""You are an expert prompt engineer. Analyze the following prompt and failure cases to identify the single most critical weakness.

Current Prompt:
{prompt_text}

Failure Examples:
{failures_text}

Instructions:
1. Analyze why the prompt failed on these examples
2. Formulate a specific, actionable critique (e.g., "The prompt is too vague about output formatting")
3. Do NOT suggest a new prompt. Only state the critique.

Critique:"""

        with dspy.context(lm=lm, temperature=self.temperature):
            response = lm(critique_prompt)

        # LM returns a list of completions, get the first one
        return (response[0] if isinstance(response, list) else response).strip()

    def _generate_failure_critique(
        self,
        program: Module,
        examples: list[Example],
        prompts: dict[str, str],
        target_loss: float,
        critic_lm: LM
    ) -> str:
        """Generate critique explaining why a candidate failed to improve."""
        current_loss = self._evaluate_program(program, examples)

        prompt_text = "\n\n".join([f"{k}: {v}" for k, v in prompts.items()])

        critique_prompt = f"""You are an expert prompt engineer. A candidate prompt was tested but failed to improve performance.

Candidate Prompt:
{prompt_text}

Current Loss: {current_loss:.4f}
Target Loss: {target_loss:.4f}
(Lower is better. Candidate should achieve loss < {target_loss:.4f} but didn't)

Analyze why this candidate failed to improve performance. Provide a specific, actionable critique.

Critique:"""

        with dspy.context(lm=critic_lm, temperature=self.temperature):
            response = critic_lm(critique_prompt)

        # LM returns a list of completions, get the first one
        return (response[0] if isinstance(response, list) else response).strip()

    def _generate_candidates(
        self,
        center_program: Module,
        critique: str,
        lm: LM
    ) -> list[Module]:
        """Generate N candidate variations addressing the critique (Proposer)."""
        center_prompts = self._extract_prompts(center_program)
        prompt_text = "\n\n".join([f"{k}: {v}" for k, v in center_prompts.items()])

        proposer_prompt = f"""You are an intelligent optimizer. Generate {self.num_candidates} variations of the prompt that address the critique while preserving the original intent.

Current Prompt:
{prompt_text}

Critique: {critique}

Constraints:
1. Local edits only: Do not rewrite the entire prompt from scratch. Keep structure similar.
2. Focus: Address the critique directly
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
        logger.info(f"PROPOSER RAW RESPONSE (first 500 chars):")
        logger.info(f"{response_text[:500]}")
        logger.info(f"{'='*60}\n")

        # Parse candidates - improved parsing to handle various LM response formats
        candidates = []
        
        # Try splitting by "CANDIDATE" keyword first
        parts = response_text.split('CANDIDATE')
        
        for i, part in enumerate(parts):
            if i == 0 and not part.strip():
                continue  # Skip empty first part
            
            # Remove the candidate number/marker (e.g., "1:", "2:", etc.)
            lines = part.strip().split('\n')
            if lines:
                # Remove first line if it's just a number/colon
                first_line = lines[0].strip()
                if first_line and (first_line[0].isdigit() or first_line.startswith(':')):
                    # Check if there's actual text after the number
                    colon_idx = first_line.find(':')
                    if colon_idx >= 0 and len(first_line) > colon_idx + 1:
                        # Text on same line as "CANDIDATE N:"
                        lines[0] = first_line[colon_idx + 1:].strip()
                    else:
                        # Number/colon only, skip this line
                        lines = lines[1:]
            
            candidate_text = '\n'.join(lines).strip()
            
            # Only add if we have non-empty text
            if candidate_text:
                logger.info(f"Parsed candidate {len(candidates)+1}: {repr(candidate_text[:100])}")
                new_prompts = {k: candidate_text for k in center_prompts.keys()}
                candidates.append(self._build_program_from_prompts(center_program, new_prompts))

        # If parsing failed completely, try to extract at least some variations
        if len(candidates) == 0:
            logger.warning("Failed to parse any candidates from LM response, using center program")
            candidates.append(center_program.deepcopy())
        
        # Fill remaining slots with center program copies if needed
        while len(candidates) < self.num_candidates:
            logger.warning(f"Only parsed {len(candidates)} candidates, padding with center program")
            candidates.append(center_program.deepcopy())

        return candidates[:self.num_candidates]

    def _compute_semantic_score(
        self,
        candidate_prompts: dict[str, str],
        reference_prompts: dict[str, str],
        critique: str,
        lm: LM
    ) -> float:
        """
        Compute smoothed semantic score Ŝ_J(p, p_ref, c).

        Returns value in [-1, 1] indicating alignment with critique.
        """
        scores = []

        for _ in range(self.num_judge_samples):
            score = self._judge_single(candidate_prompts, reference_prompts, critique, lm)
            scores.append(score)

        return sum(scores) / len(scores)

    def _judge_single(
        self,
        candidate_prompts: dict[str, str],
        reference_prompts: dict[str, str],
        critique: str,
        lm: LM
    ) -> float:
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
• +1.0 (Strong Descent): Candidate completely resolves the issue
• +0.5 (Weak Descent): Candidate partially fixes the issue
• 0.0 (Orthogonal): Candidate ignores the critique
• -0.5 (Weak Regression): Candidate slightly worsens the issue
• -1.0 (Strong Regression): Candidate explicitly violates the critique

Output ONLY a single number between -1.0 and 1.0. No explanation.

Score:"""

        with dspy.context(lm=lm, temperature=0.0):  # Deterministic
            response = lm(judge_prompt)

        # LM returns a list of completions, get the first one
        response_text = (response[0] if isinstance(response, list) else response).strip()

        # Parse score
        try:
            score = float(response_text)
            return max(-1.0, min(1.0, score))
        except:
            logger.warning(f"Failed to parse judge score: {response_text}")
            return 0.0

    def _select_best_candidate(
        self,
        candidates: list[Module],
        bundle: list[BundleEntry],
        lambda_current: float,
        lm: LM
    ) -> tuple[Module, dict[str, str]]:
        """
        Select best candidate by minimizing cumulative semantic violation (Verifier).

        Returns: (best_program, best_prompts)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"VERIFIER: Evaluating {len(candidates)} candidates against bundle of {len(bundle)}")
        logger.info(f"  (This requires {len(candidates) * len(bundle) * self.num_judge_samples} judge LM calls)")
        logger.info(f"{'='*60}")

        best_violation = float('inf')
        best_candidate = candidates[0]
        best_prompts = self._extract_prompts(candidates[0])

        for idx, candidate in enumerate(candidates):
            candidate_prompts = self._extract_prompts(candidate)

            # Compute cumulative violation against all bundle entries
            total_violation = 0.0
            scores_detail = []
            for entry in bundle:
                score = self._compute_semantic_score(
                    candidate_prompts,
                    entry.prompt,
                    entry.critique,
                    lm
                )
                # Hinge loss: max(0, τ - score)
                violation = max(0, self.tau_margin - score)
                total_violation += violation
                scores_detail.append((score, violation))

            logger.info(f"  Candidate {idx+1}: total_violation={total_violation:.4f}")
            for i, (score, viol) in enumerate(scores_detail):
                logger.info(f"    vs bundle[{i}]: score={score:.3f}, violation={viol:.3f}")

            if total_violation < best_violation:
                best_violation = total_violation
                best_candidate = candidate
                best_prompts = candidate_prompts

        logger.info(f"\n✓ Selected candidate with violation: {best_violation:.4f}")
        logger.info(f"{'='*60}\n")
        return best_candidate, best_prompts

    def _compute_model_value(
        self,
        prompts: dict[str, str],
        bundle: list[BundleEntry],
        lambda_current: float,
        lm: LM
    ) -> float:
        """
        Compute cutting-plane model M_k(p) = max_i {F̃_i - λ·Ŝ_J(p, p_i, c_i)}.
        """
        model_value = float('-inf')

        for entry in bundle:
            score = self._compute_semantic_score(prompts, entry.prompt, entry.critique, lm)
            value = entry.loss - lambda_current * score
            model_value = max(model_value, value)

        return model_value
