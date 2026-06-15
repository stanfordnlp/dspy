"""
Semantic Bundle Optimization (SBO) adapter for benchmark framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from optimizers.base import OptimizerAdapter

if TYPE_CHECKING:
    from dspy import Example, Module


class SBOAdapter(OptimizerAdapter):
    """Adapter for Semantic Bundle Optimization (SBO)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def optimize(
        self,
        program: Module,
        train_set: list[Example],
        val_set: list[Example],
        metric: Any,
        **kwargs
    ) -> Module:
        """Optimize the given program using SBO.

        Args:
            program: DSPy program to optimize.
            train_set: Training examples (used for critique generation).
            val_set: Validation examples (used for robust loss estimation).
            metric: Evaluation metric.
            **kwargs: Additional arguments.

        Returns:
            Optimized program.
        """
        from dspy.teleprompt.sbo import SemanticBundleOptimization

        # Setup LMs
        model_config = kwargs.get("model_config", {})
        model_name = model_config.get("name", "gpt-3.5-turbo")
        api_base = model_config.get("api_base", "http://localhost:11434")

        # Get LM configurations from params
        params = self.config.get("params", {})

        # Judge LM (for semantic scoring)
        judge_lm_name = params.get("judge_lm") or model_name
        judge_lm = dspy.LM(
            model=judge_lm_name,
            api_base=api_base,
            temperature=0.0,  # Deterministic for judge
            max_tokens=10,  # Judge only outputs a number
        )

        # Proposer LM (for generating candidates)
        proposer_lm_name = params.get("proposer_lm") or model_name
        proposer_lm = dspy.LM(
            model=proposer_lm_name,
            api_base=api_base,
            temperature=params.get("proposer_temperature", 0.7),
            max_tokens=2048,
        )

        # Critic LM (for generating critiques)
        critic_lm_name = params.get("critic_lm") or model_name
        critic_lm = dspy.LM(
            model=critic_lm_name,
            api_base=api_base,
            temperature=params.get("critic_temperature", 0.7),
            max_tokens=1024,
        )

        # Create SBO optimizer
        optimizer = SemanticBundleOptimization(
            metric=metric,
            judge_lm=judge_lm,
            proposer_lm=proposer_lm,
            critic_lm=critic_lm,
            num_candidates=params.get("num_candidates", 5),
            num_judge_samples=params.get("num_judge_samples", 3),
            descent_param=params.get("descent_param", 0.1),
            lambda_init=params.get("lambda_init", 1.0),
            lambda_min=params.get("lambda_min", 0.1),
            lambda_max=params.get("lambda_max", 10.0),
            lambda_gamma=params.get("lambda_gamma", 0.3),
            tau_margin=params.get("tau_margin", 0.5),
            max_iterations=params.get("max_iterations", 50),
            max_null_steps=params.get("max_null_steps", 5),
            temperature=params.get("temperature", 0.7),
            track_stats=params.get("track_stats", True),
        )

        # Run optimization
        optimized_program = optimizer.compile(
            program,
            trainset=train_set,
            valset=val_set
        )

        # Store result for analysis
        self.result = optimizer.result

        return optimized_program

    @property
    def name(self) -> str:
        """Return the optimizer name."""
        return "sbo"

    def supports_gepa_metric(self) -> bool:
        """SBO uses standard metrics, not GEPA-style metrics."""
        return False
