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

    optimizer_class_name = "SemanticBundleOptimization"
    optimizer_name = "sbo"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.result = None
        self.trace = None

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
        from dspy.teleprompt.sbo import SemanticBundleOptimization, SemanticBundleOptimizationLite

        # Setup LMs
        model_config = kwargs.get("model_config", {})
        model_name = model_config.get("name", "gpt-3.5-turbo")
        api_base = model_config.get("api_base", "http://localhost:11434")
        model_params = model_config.get("params", {})
        role_model_params = {
            key: value
            for key, value in model_params.items()
            if key not in {"temperature", "max_tokens", "cache"}
        }

        # Get LM configurations from params
        params = self.config.get("params", {})

        # Judge LM (for semantic scoring)
        judge_lm_name = params.get("judge_lm") or model_name
        judge_lm = dspy.LM(
            model=judge_lm_name,
            api_base=api_base,
            **role_model_params,
            temperature=params.get("judge_temperature", 0.7),
            max_tokens=params.get("judge_max_tokens", 10),  # Judge only outputs a number
            cache=params.get("judge_cache", False),
        )

        # Proposer LM (for generating candidates)
        proposer_lm_name = params.get("proposer_lm") or model_name
        proposer_lm = dspy.LM(
            model=proposer_lm_name,
            api_base=api_base,
            **role_model_params,
            temperature=params.get("proposer_temperature", 0.7),
            max_tokens=params.get("proposer_max_tokens", 2048),
        )

        # Critic LM (for generating critiques)
        critic_lm_name = params.get("critic_lm") or model_name
        critic_lm = dspy.LM(
            model=critic_lm_name,
            api_base=api_base,
            **role_model_params,
            temperature=params.get("critic_temperature", 0.7),
            max_tokens=params.get("critic_max_tokens", 1024),
        )

        optimizer_class = (
            SemanticBundleOptimizationLite
            if self.optimizer_class_name == "SemanticBundleOptimizationLite"
            else SemanticBundleOptimization
        )

        # Create SBO optimizer
        optimizer = optimizer_class(
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
            bundle_size=params.get("bundle_size", 10),
            active_bundle_size=params.get("active_bundle_size", 3),
            watchlist_size=params.get("watchlist_size", 2),
            active_tau_margin=params.get("active_tau_margin", 0.0),
            watchlist_tau_margin=params.get("watchlist_tau_margin", 0.0),
            active_violation_tolerance=params.get("active_violation_tolerance", 0.0),
            watchlist_violation_tolerance=params.get("watchlist_violation_tolerance", 0.0),
            lambda_stability_epsilon=params.get("lambda_stability_epsilon", 1e-6),
            tau_stop=params.get("tau_stop", 0.0),
            max_iterations=params.get("max_iterations", 50),
            max_null_steps=params.get("max_null_steps", 5),
            temperature=params.get("temperature", 0.7),
            judge_temperature=params.get("judge_temperature", 0.7),
            num_eval_samples=params.get("num_eval_samples", 1),
            eval_temperature=params.get("eval_temperature", 0.7),
            eval_cache=params.get("eval_cache", False),
            parse_failure_retries=params.get("parse_failure_retries", 0),
            parse_retry_temperature=params.get("parse_retry_temperature"),
            max_critique_examples=params.get("max_critique_examples", 3),
            max_critique_field_chars=params.get("max_critique_field_chars"),
            max_bundle_critique_chars=params.get("max_bundle_critique_chars", 900),
            stop_on_no_improving_candidate=params.get("stop_on_no_improving_candidate", True),
            enable_exact_null_cuts=params.get("enable_exact_null_cuts", True),
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
        self.trace = optimizer.result.trace if optimizer.result else None

        return optimized_program

    @property
    def name(self) -> str:
        """Return the optimizer name."""
        return self.optimizer_name

    def supports_gepa_metric(self) -> bool:
        """SBO uses standard metrics, not GEPA-style metrics."""
        return False


class SBOLiteAdapter(SBOAdapter):
    """Adapter for SBO-Lite qualitative critique-bundle optimization."""

    optimizer_class_name = "SemanticBundleOptimizationLite"
    optimizer_name = "sbo_lite"
