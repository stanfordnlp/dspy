"""
GEPA optimizer adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from optimizers.base import OptimizerAdapter

if TYPE_CHECKING:
    from dspy import Example, Module


class GepaAdapter(OptimizerAdapter):
    """Adapter for GEPA optimizer."""
    
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
        """Optimize the given program using GEPA.
        
        Args:
            program: DSPy program to optimize.
            train_set: Training examples.
            val_set: Validation examples.
            metric: GEPA-compatible evaluation metric.
            **kwargs: Additional arguments.
            
        Returns:
            Optimized program.
        """
        from dspy import GEPA
        
        # Setup reflection model
        model_config = kwargs.get("model_config", {})
        reflection_model = self.config["params"].get("reflection_lm") or model_config.get("name", "gpt-3.5-turbo")
        api_base = model_config.get("api_base", "http://localhost:11434")
        
        reflection_lm = dspy.LM(
            model=reflection_model,
            api_base=api_base,
            temperature=1.0,
            max_tokens=4096,  # Ensure enough tokens for instruction generation
        )

        optimizer = GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            max_metric_calls=self.config["params"].get("max_metric_calls", 50),
            num_threads=self.config.get("num_threads", 1),
            track_stats=self.config["params"].get("track_stats", True),
            reflection_minibatch_size=self.config["params"].get("reflection_minibatch_size", 3),
        )

        return optimizer.compile(program, trainset=train_set, valset=val_set)
    
    @property
    def name(self) -> str:
        """Return the optimizer name."""
        return "gepa"
    
    def supports_gepa_metric(self) -> bool:
        """Return whether this optimizer supports GEPA-style metrics."""
        return True