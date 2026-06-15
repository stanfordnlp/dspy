"""
BootstrapFewShot optimizer adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optimizers.base import OptimizerAdapter

if TYPE_CHECKING:
    from dspy import Example, Module


class BootstrapAdapter(OptimizerAdapter):
    """Adapter for BootstrapFewShot optimizer."""
    
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
        """Optimize the given program using BootstrapFewShot.
        
        Args:
            program: DSPy program to optimize.
            train_set: Training examples.
            val_set: Validation examples (unused by BootstrapFewShot).
            metric: Standard evaluation metric.
            **kwargs: Additional arguments.
            
        Returns:
            Optimized program.
        """
        from dspy.teleprompt import BootstrapFewShot

        optimizer = BootstrapFewShot(
            metric=metric,
            max_rounds=self.config["params"].get("max_rounds", 3),
            max_bootstrapped_demos=self.config["params"].get("max_demos", 4),
        )

        return optimizer.compile(program, trainset=train_set)
    
    @property
    def name(self) -> str:
        """Return the optimizer name."""
        return "bootstrap"
    
    def supports_gepa_metric(self) -> bool:
        """Return whether this optimizer supports GEPA-style metrics."""
        return False