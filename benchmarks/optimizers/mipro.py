"""
MIPROv2 optimizer adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optimizers.base import OptimizerAdapter

if TYPE_CHECKING:
    from dspy import Example, Module


class MiproAdapter(OptimizerAdapter):
    """Adapter for MIPROv2 optimizer."""
    
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
        """Optimize the given program using MIPROv2.
        
        Args:
            program: DSPy program to optimize.
            train_set: Training examples.
            val_set: Validation examples.
            metric: Standard evaluation metric.
            **kwargs: Additional arguments.
            
        Returns:
            Optimized program.
        """
        from dspy.teleprompt import MIPROv2

        optimizer = MIPROv2(
            metric=metric,
            auto=self.config.get("auto", "light"),
            num_threads=self.config.get("num_threads", 1),
        )

        return optimizer.compile(program, trainset=train_set, valset=val_set)
    
    @property
    def name(self) -> str:
        """Return the optimizer name."""
        return "mipro"
    
    def supports_gepa_metric(self) -> bool:
        """Return whether this optimizer supports GEPA-style metrics."""
        return False