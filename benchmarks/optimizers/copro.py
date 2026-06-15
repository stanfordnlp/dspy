"""
COPRO optimizer adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optimizers.base import OptimizerAdapter

if TYPE_CHECKING:
    from dspy import Example, Module


class CoproAdapter(OptimizerAdapter):
    """Adapter for COPRO optimizer."""
    
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
        """Optimize the given program using COPRO.
        
        Args:
            program: DSPy program to optimize.
            train_set: Training examples.
            val_set: Validation examples.
            metric: Standard evaluation metric.
            **kwargs: Additional arguments.
            
        Returns:
            Optimized program.
        """
        from dspy.teleprompt import COPRO

        optimizer = COPRO(
            metric=metric,
            breadth=self.config["params"].get("breadth", 10),
            depth=self.config["params"].get("depth", 3),
        )

        return optimizer.compile(program, trainset=train_set, eval_kwargs={"devset": val_set})
    
    @property
    def name(self) -> str:
        """Return the optimizer name."""
        return "copro"
    
    def supports_gepa_metric(self) -> bool:
        """Return whether this optimizer supports GEPA-style metrics."""
        return False