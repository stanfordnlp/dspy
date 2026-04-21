"""
Baseline optimizer adapter (no optimization).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optimizers.base import OptimizerAdapter

if TYPE_CHECKING:
    from dspy import Example, Module


class BaselineAdapter(OptimizerAdapter):
    """Adapter for baseline (no optimization)."""
    
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
        """Return the program unchanged (no optimization).
        
        Args:
            program: DSPy program to optimize.
            train_set: Training examples (unused).
            val_set: Validation examples (unused).
            metric: Evaluation metric (unused).
            **kwargs: Additional arguments (unused).
            
        Returns:
            Original program unchanged.
        """
        # Return the original program unchanged
        return program
    
    @property
    def name(self) -> str:
        """Return the optimizer name."""
        return "baseline"
    
    def supports_gepa_metric(self) -> bool:
        """Return whether this optimizer supports GEPA-style metrics."""
        return False