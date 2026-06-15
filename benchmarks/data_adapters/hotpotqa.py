"""
HotPotQA dataset adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.datasets import HotPotQA

from data_adapters.base import DatasetAdapter
from core.metrics import MetricRegistry

if TYPE_CHECKING:
    from dspy import Example


class HotPotQAAdapter(DatasetAdapter):
    """Adapter for the HotPotQA dataset."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
    def load_dataset(self) -> tuple[list[Example], list[Example], list[Example]]:
        """Load and return train, validation, and test sets.
        
        Returns:
            Tuple of (train_set, val_set, test_set) with appropriate input fields.
        """
        dataset = HotPotQA(
            train_seed=self.config.get("train_seed", 1),
            train_size=self.config.get("train_size", 50),
            eval_seed=self.config.get("eval_seed", 2023),
            dev_size=self.config.get("dev_size", 100),
            test_size=self.config.get("test_size", 0),
            only_hard_examples=True,  # Always use hard examples for proper benchmarking
            keep_details=self.config.get("keep_details", True),
        )

        # Determine input fields based on whether context is used
        use_context = self.config.get("use_context", True)
        input_fields = ("question", "context") if use_context else ("question",)
        
        train_set = [ex.with_inputs(*input_fields) for ex in dataset.train]
        val_set = [ex.with_inputs(*input_fields) for ex in dataset.dev]
        test_set = [ex.with_inputs(*input_fields) for ex in dataset.test]

        return train_set, val_set, test_set
    
    def get_metric(self) -> Any:
        """Return the evaluation metric for this dataset."""
        return MetricRegistry.get_metric("hotpotqa", "standard")
    
    def get_gepa_metric(self) -> Any:
        """Return the GEPA-compatible metric for this dataset."""
        return MetricRegistry.get_metric("hotpotqa", "gepa")
    
    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "hotpotqa"
    
    @property
    def uses_context(self) -> bool:
        """Return whether this dataset uses context as input."""
        return self.config.get("use_context", True)