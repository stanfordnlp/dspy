"""
AIME (American Invitational Mathematics Examination) dataset adapter.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import dspy
from datasets import load_dataset

from data_adapters.base import DatasetAdapter
from core.metrics import MetricRegistry

if TYPE_CHECKING:
    from dspy import Example


class AIMEAdapter(DatasetAdapter):
    """Adapter for the AIME dataset."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def load_dataset(self) -> tuple[list[Example], list[Example], list[Example]]:
        """Load and return train, validation, and test sets.

        The AIME exam consists of 2 problem sets of size 15 for each year.
        - Training data: AI-MO/aimo-validation-aime (2022-2024, 90 problems total)
        - Test data: MathArena/aime_2025 (30 problems, optionally repeated for stability)

        Returns:
            Tuple of (train_set, val_set, test_set) with appropriate input fields.
        """
        input_fields = self.get_input_fields(("problem",))

        # Load training data from previous years (2022-2024)
        train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
        train_split = [
            dspy.Example({
                "problem": x['problem'],
                'solution': x.get('solution', ''),  # May not always be present
                'answer': str(x['answer']),  # Ensure answer is a string
            })
            for x in train_split
        ]

        # Shuffle with seed for reproducibility
        train_seed = self.config.get("train_seed", 1)
        random.Random(train_seed).shuffle(train_split)
        tot_num = len(train_split)

        # Load test data from AIME 2025
        test_split = load_dataset("MathArena/aime_2025")['train']
        test_split = [
            dspy.Example({
                "problem": x['problem'],
                'answer': str(x['answer']),  # Ensure answer is a string
            })
            for x in test_split
        ]
        train_split = self.apply_input_fields(train_split, input_fields)
        test_split = self.apply_input_fields(test_split, input_fields)

        # Apply size configurations
        train_size = self.config.get("train_size", 45)
        dev_size = self.config.get("dev_size", 45)
        test_size = self.config.get("test_size", 30)
        test_repeat = self.config.get("test_repeat", 5)  # Repeat test set for stability

        # Split train data into train and val
        train_set = train_split[:train_size]
        val_set = train_split[train_size:train_size + dev_size]

        # Optionally repeat test set for statistical stability
        test_set = test_split[:test_size] * test_repeat

        return train_set, val_set, test_set

    def get_metric(self) -> Any:
        """Return the evaluation metric for this dataset."""
        return MetricRegistry.get_metric("aime", "standard")

    def get_gepa_metric(self) -> Any:
        """Return the GEPA-compatible metric for this dataset."""
        return MetricRegistry.get_metric("aime", "gepa")

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "aime"

    @property
    def uses_context(self) -> bool:
        """Return whether this dataset uses context as input."""
        return False  # AIME only uses problem as input
