"""
Dataset adapters for prompt optimization experiments.
"""

from .base import DatasetAdapter
from .registry import DatasetRegistry
from .hotpotqa import HotPotQAAdapter

__all__ = [
    "DatasetAdapter",
    "DatasetRegistry", 
    "HotPotQAAdapter",
]