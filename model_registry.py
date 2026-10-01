"""Compatibility wrapper for the model registry API.

ModelRegistry lives in lm15.models alongside ModelInfo. This module exists for
users who prefer importing it from lm15.model_registry.
"""

from .models import ModelRegistry

__all__ = ["ModelRegistry"]
