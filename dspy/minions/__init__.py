#!/usr/bin/env python3
"""
DSPy Minions package: Structured output support for DSPy.

This package contains an implementation of structured data extraction using MinionsLM,
enabling more efficient and reliable extraction of structured data from language models.

The module includes the StructuredMinionsLM class that extends MinionsLM with capabilities
for structured data extraction and processing.
"""

from dspy.minions.structured_minions import StructuredMinionsLM, create_structured_minions

__all__ = [
    "StructuredMinionsLM",
    "create_structured_minions",
] 