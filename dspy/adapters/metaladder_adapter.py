"""MetaLadder adapter for enhancing mathematical reasoning through analogical learning.

This module implements the MetaLadder framework as described in the paper
"MetaLadder: Ascending Mathematical Solution Quality via Analogical-Problem Reasoning Transfer".
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import re
import hashlib
from functools import lru_cache

from dspy.adapters.base import Adapter
from dspy.adapters.types.response import AdapterResponse
from dspy.dsp.utils import normalize_text
from dspy.teleprompt import BootstrapFewShot
from dspy.primitives.program import Module


@lru_cache(maxsize=1000)
def _get_cache_key(text: str) -> str:
    """Generate a stable cache key for a given text.
    
    Args:
        text: The text to generate a cache key for.
        
    Returns:
        A stable hash of the text.
    """
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class MetaProblem:
    """A class representing a meta problem for the MetaLadder adapter.

    Attributes:
        problem_type: The type of the problem.
        meta_problem: The meta problem description.
        restatement: The restatement of the problem.
    """
    problem_type: str
    meta_problem: str
    restatement: str

    def __hash__(self) -> int:
        """Generate a hash for the MetaProblem instance.

        Returns:
            int: The hash value.
        """
        return hash((self.problem_type, self.meta_problem, self.restatement))


class MetaLadderAdapter(Adapter):
    """An adapter that implements the MetaLadder approach for mathematical reasoning.

    This adapter enhances mathematical reasoning through analogical learning by:
    1. Identifying the problem type
    2. Generating a meta problem
    3. Restating the problem
    4. Using either a shortcut or full reasoning path

    Attributes:
        model (Module): The language model to use.
        optimizer (Optional[BootstrapFewShot]): The optimizer for improving prompts.
        use_shortcut (bool): Whether to use shortcut inference.
        max_tokens (int): Maximum number of tokens for responses.
        cache_size (int): Size of the LRU cache for method results.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optional[BootstrapFewShot] = None,
        use_shortcut: bool = True,
        max_tokens: int = 1000,
        cache_size: int = 1000,
    ) -> None:
        """Initialize the MetaLadderAdapter.

        Args:
            model: The language model to use.
            optimizer: Optional optimizer for improving prompts.
            use_shortcut: Whether to use shortcut inference.
            max_tokens: Maximum number of tokens for responses.
            cache_size: Size of the LRU cache for method results.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.use_shortcut = use_shortcut
        self.max_tokens = max_tokens

        # Initialize cached methods
        self._identify_problem_type = self._create_cached_method(
            self._identify_problem_type_impl, cache_size
        )
        self._generate_meta_problem = self._create_cached_method(
            self._generate_meta_problem_impl, cache_size
        )
        self._restate_problem = self._create_cached_method(
            self._restate_problem_impl, cache_size
        )

    def _create_cached_method(self, method: Any, cache_size: int) -> Any:
        """Create a cached version of a method.

        Args:
            method: The method to cache.
            cache_size: Size of the LRU cache.

        Returns:
            The cached method.
        """
        return lru_cache(maxsize=cache_size)(method)

    def _call_model(self, prompt: str) -> str:
        """Call the model with a prompt.

        Args:
            prompt: The input prompt.

        Returns:
            The model's response.
        """
        if self.optimizer:
            return self.optimizer.compile(self.model, trainset=[prompt])
        return self.model.__call__(prompt)

    def _identify_problem_type_impl(self, problem: str) -> str:
        """Identify the type of mathematical problem.

        Args:
            problem: The problem description.

        Returns:
            The identified problem type.
        """
        prompt = f"Identify the type of this math problem: {problem}"
        return self._call_model(prompt)

    def _generate_meta_problem_impl(self, problem_type: str, problem: str) -> str:
        """Generate a meta problem description.

        Args:
            problem_type: The type of problem.
            problem: The original problem.

        Returns:
            The meta problem description.
        """
        prompt = f"Generate a meta problem for this {problem_type} problem: {problem}"
        return self._call_model(prompt)

    def _restate_problem_impl(
        self, problem_type: str, meta_problem: str, problem: str
    ) -> str:
        """Restate the problem using the meta problem structure.

        Args:
            problem_type: The type of problem.
            meta_problem: The meta problem description.
            problem: The original problem.

        Returns:
            The restated problem.
        """
        prompt = (
            f"Restate this {problem_type} problem using the structure of the meta problem.\n"
            f"Meta problem: {meta_problem}\n"
            f"Problem: {problem}"
        )
        return self._call_model(prompt)

    def forward(self, prompt: str) -> Tuple[str, Optional[MetaProblem]]:
        """Process a prompt using the MetaLadder approach.

        Args:
            prompt: The input prompt.

        Returns:
            A tuple containing:
            - The model's response
            - The MetaProblem object (if not using shortcut)
        """
        if self.use_shortcut:
            return self._call_model(prompt), None

        # Full reasoning path
        problem_type = self._identify_problem_type(prompt)
        meta_problem = self._generate_meta_problem(problem_type, prompt)
        restatement = self._restate_problem(problem_type, meta_problem, prompt)

        meta_problem_obj = MetaProblem(
            problem_type=problem_type,
            meta_problem=meta_problem,
            restatement=restatement,
        )

        response = self._call_model(restatement)
        return response, meta_problem_obj

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._identify_problem_type.cache_clear()
        self._generate_meta_problem.cache_clear()
        self._restate_problem.cache_clear()
