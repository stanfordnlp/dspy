"""Cost tracking and budget-aware optimization utilities for DSPy.

This module provides cost tracking capabilities that build on top of DSPy's
existing usage tracking infrastructure. It leverages litellm's model cost
database to automatically calculate costs per LM call, and supports budget
limits to warn or stop when a spending threshold is exceeded.

Example usage:

    with dspy.track_cost() as cost_tracker:
        result = my_dspy_program(input_data)

    print(f"Total cost: ${cost_tracker.total_cost:.4f}")
    print(cost_tracker.get_costs_by_model())
"""

import logging
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

import litellm

from dspy.dsp.utils.settings import settings

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when total cost exceeds the configured budget limit."""

    def __init__(self, budget: float, total_cost: float, model: str | None = None):
        self.budget = budget
        self.total_cost = total_cost
        self.model = model
        model_info = f" (triggered by {model})" if model else ""
        super().__init__(
            f"Budget of ${budget:.4f} exceeded: total cost is ${total_cost:.4f}{model_info}"
        )


# Default pricing for models not found in litellm's cost map.
# Users can override this via CostTracker.set_model_pricing().
_DEFAULT_PRICING: dict[str, dict[str, float]] = {}


class CostTracker:
    """Tracks LLM costs automatically during DSPy program execution.

    The CostTracker calculates costs per LM call using token counts from
    usage data and per-model pricing from litellm's cost database. It
    supports cumulative cost tracking, per-model cost breakdown, budget
    limits (warn or hard stop), and custom pricing overrides.

    Attributes:
        total_cost: The cumulative cost in USD across all tracked calls.
        budget: Optional maximum budget in USD. When set, the tracker
            will warn or raise an error when the budget is exceeded.
        budget_action: What to do when the budget is exceeded. One of
            "warn" (log a warning) or "stop" (raise BudgetExceededError).
    """

    def __init__(
        self,
        budget: float | None = None,
        budget_action: str = "warn",
    ):
        """Initialize a CostTracker.

        Args:
            budget: Optional maximum budget in USD. Set to None for no limit.
            budget_action: Action when budget is exceeded. Either "warn"
                (default) to emit a warning, or "stop" to raise
                BudgetExceededError.
        """
        if budget_action not in ("warn", "stop"):
            raise ValueError(f"budget_action must be 'warn' or 'stop', got '{budget_action}'")
        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")

        self.budget = budget
        self.budget_action = budget_action
        self.total_cost: float = 0.0

        # Per-model cost tracking: {model_name: total_cost_usd}
        self._model_costs: dict[str, float] = defaultdict(float)
        # Per-model token tracking: {model_name: {prompt_tokens: N, completion_tokens: N}}
        self._model_tokens: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        # Detailed call log: list of per-call entries
        self._call_log: list[dict[str, Any]] = []
        # Custom per-model pricing overrides
        self._custom_pricing: dict[str, dict[str, float]] = {}
        # Thread safety lock
        self._lock = threading.Lock()
        # Whether budget warning has been emitted (to avoid spamming)
        self._budget_warned = False

    def set_model_pricing(
        self,
        model: str,
        input_cost_per_token: float,
        output_cost_per_token: float,
    ) -> None:
        """Set custom pricing for a specific model.

        This overrides litellm's built-in pricing for the given model.

        Args:
            model: The model name (e.g., "openai/gpt-4o-mini").
            input_cost_per_token: Cost in USD per input (prompt) token.
            output_cost_per_token: Cost in USD per output (completion) token.
        """
        self._custom_pricing[model] = {
            "input_cost_per_token": input_cost_per_token,
            "output_cost_per_token": output_cost_per_token,
        }

    def _get_model_pricing(self, model: str) -> dict[str, float] | None:
        """Look up pricing for a model.

        Checks custom pricing first, then litellm's cost map, then
        the global default pricing registry.

        Returns:
            A dict with 'input_cost_per_token' and 'output_cost_per_token',
            or None if no pricing is found.
        """
        # Check custom pricing first
        if model in self._custom_pricing:
            return self._custom_pricing[model]

        # Check litellm's model cost map
        model_info = _lookup_litellm_pricing(model)
        if model_info is not None:
            return model_info

        # Check the global default pricing registry
        if model in _DEFAULT_PRICING:
            return _DEFAULT_PRICING[model]

        return None

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        """Calculate the cost of a single LM call.

        Args:
            model: The model name.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            The cost in USD, or None if pricing is unavailable.
        """
        pricing = self._get_model_pricing(model)
        if pricing is None:
            return None

        input_cost = prompt_tokens * pricing["input_cost_per_token"]
        output_cost = completion_tokens * pricing["output_cost_per_token"]
        return input_cost + output_cost

    def add_usage(self, model: str, usage_entry: dict[str, Any]) -> None:
        """Record usage from a single LM call and update cost tracking.

        This method is called automatically by the DSPy LM infrastructure
        when cost tracking is active.

        Args:
            model: The model name (e.g., "openai/gpt-4o-mini").
            usage_entry: A dict containing token usage, typically with
                keys like 'prompt_tokens', 'completion_tokens', 'total_tokens'.
        """
        if not usage_entry:
            return

        prompt_tokens = usage_entry.get("prompt_tokens", 0) or 0
        completion_tokens = usage_entry.get("completion_tokens", 0) or 0
        total_tokens = usage_entry.get("total_tokens", 0) or 0

        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        with self._lock:
            # Update token counts
            self._model_tokens[model]["prompt_tokens"] += prompt_tokens
            self._model_tokens[model]["completion_tokens"] += completion_tokens
            self._model_tokens[model]["total_tokens"] += total_tokens

            # Update cost
            if cost is not None:
                self._model_costs[model] += cost
                self.total_cost += cost

            # Add to call log
            self._call_log.append({
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
            })

            # Check budget
            self._check_budget(model)

    def _check_budget(self, model: str | None = None) -> None:
        """Check if the total cost exceeds the budget.

        Must be called while holding self._lock.
        """
        if self.budget is None:
            return

        if self.total_cost > self.budget:
            if self.budget_action == "stop":
                raise BudgetExceededError(
                    budget=self.budget,
                    total_cost=self.total_cost,
                    model=model,
                )
            elif self.budget_action == "warn" and not self._budget_warned:
                self._budget_warned = True
                warnings.warn(
                    f"Budget of ${self.budget:.4f} exceeded: total cost is "
                    f"${self.total_cost:.4f}. Set budget_action='stop' to halt execution.",
                    stacklevel=4,
                )

    def get_costs_by_model(self) -> dict[str, float]:
        """Get a breakdown of costs by model.

        Returns:
            A dict mapping model names to their total cost in USD.
        """
        with self._lock:
            return dict(self._model_costs)

    def get_tokens_by_model(self) -> dict[str, dict[str, int]]:
        """Get a breakdown of token usage by model.

        Returns:
            A dict mapping model names to their token counts
            (prompt_tokens, completion_tokens, total_tokens).
        """
        with self._lock:
            return {model: dict(tokens) for model, tokens in self._model_tokens.items()}

    def get_call_log(self) -> list[dict[str, Any]]:
        """Get the detailed log of all tracked LM calls.

        Returns:
            A list of dicts, each containing model, token counts, and cost
            for a single call.
        """
        with self._lock:
            return list(self._call_log)

    @property
    def num_calls(self) -> int:
        """Total number of tracked LM calls."""
        with self._lock:
            return len(self._call_log)

    @property
    def budget_remaining(self) -> float | None:
        """Remaining budget in USD, or None if no budget is set."""
        if self.budget is None:
            return None
        return max(0.0, self.budget - self.total_cost)

    @property
    def budget_utilization(self) -> float | None:
        """Budget utilization as a fraction (0.0 to 1.0+), or None if no budget."""
        if self.budget is None:
            return None
        return self.total_cost / self.budget

    def get_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of cost tracking data.

        Returns:
            A dict containing total cost, per-model costs, per-model tokens,
            number of calls, and budget information.
        """
        with self._lock:
            summary: dict[str, Any] = {
                "total_cost": self.total_cost,
                "num_calls": len(self._call_log),
                "costs_by_model": dict(self._model_costs),
                "tokens_by_model": {
                    model: dict(tokens) for model, tokens in self._model_tokens.items()
                },
            }
            if self.budget is not None:
                summary["budget"] = self.budget
                summary["budget_remaining"] = max(0.0, self.budget - self.total_cost)
                summary["budget_utilization"] = self.total_cost / self.budget
            return summary

    def get_total_tokens(self) -> dict[str, dict[str, int]]:
        """Get total token usage by model.

        This method provides compatibility with the UsageTracker interface
        so that CostTracker can be used as a drop-in replacement via the
        settings.usage_tracker slot.

        Returns:
            A dict mapping model names to their aggregated token counts.
        """
        with self._lock:
            return {model: dict(tokens) for model, tokens in self._model_tokens.items()}

    def reset(self) -> None:
        """Reset all tracked costs and usage data."""
        with self._lock:
            self.total_cost = 0.0
            self._model_costs.clear()
            self._model_tokens.clear()
            self._call_log.clear()
            self._budget_warned = False

    def __repr__(self) -> str:
        budget_str = f", budget=${self.budget:.4f}" if self.budget else ""
        return f"CostTracker(total_cost=${self.total_cost:.4f}, num_calls={self.num_calls}{budget_str})"


def _lookup_litellm_pricing(model: str) -> dict[str, float] | None:
    """Look up model pricing from litellm's cost database.

    Tries the model name as-is, then strips the provider prefix
    (e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini").

    Returns:
        A dict with 'input_cost_per_token' and 'output_cost_per_token',
        or None if not found.
    """
    for name in [model, model.split("/", 1)[-1] if "/" in model else None]:
        if name is None:
            continue
        info = litellm.model_cost.get(name)
        if info and "input_cost_per_token" in info and "output_cost_per_token" in info:
            return {
                "input_cost_per_token": info["input_cost_per_token"],
                "output_cost_per_token": info["output_cost_per_token"],
            }
    return None


def register_model_pricing(
    model: str,
    input_cost_per_token: float,
    output_cost_per_token: float,
) -> None:
    """Register default pricing for a model globally.

    This is useful for models that are not in litellm's cost database.
    Pricing registered here will be used by all CostTracker instances
    unless overridden by CostTracker.set_model_pricing().

    Args:
        model: The model name (e.g., "my-provider/custom-model").
        input_cost_per_token: Cost in USD per input token.
        output_cost_per_token: Cost in USD per output token.
    """
    _DEFAULT_PRICING[model] = {
        "input_cost_per_token": input_cost_per_token,
        "output_cost_per_token": output_cost_per_token,
    }


@contextmanager
def track_cost(
    budget: float | None = None,
    budget_action: str = "warn",
) -> Generator[CostTracker, None, None]:
    """Context manager for tracking LLM costs during DSPy program execution.

    Automatically tracks token usage and calculates costs for all LM calls
    made within the context. Optionally enforces a budget limit.

    Args:
        budget: Optional maximum budget in USD. Set to None for no limit.
        budget_action: What to do when budget is exceeded. Either "warn"
            (default) to emit a warning, or "stop" to raise BudgetExceededError.

    Yields:
        A CostTracker instance with cost data populated as calls are made.

    Example:
        >>> with track_cost(budget=1.00) as tracker:
        ...     result = my_program(input_data)
        >>> print(f"Total cost: ${tracker.total_cost:.4f}")
        >>> print(tracker.get_costs_by_model())
    """
    tracker = CostTracker(budget=budget, budget_action=budget_action)

    with settings.context(usage_tracker=tracker, track_usage=True):
        yield tracker
