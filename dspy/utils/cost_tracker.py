"""Cost tracking utilities for DSPy.

Provides automatic cost tracking for LLM calls, enabling users to:
- Track costs automatically during development and inference
- Get per-model cost breakdowns
- Set budget limits and receive warnings when exceeded
"""

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

from dspy.dsp.utils.settings import settings


class BudgetExceededError(Exception):
    """Raised when the budget limit is exceeded."""

    def __init__(self, current_cost: float, budget: float, model: str | None = None):
        self.current_cost = current_cost
        self.budget = budget
        self.model = model
        model_msg = f" for model '{model}'" if model else ""
        super().__init__(
            f"Budget exceeded{model_msg}: current cost ${current_cost:.4f} exceeds budget ${budget:.4f}"
        )


class CostTracker:
    """Tracks LLM cost data within a context.

    Example:
        >>> from dspy.utils.cost_tracker import track_cost
        >>> with track_cost() as cost_tracker:
        ...     result = my_dspy_program(input_data)
        >>> print(f"Total cost: ${cost_tracker.total_cost:.4f}")
        >>> print(f"By model: {cost_tracker.get_costs_by_model()}")
    """

    def __init__(self, budget: float | None = None, budget_per_model: dict[str, float] | None = None):
        """Initialize the cost tracker.

        Args:
            budget: Optional total budget limit in dollars. If set, raises BudgetExceededError
                when the total cost exceeds this limit.
            budget_per_model: Optional dict mapping model names to their individual budget limits.
                If set, raises BudgetExceededError when any model exceeds its limit.
        """
        # Map of LM name to list of cost entries
        # {
        #     "openai/gpt-4o-mini": [0.00015, 0.00012, ...],
        # }
        self.cost_data: dict[str, list[float]] = defaultdict(list)

        # Map of LM name to usage data (for reference)
        self.usage_data: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # Budget constraints
        self.budget = budget
        self.budget_per_model = budget_per_model or {}

        # Track number of calls per model
        self.call_counts: dict[str, int] = defaultdict(int)

    def add_cost(self, model: str, cost: float | None, usage: dict[str, Any] | None = None) -> None:
        """Add a cost entry to the tracker.

        Args:
            model: The model name (e.g., "openai/gpt-4o-mini")
            cost: The cost in dollars for this call. Can be None if cost is not available.
            usage: Optional usage data (tokens) for reference.
        """
        if cost is not None and cost > 0:
            self.cost_data[model].append(cost)

            # Check budget constraints
            self._check_budget(model)

        if usage:
            self.usage_data[model].append(usage)

        self.call_counts[model] += 1

    def _check_budget(self, model: str) -> None:
        """Check if budget limits have been exceeded."""
        # Check per-model budget
        if model in self.budget_per_model:
            model_cost = sum(self.cost_data[model])
            if model_cost > self.budget_per_model[model]:
                raise BudgetExceededError(model_cost, self.budget_per_model[model], model)

        # Check total budget
        if self.budget is not None:
            if self.total_cost > self.budget:
                raise BudgetExceededError(self.total_cost, self.budget)

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all models.

        Returns:
            Total cost in dollars.
        """
        return sum(sum(costs) for costs in self.cost_data.values())

    def get_costs_by_model(self) -> dict[str, float]:
        """Get total cost for each model.

        Returns:
            Dict mapping model names to their total costs.
        """
        return {model: sum(costs) for model, costs in self.cost_data.items()}

    def get_cost_summary(self) -> dict[str, Any]:
        """Get a comprehensive cost summary.

        Returns:
            Dict containing:
                - total_cost: Total cost across all models
                - costs_by_model: Per-model cost breakdown
                - calls_by_model: Number of calls per model
                - average_cost_per_call: Average cost per call for each model
                - usage_by_model: Token usage summary per model
        """
        costs_by_model = self.get_costs_by_model()

        # Calculate average cost per call
        avg_cost_per_call = {}
        for model, count in self.call_counts.items():
            if count > 0 and model in costs_by_model:
                avg_cost_per_call[model] = costs_by_model[model] / count

        # Summarize usage data
        usage_summary = {}
        for model, usage_list in self.usage_data.items():
            if usage_list:
                total_prompt = sum(u.get("prompt_tokens", 0) or 0 for u in usage_list)
                total_completion = sum(u.get("completion_tokens", 0) or 0 for u in usage_list)
                total_tokens = sum(u.get("total_tokens", 0) or 0 for u in usage_list)
                usage_summary[model] = {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_tokens,
                }

        return {
            "total_cost": self.total_cost,
            "costs_by_model": costs_by_model,
            "calls_by_model": dict(self.call_counts),
            "average_cost_per_call": avg_cost_per_call,
            "usage_by_model": usage_summary,
        }

    def __repr__(self) -> str:
        costs = self.get_costs_by_model()
        costs_str = ", ".join(f"{m}: ${c:.4f}" for m, c in costs.items())
        return f"CostTracker(total=${self.total_cost:.4f}, by_model={{{costs_str}}})"


@contextmanager
def track_cost(
    budget: float | None = None,
    budget_per_model: dict[str, float] | None = None,
) -> Generator[CostTracker, None, None]:
    """Context manager for tracking LLM costs.

    Args:
        budget: Optional total budget limit in dollars.
        budget_per_model: Optional dict mapping model names to their budget limits.

    Yields:
        CostTracker instance for accessing cost data.

    Example:
        >>> with track_cost() as tracker:
        ...     result = my_dspy_program(input_data)
        >>> print(f"Total cost: ${tracker.total_cost:.4f}")

        >>> # With budget limit
        >>> with track_cost(budget=1.0) as tracker:
        ...     result = expensive_program(data)  # Raises BudgetExceededError if cost > $1.0

        >>> # With per-model budget
        >>> with track_cost(budget_per_model={"openai/gpt-4": 0.5}) as tracker:
        ...     result = program(data)
    """
    tracker = CostTracker(budget=budget, budget_per_model=budget_per_model)

    with settings.context(cost_tracker=tracker):
        yield tracker
