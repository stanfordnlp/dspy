"""
IFEval dataset adapter.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import dspy
from datasets import load_dataset

from core.ifeval_metrics import ifeval_constraint_metric, ifeval_metric, ifeval_metric_gepa
from data_adapters.base import DatasetAdapter

if TYPE_CHECKING:
    from dspy import Example


def _stratify_key(example: dict[str, Any]) -> str:
    instruction_ids = example.get("instruction_id_list") or []
    if not instruction_ids:
        return "unknown"
    return instruction_ids[0].split(":")[0]


def _split_examples(
    rows: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified split by primary instruction family."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_stratify_key(row), []).append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    rng = random.Random(seed)
    for bucket_rows in buckets.values():
        shuffled = list(bucket_rows)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else max(0, n - 2)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else (1 if n >= 2 else 0)
        if n_train + n_val >= n:
            n_train = max(0, n - 2)
            n_val = 1 if n - n_train >= 2 else 0
        n_test = max(0, n - n_train - n_val)

        train_rows.extend(shuffled[:n_train])
        val_rows.extend(shuffled[n_train : n_train + n_val])
        test_rows.extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def _to_dspy_example(row: dict[str, Any]) -> Example:
    return dspy.Example(
        key=row["key"],
        prompt=row["prompt"],
        instruction_id_list=row["instruction_id_list"],
        kwargs=row["kwargs"],
    ).with_inputs("prompt")


class IFEvalAdapter(DatasetAdapter):
    """Adapter for the IFEval instruction-following benchmark."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def load_dataset(self) -> tuple[list[Example], list[Example], list[Example]]:
        dataset = load_dataset("google/IFEval", split="train")
        rows = list(dataset)

        train_frac = self.config.get("train_frac", 0.5)
        val_frac = self.config.get("val_frac", 0.25)
        split_seed = self.config.get("split_seed", self.config.get("train_seed", 1))

        train_rows, val_rows, test_rows = _split_examples(rows, train_frac, val_frac, split_seed)

        train_size = self.config.get("train_size")
        dev_size = self.config.get("dev_size")
        test_size = self.config.get("test_size")

        if train_size is not None:
            train_rows = train_rows[:train_size]
        if dev_size is not None:
            val_rows = val_rows[:dev_size]
        if test_size is not None:
            test_rows = test_rows[:test_size]

        return (
            [_to_dspy_example(row) for row in train_rows],
            [_to_dspy_example(row) for row in val_rows],
            [_to_dspy_example(row) for row in test_rows],
        )

    def get_metric(self) -> Any:
        metric_name = self.config.get("metric", "hard")
        if metric_name == "constraint":
            return ifeval_constraint_metric
        return ifeval_metric

    def get_gepa_metric(self) -> Any:
        return ifeval_metric_gepa

    @property
    def name(self) -> str:
        return "ifeval"

    @property
    def uses_context(self) -> bool:
        return False
