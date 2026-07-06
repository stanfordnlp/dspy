"""IFEval dataset loading — no DSPy dependency."""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class IFEvalExample:
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]


def _stratify_key(row: dict[str, Any]) -> str:
    ids = row.get("instruction_id_list") or []
    return ids[0].split(":")[0] if ids else "unknown"


def _split_rows(
    rows: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    buckets: dict[str, list] = {}
    for row in rows:
        buckets.setdefault(_stratify_key(row), []).append(row)

    train_rows, val_rows, test_rows = [], [], []
    rng = random.Random(seed)
    for bucket in buckets.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else max(0, n - 2)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else (1 if n >= 2 else 0)
        if n_train + n_val >= n:
            n_train = max(0, n - 2)
            n_val = 1 if n - n_train >= 2 else 0
        train_rows.extend(shuffled[:n_train])
        val_rows.extend(shuffled[n_train : n_train + n_val])
        test_rows.extend(shuffled[n_train + n_val :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def _to_example(row: dict[str, Any]) -> IFEvalExample:
    return IFEvalExample(
        key=row["key"],
        prompt=row["prompt"],
        instruction_id_list=list(row["instruction_id_list"]),
        kwargs=list(row["kwargs"]),
    )


def load_ifeval(
    *,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    seed: int = 1,
) -> tuple[list[IFEvalExample], list[IFEvalExample], list[IFEvalExample]]:
    """Load IFEval from HuggingFace and return (train, val, test) as plain dataclasses."""
    from datasets import load_dataset

    rows = list(load_dataset("google/IFEval", split="train"))
    train_rows, val_rows, test_rows = _split_rows(rows, train_frac, val_frac, seed)

    if train_size is not None:
        train_rows = train_rows[:train_size]
    if val_size is not None:
        val_rows = val_rows[:val_size]
    if test_size is not None:
        test_rows = test_rows[:test_size]

    return (
        [_to_example(r) for r in train_rows],
        [_to_example(r) for r in val_rows],
        [_to_example(r) for r in test_rows],
    )
