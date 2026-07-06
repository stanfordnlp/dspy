"""IFEval scoring — no DSPy dependency."""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from ifeval_eval import instructions_registry
from ifeval_eval.evaluation_lib import InputExample, test_instruction_following_strict

if TYPE_CHECKING:
    from .data import IFEvalExample
    from .lm import LMClient

logger = logging.getLogger(__name__)


def _filter_kwargs(instruction_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    allowed = set(cls(instruction_id).get_instruction_args_keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _to_input_example(ex: "IFEvalExample") -> InputExample:
    filtered = [
        _filter_kwargs(iid, kw)
        for iid, kw in zip(ex.instruction_id_list, ex.kwargs)
    ]
    return InputExample(
        key=ex.key,
        instruction_id_list=list(ex.instruction_id_list),
        prompt=ex.prompt,
        kwargs=filtered,
    )


def score(ex: "IFEvalExample", response: str) -> float:
    """Prompt-level hard accuracy: 1.0 iff ALL constraints pass."""
    inp = _to_input_example(ex)
    out = test_instruction_following_strict(inp, {ex.prompt: response})
    if not out.follow_instruction_list:
        return 0.0
    return 1.0 if all(out.follow_instruction_list) else 0.0


def constraint_rate(ex: "IFEvalExample", response: str) -> float:
    """Per-example constraint satisfaction rate in [0, 1]."""
    inp = _to_input_example(ex)
    out = test_instruction_following_strict(inp, {ex.prompt: response})
    if not out.follow_instruction_list:
        return 0.0
    n = len(out.follow_instruction_list)
    return sum(out.follow_instruction_list) / n if n else 0.0


def score_with_detail(
    ex: "IFEvalExample", response: str
) -> tuple[float, float, list[tuple[str, bool]]]:
    """Returns (hard_score, constraint_rate, per_constraint_breakdown).

    breakdown is a list of (instruction_id, passed) pairs, one per constraint.
    Useful for building richer critique evidence.
    """
    inp = _to_input_example(ex)
    out = test_instruction_following_strict(inp, {ex.prompt: response})
    follow = out.follow_instruction_list or []
    n = len(follow)
    hard = 1.0 if (follow and all(follow)) else 0.0
    cr = sum(follow) / n if n else 0.0
    breakdown = list(zip(ex.instruction_id_list, follow))
    return hard, cr, breakdown


def evaluate_dataset(
    task_lm: "LMClient",
    system_prompt: str,
    examples: "list[IFEvalExample]",
    threads: int = 1,
    label: str = "",
) -> tuple[float, float, float]:
    """Run system_prompt on all examples.

    Returns:
        (avg_loss, prompt_level_accuracy, instruction_level_accuracy)
    """
    import threading as _threading

    from .data import IFEvalExample  # local import to avoid circular

    n = len(examples)
    done = [0]
    lock = _threading.Lock()
    tag = f" [{label}]" if label else ""

    def run_one(ex: IFEvalExample) -> tuple[float, list[bool]]:
        try:
            response = task_lm.task(system_prompt, ex.prompt)
            inp = _to_input_example(ex)
            out = test_instruction_following_strict(inp, {ex.prompt: response})
            follow_list = out.follow_instruction_list or []
            s = 1.0 if follow_list and all(follow_list) else 0.0
        except Exception as e:
            logger.warning("Eval error on key=%s: %s", ex.key, e)
            s, follow_list = 0.0, []
        with lock:
            done[0] += 1
            print(f"\r  {done[0]}/{n} examples scored{tag}...", end="", flush=True)
        return s, list(follow_list)

    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(run_one, examples))
    else:
        results = [run_one(ex) for ex in examples]

    print()  # newline after progress line
    scores = [s for s, _ in results]
    all_follow = [f for _, follow in results for f in follow]
    prompt_acc = sum(scores) / len(scores) if scores else 0.0
    inst_acc = sum(all_follow) / len(all_follow) if all_follow else 0.0
    return 1.0 - prompt_acc, prompt_acc, inst_acc
