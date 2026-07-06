#!/usr/bin/env python3
"""SBO / SBO-Lite optimization for IFEval — no DSPy, no adapter overhead.

Designed for small models (e.g. qwen3.5:0.8b) where ChatAdapter format
requirements hurt instruction-following scores.

SBO-Lite (default, --lite): replaces Monte Carlo judge calls with a single
qualitative verifier call per candidate. ~5-10x cheaper per iteration.
Good starting point for 0.8B experiments.

SBO (full, --no-lite): uses bundle model M_k(p) with semantic judge + lambda
adaptation. More rigorous but significantly more optimizer LM calls.

Examples:
    # Smoke test with SBO-Lite (default)
    python sbo_ifeval/run.py --preset smoke --no-thinking

    # Fast iteration: 0.8b task, 4b optimizer, SBO-Lite
    python sbo_ifeval/run.py --preset fast \\
        --task-model ollama_chat/qwen3.5:0.8b \\
        --optimizer-model ollama_chat/qwen3:4b-instruct \\
        --no-thinking

    # Full SBO (more expensive)
    python sbo_ifeval/run.py --preset fast --no-lite \\
        --task-model ollama_chat/qwen3.5:0.8b \\
        --optimizer-model ollama_chat/qwen3:4b-instruct \\
        --no-thinking --iterations 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from sbo_ifeval.data import load_ifeval
from sbo_ifeval.lm import LMClient
from sbo_ifeval.metrics import constraint_rate, evaluate_dataset, score
from sbo_ifeval.optimizer import SBOConfig, SBOOptimizer, SBOLiteConfig, SBOLiteOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress LiteLLM's verbose INFO spam — only show warnings and above
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# NOTE: trainset is accepted by optimizer.optimize() to match the original DSPy SBO
# interface, but is never used — all optimization (loss estimation, critique sampling)
# runs on valset. val_size is therefore the primary budget knob.
PRESETS: dict[str, dict[str, int | None]] = {
    "smoke":    {"train_size": 4,   "val_size": 8,   "test_size": 4},
    "fast":     {"train_size": 16,  "val_size": 50,  "test_size": 15},
    "standard": {"train_size": 100, "val_size": 100, "test_size": 50},
    "full":     {"train_size": None, "val_size": None, "test_size": None},
}

DEFAULT_SYSTEM_PROMPT = (
    "Follow every instruction in the user prompt exactly. "
    "Satisfy all explicit constraints such as length, format, keywords, and structure. "
    "Return only the requested content with no preamble or explanation unless asked."
)


def _print_eval_results(
    tag: str,
    avg_loss: float,
    prompt_acc: float,
    inst_acc: float,
    n: int,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"IFEval results — {tag}")
    print("=" * 60)
    print(f"Prompt-level accuracy:      {100 * prompt_acc:.1f}%  ({int(round(prompt_acc * n))}/{n})")
    print(f"Instruction-level accuracy: {100 * inst_acc:.1f}%")
    print(f"Average loss:               {avg_loss:.4f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SBO / SBO-Lite optimization for IFEval without DSPy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    parser.add_argument(
        "--task-model",
        default="ollama_chat/qwen3.5:0.8b",
        help="LiteLLM model id for the task (IFEval answering).",
    )
    parser.add_argument(
        "--optimizer-model",
        default="ollama_chat/qwen3:4b-instruct",
        help="LiteLLM model id for judge/proposer/critic.",
    )
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Set reasoning_effort=none on task and optimizer models (for qwen3.5 etc).",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Max SBO iterations.")
    parser.add_argument("--candidates", type=int, default=3, help="Candidates per iteration.")
    parser.add_argument("--threads", type=int, default=1, help="Evaluation threads.")
    parser.add_argument("--output-json", type=str, help="Save results to JSON.")
    parser.add_argument("--print-io", action="store_true", help="Print every LM input and output.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    lite_group = parser.add_mutually_exclusive_group()
    lite_group.add_argument(
        "--lite",
        dest="lite",
        action="store_true",
        default=True,
        help="Use SBO-Lite (default): qualitative verifier instead of semantic judge. Cheaper.",
    )
    lite_group.add_argument(
        "--no-lite",
        dest="lite",
        action="store_false",
        help="Use full SBO with bundle model and lambda adaptation.",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    mode = "SBO-Lite" if args.lite else "SBO (full)"
    print(f"Mode:            {mode}")
    print(f"Task model:      {args.task_model}")
    print(f"Optimizer model: {args.optimizer_model}")
    print(f"Preset:          {args.preset}")
    print(f"Iterations:      {args.iterations}")
    print(f"Candidates:      {args.candidates}")

    # Load data
    sizes = PRESETS[args.preset]
    train_set, val_set, test_set = load_ifeval(**sizes)
    # train_set is loaded but unused (see NOTE above); val_set drives all optimization.
    print(f"\nDataset: val={len(val_set)} (optimization), test={len(test_set)} (final eval)")

    # Set up LM clients
    extra_kwargs: dict[str, Any] = {}
    if args.no_thinking:
        extra_kwargs["reasoning_effort"] = "none"

    task_lm = LMClient(
        model=args.task_model,
        api_base=args.api_base,
        temperature=0.0,
        max_tokens=2048,
        print_io=args.print_io,
        label="task",
        **extra_kwargs,
    )
    optimizer_lm = LMClient(
        model=args.optimizer_model,
        api_base=args.api_base,
        temperature=0.7,
        max_tokens=4096,
        print_io=args.print_io,
        label="optimizer",
        **extra_kwargs,
    )

    # Baseline evaluation
    print(f"\nRunning baseline evaluation on test set ({len(test_set)} examples)...")
    baseline_loss, baseline_prompt_acc, baseline_inst_acc = evaluate_dataset(
        task_lm, DEFAULT_SYSTEM_PROMPT, test_set, threads=args.threads, label="baseline"
    )
    _print_eval_results("baseline (test)", baseline_loss, baseline_prompt_acc, baseline_inst_acc, len(test_set))

    # Build optimizer
    if args.lite:
        config = SBOLiteConfig(
            num_candidates=args.candidates,
            max_iterations=args.iterations,
            max_critique_examples=5,
        )
        optimizer = SBOLiteOptimizer(task_lm=task_lm, optimizer_lm=optimizer_lm, metric=score, config=config)
    else:
        config = SBOConfig(
            num_candidates=args.candidates,
            max_iterations=args.iterations,
            num_judge_samples=2,
            max_critique_examples=5,
        )
        optimizer = SBOOptimizer(task_lm=task_lm, optimizer_lm=optimizer_lm, metric=score, config=config)

    print(f"\nStarting {mode} optimization ({args.iterations} max iterations)...")
    result = optimizer.optimize(
        initial_prompt=DEFAULT_SYSTEM_PROMPT,
        trainset=train_set,
        valset=val_set,
    )

    print(f"\nOptimized prompt:\n{'─' * 60}\n{result.best_prompt}\n{'─' * 60}")
    print(f"Best val loss: {result.best_loss:.4f} (iteration {result.best_idx})")
    print(f"Serious steps: {result.num_serious_steps}, Null steps: {result.num_null_steps}")

    # Final evaluation on test set
    print(f"\nRunning final evaluation on test set ({len(test_set)} examples)...")
    final_loss, final_prompt_acc, final_inst_acc = evaluate_dataset(
        task_lm, result.best_prompt, test_set, threads=args.threads, label="optimized"
    )
    _print_eval_results("optimized (test)", final_loss, final_prompt_acc, final_inst_acc, len(test_set))

    delta = final_prompt_acc - baseline_prompt_acc
    print(f"\nDelta (optimized − baseline): {delta:+.1%}")

    if args.output_json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "task_model": args.task_model,
            "optimizer_model": args.optimizer_model,
            "preset": args.preset,
            "baseline": {
                "prompt_accuracy": baseline_prompt_acc,
                "instruction_accuracy": baseline_inst_acc,
                "loss": baseline_loss,
            },
            "optimized": {
                "prompt_accuracy": final_prompt_acc,
                "instruction_accuracy": final_inst_acc,
                "loss": final_loss,
                "prompt": result.best_prompt,
                "iterations": result.total_iterations,
                "serious_steps": result.num_serious_steps,
                "null_steps": result.num_null_steps,
            },
            "delta_prompt_accuracy": delta,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
