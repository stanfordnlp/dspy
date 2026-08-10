"""Compile the vibe pipeline with a DSPy optimizer.

    python dspy_vibe/examples/optimize.py --model openai/gpt-4o-mini

What it does: scores the deterministic baseline, compiles `VibeToSpec` against
`spec_quality` with MIPROv2, scores the result on a held-out split, and saves
the compiled program. Nothing is adopted automatically — a run that does not
beat the baseline should not be shipped.

Run it without `--model` to see the baseline numbers alone; that path needs no
API key.
"""

from __future__ import annotations

import argparse
import statistics

import dspy
from dspy_vibe import spec_from_instruction
from dspy_vibe.metrics import spec_quality
from dspy_vibe.modules import VibeToSpec

TRAIN = [
    ("csinálj egy login formot, legyen szép és gyors, ne kelljen hozzá backend", ""),
    ("add a dark mode toggle to the settings page", "Next.js app router, Tailwind"),
    ("kellene egy CSV export a riport oldalra, nagy fájlokkal is bírja", "Django, Postgres"),
    ("make the search bar fuzzy, but don't slow down typing", "React, 50k items client-side"),
    ("rakj rá cache-t az API-ra, ne törjön el a friss adat", "FastAPI, Redis available"),
    ("clean up the error handling in the upload flow, it's a mess", "Node, Express"),
]

DEV = [
    ("csinálj egy sötét témát a dashboardhoz, ne nyúlj a loginhoz", "Vue 3"),
    ("add pagination to the users table, keep it fast on 100k rows", "Rails, Postgres"),
    ("legyen retry a webhook küldésre, de ne duplikáljon", "Python, Celery"),
]


def _examples(rows: list[tuple[str, str]]) -> list[dspy.Example]:
    return [
        dspy.Example(instruction=instruction, repo_context=context).with_inputs("instruction", "repo_context")
        for instruction, context in rows
    ]


def _score(program, examples: list[dspy.Example]) -> float:
    scores = []
    for example in examples:
        prediction = program(instruction=example.instruction, repo_context=example.repo_context)
        scores.append(spec_quality(example, prediction))
    return statistics.mean(scores)


def _baseline_score(examples: list[dspy.Example]) -> float:
    scores = []
    for example in examples:
        spec = spec_from_instruction(example.instruction, example.repo_context)
        scores.append(spec_quality(example, dspy.Prediction(spec=spec)))
    return statistics.mean(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="LM id; omit to only report the deterministic baseline")
    parser.add_argument("--save", default="", help="path to save the compiled program")
    args = parser.parse_args()

    train, dev = _examples(TRAIN), _examples(DEV)

    baseline_dev = _baseline_score(dev)
    print(f"deterministic baseline (dev): {baseline_dev:.3f}")
    if not args.model:
        print("no --model given; stopping before optimization")
        return 0

    dspy.configure(lm=dspy.LM(args.model))
    program = VibeToSpec()

    uncompiled_dev = _score(program, dev)
    print(f"uncompiled LM program  (dev): {uncompiled_dev:.3f}")

    optimizer = dspy.MIPROv2(metric=spec_quality, auto="light")
    compiled = optimizer.compile(program, trainset=train, requires_permission_to_run=False)

    compiled_dev = _score(compiled, dev)
    print(f"compiled LM program    (dev): {compiled_dev:.3f}")

    verdict = "improved" if compiled_dev > max(uncompiled_dev, baseline_dev) else "no improvement"
    print(f"\nverdict: {verdict}")
    if verdict == "no improvement":
        print("Do not adopt this candidate. A compiled program that loses to string handling is not worth its cost.")

    if args.save:
        compiled.save(args.save)
        print(f"saved compiled program to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
