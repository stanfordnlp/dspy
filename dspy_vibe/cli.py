"""Command line entry point.

    python -m dspy_vibe convert "csinálj egy login formot, legyen szép és gyors" --out ./out
    python -m dspy_vibe convert --file request.txt --format json --model openai/gpt-4o-mini
    python -m dspy_vibe check ./out/login-form.agent

Without `--model` the pipeline runs its deterministic converter, so the command
works with no API key and no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from dspy_vibe import offline
from dspy_vibe.emitters import write_bundle
from dspy_vibe.metrics import agent_validity, bundle_quality, skill_validity
from dspy_vibe.types import AgentArtifact, SkillArtifact, VibeBundle


def _read_instruction(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.instruction:
        return args.instruction
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("no instruction: pass text, --file PATH, or pipe it on stdin")


def _build_bundle(args: argparse.Namespace, instruction: str) -> VibeBundle:
    tools = [tool.strip() for tool in (args.tools or "").split(",") if tool.strip()]
    if not args.model:
        return offline.bundle_from_instruction(instruction, args.context, tools)

    import dspy  # imported lazily: the offline path must not require a configured LM
    from dspy_vibe.modules import VibeCoder

    dspy.configure(lm=dspy.LM(args.model))
    return VibeCoder(available_tools=tools).forward(
        instruction=instruction, repo_context=args.context
    ).bundle


def cmd_convert(args: argparse.Namespace) -> int:
    instruction = _read_instruction(args)
    bundle = _build_bundle(args, instruction)

    if args.stdout:
        print(json.dumps(bundle.model_dump(), indent=2, ensure_ascii=False))
        return 0

    try:
        written = write_bundle(bundle, args.out, fmt=args.format, overwrite=args.overwrite)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1

    for kind, path in written.items():
        print(f"{kind:<6} {path}")
    score = bundle_quality(
        SimpleNamespace(instruction=instruction, available_tools=None),
        SimpleNamespace(bundle=bundle, spec=bundle.spec),
    )
    print(f"\nbundle quality: {score:.2f}  (confidence: {bundle.spec.confidence})")
    blocking = bundle.spec.blocking_questions
    if blocking:
        print(f"\n{len(blocking)} blocking question(s) — answer these before implementing:")
        for question in blocking:
            print(f"  - {question.question}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    failures = 0
    for target in args.paths:
        path = Path(target)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"FAIL {path}: not a readable JSON artifact ({error})")
            print("     (markdown artifacts are validated at generation time; check the .json format)")
            failures += 1
            continue
        try:
            if path.suffix == ".agent":
                artifact = AgentArtifact.model_validate(payload)
                score = agent_validity(artifact)
            elif path.suffix == ".skill":
                artifact = SkillArtifact.model_validate(payload)
                score = skill_validity(artifact)
            else:
                print(f"SKIP {path}: unknown artifact type")
                continue
        except Exception as error:  # pydantic ValidationError and friends
            print(f"FAIL {path}: {error}")
            failures += 1
            continue
        status = "PASS" if score == 1.0 else "WARN"
        print(f"{status} {path}: validity {score:.2f}")
        failures += int(score < 0.5)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dspy_vibe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="turn a vibe instruction into a spec, .agent, and .skill")
    convert.add_argument("instruction", nargs="?", help="the informal request; omit to use --file or stdin")
    convert.add_argument("--file", help="read the instruction from a file")
    convert.add_argument("--context", default="", help="known repository facts: stack, conventions")
    convert.add_argument("--tools", default="", help="comma-separated tool names the host offers")
    convert.add_argument("--out", default="./vibe-out", help="output directory")
    convert.add_argument("--format", choices=("markdown", "json"), default="markdown")
    convert.add_argument("--model", default="", help="LM id, e.g. openai/gpt-4o-mini; omit for offline mode")
    convert.add_argument("--overwrite", action="store_true", help="replace existing artifacts")
    convert.add_argument("--stdout", action="store_true", help="print the bundle as JSON instead of writing files")
    convert.set_defaults(func=cmd_convert)

    check = sub.add_parser("check", help="validate generated .agent/.skill JSON artifacts")
    check.add_argument("paths", nargs="+")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
