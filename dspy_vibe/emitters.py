"""Write artifacts to disk in the format the target host reads.

Two emitters ship: `markdown` produces YAML-frontmatter Markdown, which agent
hosts load directly; `json` produces plain data files for programmatic use.
Both write the same validated model, so switching format cannot change meaning.

The YAML writer here is intentionally small and handles only the shapes the
artifact models can produce (strings and string lists). That avoids a runtime
dependency on PyYAML for a job this narrow, and it fails loudly rather than
silently mangling an unexpected type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel

from dspy_vibe.types import AgentArtifact, SkillArtifact, VibeBundle, VibeSpec

Format = Literal["markdown", "json"]

_LEADING_SPECIALS = set(":#{}[],&*?|-<>=!%@`\"'")
_YAML_KEYWORDS = {"true", "false", "null", "yes", "no", "on", "off", "~"}
# ": " turns a scalar into a mapping and " #" starts a comment, wherever they sit.
_INLINE_BREAKERS = (": ", " #")


def _yaml_scalar(value: str) -> str:
    text = str(value)
    if text == "":
        return '""'
    if "\n" in text:
        # Fold a multi-line scalar into one line: frontmatter values are labels,
        # not prose, and a folded block would complicate every reader.
        text = " ".join(text.split())
    needs_quotes = (
        text[0] in _LEADING_SPECIALS
        or text.strip() != text
        or text.lower() in _YAML_KEYWORDS
        or text.endswith(":")
        or any(breaker in text for breaker in _INLINE_BREAKERS)
    )
    if needs_quotes:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _yaml_block(data: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, str):
            if not value:
                continue
            lines.append(f"{key}: {_yaml_scalar(value)}")
        elif isinstance(value, (list, tuple)):
            items = [item for item in value if str(item).strip()]
            if not items:
                continue
            lines.append(f"{key}:")
            lines.extend(f"  - {_yaml_scalar(str(item))}" for item in items)
        else:
            raise TypeError(f"frontmatter value for {key!r} must be a string or list, got {type(value).__name__}")
    return "\n".join(lines)


def _section(title: str, items: Iterable[str], *, numbered: bool = False) -> str:
    items = [item for item in items if str(item).strip()]
    if not items:
        return ""
    if numbered:
        body = "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))
    else:
        body = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n\n{body}\n"


def spec_to_markdown(spec: VibeSpec) -> str:
    frontmatter = _yaml_block(
        {
            "name": spec.slug,
            "title": spec.title,
            "goal": spec.goal,
            "confidence": spec.confidence,
        }
    )
    parts = [f"---\n{frontmatter}\n---\n", f"# {spec.title}\n", f"**Goal:** {spec.goal}\n"]
    if spec.context:
        parts.append(f"## Context\n\n{spec.context}\n")
    parts.append(_section("Scope", spec.scope))
    parts.append(_section("Non-goals", spec.non_goals))
    parts.append(_section("Constraints", spec.constraints))
    if spec.acceptance:
        rows = "\n".join(
            f"| {criterion.statement} | {criterion.verification} |" for criterion in spec.acceptance
        )
        parts.append("## Acceptance criteria\n\n| Criterion | Verification |\n| --- | --- |\n" + rows + "\n")
    parts.append(_section("Risks", spec.risks))
    if spec.open_questions:
        rows = "\n".join(
            f"| {question.question} | {question.why_it_matters} | "
            f"{question.assumption_used or '**blocking — needs an answer**'} |"
            for question in spec.open_questions
        )
        parts.append(
            "## Open questions\n\n| Question | Why it matters | Assumption used |\n| --- | --- | --- |\n" + rows + "\n"
        )
    if spec.source_instruction:
        parts.append(f"## Original instruction\n\n> {spec.source_instruction}\n")
    return "\n".join(part for part in parts if part).rstrip() + "\n"


def agent_to_markdown(agent: AgentArtifact) -> str:
    frontmatter = _yaml_block(
        {
            "name": agent.name,
            "description": agent.description,
            "tools": agent.tools,
            "model": agent.model,
            "source_spec": agent.source_spec,
        }
    )
    parts = [f"---\n{frontmatter}\n---\n", f"# {agent.name}\n", f"{agent.role}\n"]
    parts.append(_section("Instructions", agent.instructions, numbered=True))
    parts.append(_section("Guardrails", agent.guardrails))
    parts.append(_section("Success criteria", agent.success_criteria))
    return "\n".join(part for part in parts if part).rstrip() + "\n"


def skill_to_markdown(skill: SkillArtifact) -> str:
    frontmatter = _yaml_block(
        {
            "name": skill.name,
            "description": skill.description,
            "triggers": skill.triggers,
            "source_spec": skill.source_spec,
        }
    )
    parts = [f"---\n{frontmatter}\n---\n", f"# {skill.name}\n", f"{skill.description}\n"]
    parts.append(_section("Triggers", skill.triggers))
    parts.append(_section("Inputs", skill.inputs))
    parts.append(_section("Procedure", skill.procedure, numbered=True))
    parts.append(_section("Outputs", skill.outputs))
    parts.append(_section("Checks", skill.checks))
    parts.append(_section("Limits", skill.limits))
    return "\n".join(part for part in parts if part).rstrip() + "\n"


def _render(artifact: BaseModel, fmt: Format) -> str:
    if fmt == "json":
        return json.dumps(artifact.model_dump(), indent=2, ensure_ascii=False) + "\n"
    if isinstance(artifact, VibeSpec):
        return spec_to_markdown(artifact)
    if isinstance(artifact, AgentArtifact):
        return agent_to_markdown(artifact)
    if isinstance(artifact, SkillArtifact):
        return skill_to_markdown(artifact)
    raise TypeError(f"no markdown emitter for {type(artifact).__name__}")


def render_bundle(bundle: VibeBundle, fmt: Format = "markdown") -> dict[str, str]:
    """Render every artifact to text without touching the filesystem.

    Callers that embed the pipeline — an editor extension, a web service — need
    the rendered text and the structured data from a *single* run. Re-running
    the pipeline to get the other half would cost a second LM call and could
    return different content.
    """
    return {
        "spec": _render(bundle.spec, fmt),
        "agent": _render(bundle.agent, fmt),
        "skill": _render(bundle.skill, fmt),
    }


def write_bundle(
    bundle: VibeBundle,
    out_dir: str | Path,
    *,
    fmt: Format = "markdown",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write spec, `.agent`, and `.skill` files. Returns the paths written.

    Existing files are never overwritten unless asked: a generated artifact the
    user then edited by hand is worth more than a regeneration.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = bundle.spec.slug
    targets = {
        "spec": out / f"{slug}.spec.md" if fmt == "markdown" else out / f"{slug}.spec.json",
        "agent": out / f"{slug}.agent",
        "skill": out / f"{slug}.skill",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing) + " (pass overwrite=True)")

    rendered = render_bundle(bundle, fmt)
    for key, path in targets.items():
        path.write_text(rendered[key], encoding="utf-8", newline="\n")
    return targets
