"""Curated knowledge base for ``dspy.vibe`` code synthesis.

The assembled string (:func:`build_knowledge_base`, exported as
``KNOWLEDGE_BASE``) is fed to the codegen / reflection LM as ``extra_guidance``
when ``dspy.GEPA`` optimizes a ``dspy.Vibe`` module's code (see
``dspy/vibe/primitives_doc.py`` and ``dspy/teleprompt/gepa``). It distills the
DSPy docs & tutorials and pairs them with validated example modules.

It is built from two folders that live next to this file:

- ``concepts/*.md`` — distilled, vibe-relevant guidance from the DSPy docs &
  tutorials (modules, signatures, optimization, patterns). Concatenated in
  filename order, so the numeric prefixes (``00_``, ``10_``, ...) set the order.
- ``examples/*.py`` — self-contained reference implementations of good
  ``dspy.Vibe`` module code. Each is a real, lint-clean Python file that is also
  validated by the test-suite (every example binds through the live ``Vibe``
  code path), so the examples cannot rot.

To extend the knowledge base, drop a new ``.md`` in ``concepts/`` or a new
``.py`` in ``examples/`` — no code change here is needed. See ``MAINTAINING.md``.

IMPORTANT: building the string is pure text processing (``ast`` + string
slicing). It MUST NOT import ``dspy`` or construct any predictor, because this
module is imported transitively during ``import dspy`` (via
``dspy/__init__.py`` -> ``dspy.vibe``), before ``dspy.Predict`` etc. exist.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).parent
_CONCEPTS_DIR = _HERE / "concepts"
_EXAMPLES_DIR = _HERE / "examples"

# Comment marker that splits an example file into its human-facing preamble (the module
# docstring, ``import dspy``, and the ``TASK`` / ``SIGNATURE`` / ``NOTES`` metadata) and the
# generated artifact below it: a single ``dspy.Module`` subclass. Everything after the marker
# is the ``module_src`` that ``Vibe._bind_code`` runs.
MODULE_MARKER = "# === MODULE ==="

_META_KEYS = ("TASK", "SIGNATURE", "NOTES")


@dataclass(frozen=True)
class KnowledgeExample:
    """One validated, self-contained ``dspy.Vibe`` reference implementation."""

    name: str  # the file stem, e.g. "invoice_total"
    task: str  # one-line description of what the module does
    signature: str  # the parent signature string, e.g. "invoice: str -> total_cents: int"
    notes: str  # why this design is good (the lesson the example teaches)
    module_src: str  # the generated artifact: one `dspy.Module` subclass


def _extract_module_src(source: str) -> str:
    """Slice the ``module_src`` (the dspy.Module subclass) out of an example file's text."""
    idx = source.find(MODULE_MARKER)
    if idx == -1:
        raise ValueError(f"example is missing the {MODULE_MARKER!r} marker")
    return source[idx + len(MODULE_MARKER) :].strip()


def _extract_meta(source: str) -> dict[str, str]:
    """Read the ``TASK`` / ``SIGNATURE`` / ``NOTES`` string literals via ``ast``.

    Uses ``ast`` rather than ``exec`` so building the knowledge base never
    imports ``dspy`` or constructs predictors (see module docstring).
    """
    meta: dict[str, str] = {}
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id in _META_KEYS
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            meta[target.id] = node.value.value.strip()
    return meta


def load_examples() -> list[KnowledgeExample]:
    """Parse every ``examples/*.py`` into a :class:`KnowledgeExample` (filename order)."""
    examples: list[KnowledgeExample] = []
    for path in sorted(_EXAMPLES_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        meta = _extract_meta(source)
        examples.append(
            KnowledgeExample(
                name=path.stem,
                task=meta.get("TASK", ""),
                signature=meta.get("SIGNATURE", ""),
                notes=meta.get("NOTES", ""),
                module_src=_extract_module_src(source),
            )
        )
    return examples


def _load_concepts() -> str:
    """Concatenate every ``concepts/*.md`` in filename order."""
    parts = [path.read_text(encoding="utf-8").strip() for path in sorted(_CONCEPTS_DIR.glob("*.md"))]
    return "\n\n".join(p for p in parts if p)


def _render_example(index: int, ex: KnowledgeExample) -> str:
    lines = [f"### Example {index}: {ex.task or ex.name}"]
    if ex.signature:
        lines.append(f"Signature: `{ex.signature}`")
    if ex.notes:
        lines.append(f"Why this is a good design: {ex.notes}")
    lines.append("\n```python\n" + ex.module_src + "\n```")
    return "\n".join(lines)


def _render_examples(examples: list[KnowledgeExample]) -> str:
    if not examples:
        return ""
    header = (
        "## Worked examples of good dspy.vibe modules\n"
        "Each is a self-contained `dspy.Module` subclass (validated by the test suite). "
        "Study the decomposition, the Python/LM split, and the output coercion."
    )
    bodies = [_render_example(i, ex) for i, ex in enumerate(examples, start=1)]
    return header + "\n\n" + "\n\n".join(bodies)


@lru_cache(maxsize=1)
def build_knowledge_base() -> str:
    """Assemble the full knowledge-base string (concepts + worked examples).

    Cached: the files are bundled package data that don't change at runtime.
    """
    sections = [s for s in (_load_concepts(), _render_examples(load_examples())) if s]
    return "\n\n".join(sections).strip()


__all__ = [
    "KnowledgeExample",
    "build_knowledge_base",
    "load_examples",
    "MODULE_MARKER",
]
