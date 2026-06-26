"""The dspy.vibe knowledge base assembles correctly and its examples never rot.

The knowledge base (``dspy/vibe/knowledge``) is curated as concept ``.md`` files and
example ``.py`` modules, assembled into the ``KNOWLEDGE_BASE`` string that GEPA feeds the
reflection LM. These tests are LM-free: building the base is pure text processing, and each
example is validated by binding it through the live ``Vibe`` code path (no LM call).
"""

from __future__ import annotations

import pytest

import dspy
from dspy.vibe import Vibe
from dspy.vibe.knowledge import (
    FORWARD_MARKER,
    PREDICTORS_MARKER,
    build_knowledge_base,
    load_examples,
)
from dspy.vibe.primitives_doc import KNOWLEDGE_BASE

EXAMPLES = load_examples()


def test_knowledge_base_assembles_and_is_exported() -> None:
    kb = build_knowledge_base()
    # primitives_doc re-exports the same assembled string.
    assert KNOWLEDGE_BASE == kb
    # Cached: same object on repeat calls.
    assert build_knowledge_base() is kb
    assert len(kb) > 1000


def test_concepts_are_included() -> None:
    kb = build_knowledge_base()
    # A line/heading from each concept file should be present.
    for marker in (
        "What a GOOD optimized module looks like",
        "DSPy modules you can use inside",
        "Writing sub-signatures",
        "How vibe optimization works",
        "Anti-patterns to AVOID",
    ):
        assert marker in kb, f"missing concept content: {marker!r}"


def test_examples_are_rendered_without_leaking_markers() -> None:
    kb = build_knowledge_base()
    assert "## Worked examples of good dspy.vibe modules" in kb
    # Every example's task + signature should appear in the rendered base.
    for ex in EXAMPLES:
        assert ex.task in kb
        assert ex.signature in kb
    # The split markers are an authoring detail and must not leak into the prompt text.
    assert PREDICTORS_MARKER not in kb
    assert FORWARD_MARKER not in kb


def test_examples_loaded_and_well_formed() -> None:
    assert len(EXAMPLES) >= 6
    names = {ex.name for ex in EXAMPLES}
    # The canonical lessons we curate.
    assert {"invoice_total", "slugify", "long_report_qa"} <= names
    for ex in EXAMPLES:
        assert ex.task and ex.signature and ex.notes, f"{ex.name} is missing metadata"
        assert "PREDICTORS" in ex.predictors_src, f"{ex.name} predictors_src lacks PREDICTORS"
        assert "def forward" in ex.forward_src, f"{ex.name} forward_src lacks def forward"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.name)
def test_example_binds_through_vibe(example) -> None:
    """Each example must bind via the real Vibe code path — this guards against rot.

    Construction binds the (LM-free) RLM baseline; ``_bind_code`` then execs the example's
    own ``predictors_src`` / ``forward_src``, constructing its predictors (no LM call) and
    attaching ``forward``. A drifted example (renamed API, bad signature, missing field)
    fails here instead of silently misleading the reflection LM.
    """
    program = Vibe(example.signature)  # persist_to=None: in-memory, LM-free baseline
    program._bind_code(example.predictors_src, example.forward_src)

    assert program.predictors_src == example.predictors_src
    assert program.forward_src == example.forward_src
    # Every predictor in PREDICTORS was attached as a discoverable attribute on the module.
    for name in program._predictor_names:
        assert hasattr(program, name), f"{example.name}: predictor {name!r} not attached"


def test_deterministic_example_runs_with_no_lm() -> None:
    slugify = next(ex for ex in EXAMPLES if ex.name == "slugify")
    program = Vibe(slugify.signature)
    program._bind_code(slugify.predictors_src, slugify.forward_src)
    # A fully-deterministic module attaches no predictors and runs without any LM.
    assert program._predictor_names == []
    result = program(title="Hello, World! Two")
    assert isinstance(result, dspy.Prediction)
    assert result.slug == "hello-world-two"
