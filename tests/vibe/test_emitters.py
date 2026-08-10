import json

import pytest

from dspy_vibe import bundle_from_instruction, write_bundle
from dspy_vibe.emitters import _yaml_block, agent_to_markdown, skill_to_markdown, spec_to_markdown
from dspy_vibe.types import AgentArtifact

INSTRUCTION = "add a dark mode toggle to the settings page, don't touch the login screen"


def _frontmatter(text: str) -> str:
    assert text.startswith("---\n")
    return text.split("---", 2)[1]


def test_markdown_artifacts_start_with_frontmatter():
    bundle = bundle_from_instruction(INSTRUCTION)
    for rendered in (
        spec_to_markdown(bundle.spec),
        agent_to_markdown(bundle.agent),
        skill_to_markdown(bundle.skill),
    ):
        assert "name:" in _frontmatter(rendered)


def test_yaml_quotes_values_that_would_break_parsing():
    block = _yaml_block({"a": "plain", "b": "has: colon", "c": ["- dash", "ok"], "d": "true"})
    assert "a: plain" in block
    assert 'b: "has: colon"' in block
    assert '  - "- dash"' in block
    assert 'd: "true"' in block


def test_yaml_skips_empty_values_and_folds_newlines():
    block = _yaml_block({"empty": "", "none": [], "multi": "one\ntwo"})
    assert "empty" not in block
    assert "none" not in block
    assert block == "multi: one two"


def test_yaml_rejects_unsupported_types():
    with pytest.raises(TypeError):
        _yaml_block({"count": 3})


def test_write_bundle_creates_agent_and_skill_files(tmp_path):
    bundle = bundle_from_instruction(INSTRUCTION)
    written = write_bundle(bundle, tmp_path)
    assert written["agent"].suffix == ".agent"
    assert written["skill"].suffix == ".skill"
    assert written["agent"].read_text(encoding="utf-8").startswith("---")


def test_json_format_round_trips_through_the_model(tmp_path):
    bundle = bundle_from_instruction(INSTRUCTION)
    written = write_bundle(bundle, tmp_path, fmt="json")
    payload = json.loads(written["agent"].read_text(encoding="utf-8"))
    assert AgentArtifact.model_validate(payload) == bundle.agent


def test_existing_files_are_not_clobbered(tmp_path):
    bundle = bundle_from_instruction(INSTRUCTION)
    write_bundle(bundle, tmp_path)
    with pytest.raises(FileExistsError):
        write_bundle(bundle, tmp_path)
    written = write_bundle(bundle, tmp_path, overwrite=True)
    assert written["spec"].exists()


def test_blocking_question_is_marked_in_the_spec_document():
    bundle = bundle_from_instruction(INSTRUCTION)
    bundle.spec.open_questions[0].assumption_used = ""
    assert "blocking" in spec_to_markdown(bundle.spec)
