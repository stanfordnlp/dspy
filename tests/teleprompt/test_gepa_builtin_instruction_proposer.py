"""Tests for `InstructionProposer`, GEPA's built-in instruction proposer."""

import logging
from pathlib import Path
from typing import Any

import pytest

import dspy
from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.primitives.repl_types import REPLHistory
from dspy.teleprompt.gepa import InstructionProposer
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter, format_history_for_reflection
from dspy.teleprompt.gepa.instruction_proposal import (
    ProposeInstruction,
    _Skill,
    _SkillLoader,
    compact_history,
    compact_repl_history,
)
from dspy.utils.dummies import DummyLM

GEPA_PARAGRAPH_1 = (
    "Read the inputs carefully and identify the input format and infer detailed task description about the "
    "task I wish to solve with the assistant."
)
GEPA_PARAGRAPH_2 = (
    "Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific "
    "factual information about the task and include it in the instruction, as a lot of it may not be available "
    "to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, "
    "if so, include that in the instruction as well."
)

EXAMPLES = [
    {"Inputs": {"question": "2+2?"}, "Generated Outputs": {"answer": "5"}, "Feedback": "Wrong answer."},
    {"Inputs": {"question": "3+3?"}, "Generated Outputs": {"answer": "6"}, "Feedback": "Correct."},
]


def json_lm(*answers: str, **kwargs) -> DummyLM:
    """A reflection LM whose answers are proposals (and, after the first, compressions)."""
    scripted = []
    for i, text in enumerate(answers):
        scripted.append({"new_instruction" if i == 0 else "shortened_instruction": text})
    return DummyLM(scripted, adapter=dspy.JSONAdapter(), **kwargs)


def last_messages(lm: DummyLM) -> tuple[str, str]:
    messages = lm.history[-1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    return messages[0]["content"], messages[1]["content"]


def propose(proposer: InstructionProposer, lm: DummyLM, examples=EXAMPLES, components=("pred",)) -> dict[str, str]:
    with dspy.context(lm=lm):
        return proposer(
            candidate={"pred": "Answer the question."},
            reflective_dataset={"pred": examples},
            components_to_update=list(components),
        )


# --- the proposer, called directly ------------------------------------------


def test_default_prompt_has_only_the_two_base_fields_and_gepa_wording():
    lm = json_lm("Add the numbers.")
    assert propose(InstructionProposer(), lm) == {"pred": "Add the numbers."}

    system, user = last_messages(lm)
    assert GEPA_PARAGRAPH_1 in system
    assert GEPA_PARAGRAPH_2 in system
    assert "Provide the new instructions within" not in system
    assert "[[ ## current_instruction ## ]]\nAnswer the question." in user
    assert "[[ ## examples_with_feedback ## ]]\n# Example 1\n## Inputs\n### question\n2+2?" in user
    for absent in ("reference_skills", "additional_instructions", "length_limit"):
        assert absent not in user
        assert absent not in system
    assert list(InstructionProposer().propose.signature.input_fields) == [
        "current_instruction",
        "examples_with_feedback",
    ]


@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"skills": ["Be terse."]}, "reference_skills"),
        ({"additional_instructions": "Use imperative voice."}, "additional_instructions"),
        ({"max_instruction_words": 50}, "length_limit"),
        ({"max_instruction_tokens": 80}, "length_limit"),
    ],
)
def test_each_option_adds_only_its_field(options, field):
    proposer = InstructionProposer(**options)
    assert list(proposer.propose.signature.input_fields) == [field, "current_instruction", "examples_with_feedback"]

    lm = json_lm("New instruction.")
    propose(proposer, lm)
    _, user = last_messages(lm)
    assert user.index(f"[[ ## {field} ## ]]") < user.index("[[ ## current_instruction ## ]]")


def test_static_fields_precede_the_dynamic_ones():
    proposer = InstructionProposer(
        skills=["Be terse."], additional_instructions="Use imperative voice.", max_instruction_words=50
    )
    assert list(proposer.propose.signature.input_fields) == [
        "reference_skills",
        "additional_instructions",
        "length_limit",
        "current_instruction",
        "examples_with_feedback",
    ]
    lm = json_lm("New instruction.")
    propose(proposer, lm)
    _, user = last_messages(lm)
    assert "[[ ## reference_skills ## ]]\n<skill name='Be terse.'>\nBe terse.\n</skill>" in user
    assert "[[ ## additional_instructions ## ]]\nUse imperative voice." in user
    assert "[[ ## length_limit ## ]]\nThe new instruction must be at most 50 words." in user


def test_base_instructions_replace_the_docstring_and_keep_the_fields():
    proposer = InstructionProposer(base_instructions="Rewrite the instruction so it generalizes.")
    assert proposer.propose.signature.instructions == "Rewrite the instruction so it generalizes."
    assert list(proposer.propose.signature.input_fields) == list(ProposeInstruction.input_fields)
    assert list(proposer.propose.signature.output_fields) == ["new_instruction"]

    lm = json_lm("New instruction.")
    propose(proposer, lm)
    system, _ = last_messages(lm)
    assert "Rewrite the instruction so it generalizes." in system
    assert GEPA_PARAGRAPH_1 not in system


def test_default_adapter_is_json_even_when_settings_adapter_differs():
    with dspy.context(adapter=dspy.ChatAdapter()):
        lm = json_lm("New instruction.")
        proposer = InstructionProposer()
        assert isinstance(proposer.adapter, dspy.JSONAdapter)
        assert propose(proposer, lm) == {"pred": "New instruction."}
        _, user = last_messages(lm)
        assert "Respond with a JSON object" in user

        chat_lm = DummyLM([{"new_instruction": "Chat instruction."}], adapter=dspy.ChatAdapter())
        proposer = InstructionProposer(adapter=dspy.ChatAdapter())
        assert propose(proposer, chat_lm) == {"pred": "Chat instruction."}
        _, user = last_messages(chat_lm)
        assert "Respond with a JSON object" not in user
        assert "[[ ## new_instruction ## ]]" in user


def test_constructor_validation():
    with pytest.raises(ValueError, match="max_instruction_words"):
        InstructionProposer(max_instruction_words=0)
    with pytest.raises(ValueError, match="max_instruction_tokens"):
        InstructionProposer(max_instruction_tokens=-1)
    with pytest.raises(TypeError, match="compaction"):
        InstructionProposer(compaction="yes")
    with pytest.raises(TypeError, match="adapter"):
        InstructionProposer(adapter=object())
    assert InstructionProposer(additional_instructions="  ").additional_instructions is None
    assert InstructionProposer(base_instructions="").base_instructions is None


def test_component_missing_from_the_reflective_dataset_is_left_out():
    lm = json_lm("New instruction.")
    with dspy.context(lm=lm):
        out = InstructionProposer()(
            candidate={"pred": "old", "other": "old too"},
            reflective_dataset={"pred": EXAMPLES},
            components_to_update=["pred", "other", "unknown"],
        )
    assert out == {"pred": "New instruction."}
    assert len(lm.history) == 1


def test_empty_instruction_raises():
    lm = json_lm("   ")
    with pytest.raises(ValueError, match="empty instruction"):
        propose(InstructionProposer(), lm)


def test_special_characters_round_trip_exactly():
    text = (
        'Report R&D spend. If score < 5, write "low" and name the <committee name>.\n\n'
        "```python\nprint('ok')\n```\n\nIt's done."
    )
    lm = json_lm(text)
    assert propose(InstructionProposer(), lm) == {"pred": text}


# --- skills -----------------------------------------------------------------


def test_skills_load_from_paths_and_inline_text_and_render_in_order(tmp_path: Path):
    skill_dir = tmp_path / "prompt-engineering"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: prompt-engineering\ndescription: Clear instructions\n---\nBe explicit.", encoding="utf-8"
    )
    proposer = InstructionProposer(skills=[skill_dir, "Be terse.", "# Style\n\nUse active voice."])
    assert [skill.name for skill in proposer.skills] == ["prompt-engineering", "Be terse.", "Style"]

    lm = json_lm("New instruction.")
    propose(proposer, lm)
    _, user = last_messages(lm)
    assert (
        "[[ ## reference_skills ## ]]\n"
        "<skill name='prompt-engineering' description='Clear instructions'>\nBe explicit.\n</skill>\n\n"
        "<skill name='Be terse.'>\nBe terse.\n</skill>\n\n"
        "<skill name='Style'>\n# Style\n\nUse active voice.\n</skill>\n\n"
    ) in user


def test_skill_directory_loads_skill_md_only(tmp_path: Path):
    skill_dir = tmp_path / "prompt-engineering"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Prompt engineering\n\nBe explicit.", encoding="utf-8")
    (skill_dir / "notes.md").write_text("ignored", encoding="utf-8")

    skill = _SkillLoader.load(str(skill_dir))
    assert skill == _Skill(name="prompt-engineering", content="# Prompt engineering\n\nBe explicit.")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"SKILL\.md"):
        _SkillLoader.load(empty_dir)


def test_skill_file_loads_with_frontmatter(tmp_path: Path):
    path = tmp_path / "openai.md"
    path.write_text(
        "---\nname: openai-style\ndescription: 'Prompting tips for OpenAI models'\n---\n\nUse markdown headers.\n",
        encoding="utf-8",
    )
    skill = _SkillLoader.load(path)
    assert skill == _Skill(
        name="openai-style", content="Use markdown headers.", description="Prompting tips for OpenAI models"
    )
    assert _SkillLoader.load(str(path)) == skill

    plain = tmp_path / "plain.txt"
    plain.write_text("Keep it short.", encoding="utf-8")
    assert _SkillLoader.load(plain) == _Skill(name="plain", content="Keep it short.")


def test_missing_skill_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="prompt-enginering"):
        _SkillLoader.load("./skills/prompt-enginering")
    with pytest.raises(FileNotFoundError):
        _SkillLoader.load("notes.markdown")
    with pytest.raises(FileNotFoundError):
        _SkillLoader.load(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        InstructionProposer(skills=[str(tmp_path / "missing")])


def test_inline_skill_and_empty_skill():
    assert _SkillLoader.load("Be terse.") == _Skill(name="Be terse.", content="Be terse.")
    assert _SkillLoader.load("x" * 80 + "\nmore").name == "x" * 60 + "…"
    for empty in ("", "   \n"):
        with pytest.raises(ValueError, match="Empty skill"):
            _SkillLoader.load(empty)
    with pytest.raises(TypeError):
        _SkillLoader.load(42)


# --- length caps --------------------------------------------------------------


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_no_cap_means_no_length_field_and_no_compress_call():
    draft = words(2000)
    lm = json_lm(draft)
    proposer = InstructionProposer()
    assert proposer.compress is None
    assert propose(proposer, lm) == {"pred": draft}
    assert len(lm.history) == 1


def test_draft_within_the_cap_makes_one_call():
    lm = json_lm(words(10))
    assert propose(InstructionProposer(max_instruction_words=50), lm) == {"pred": words(10)}
    assert len(lm.history) == 1


def test_over_length_draft_is_compressed_within_the_cap(caplog):
    lm = json_lm(words(10), words(3))
    with caplog.at_level(logging.WARNING):
        assert propose(InstructionProposer(max_instruction_words=5), lm) == {"pred": words(3)}
    assert len(lm.history) == 2
    _, user = last_messages(lm)
    assert "[[ ## instruction ## ]]\n" + words(10) in user
    assert "[[ ## length_limit ## ]]\nThe new instruction must be at most 5 words." in user
    assert not caplog.records


def test_compressed_draft_still_over_but_shorter_is_kept_with_a_warning(caplog):
    lm = json_lm(words(10), words(7))
    with caplog.at_level(logging.WARNING):
        assert propose(InstructionProposer(max_instruction_words=5), lm) == {"pred": words(7)}
    [record] = caplog.records
    assert "'pred'" in record.message
    assert "7 words" in record.message
    assert "limit: 5 words" in record.message


@pytest.mark.parametrize("compressed", [words(12), ""])
def test_compression_that_does_not_shorten_keeps_the_draft_with_a_warning(compressed, caplog):
    lm = json_lm(words(10), compressed)
    with caplog.at_level(logging.WARNING):
        assert propose(InstructionProposer(max_instruction_words=5), lm) == {"pred": words(10)}
    assert len(lm.history) == 2
    [record] = caplog.records
    assert "'pred'" in record.message
    assert "10 words" in record.message
    assert "limit: 5 words" in record.message


def test_multi_line_drafts_keep_their_newlines_in_every_path():
    within = "Line one.\n\nLine two.\n- a\n- b"
    assert propose(InstructionProposer(max_instruction_words=50), json_lm(within)) == {"pred": within}

    over = "\n".join(words(4) for _ in range(5))
    compressed = "Keep this.\nAnd this."
    assert propose(InstructionProposer(max_instruction_words=5), json_lm(over, compressed)) == {"pred": compressed}

    still_over = "Keep this.\nAnd this.\nAnd that too."
    assert propose(InstructionProposer(max_instruction_words=5), json_lm(over, still_over)) == {"pred": still_over}

    longer = over + "\nand more"
    assert propose(InstructionProposer(max_instruction_words=5), json_lm(over, longer)) == {"pred": over}


def test_token_cap_with_the_fallback_counter(monkeypatch):
    import litellm

    def broken_counter(**kwargs):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(litellm, "token_counter", broken_counter)

    draft = "x" * 40  # 10 tokens with the ~4 characters per token fallback
    lm = json_lm(draft, "y" * 12)
    proposer = InstructionProposer(max_instruction_tokens=5)
    assert propose(proposer, lm) == {"pred": "y" * 12}
    assert len(lm.history) == 2
    _, user = last_messages(lm)
    assert "The new instruction must be at most 5 tokens." in user

    both = InstructionProposer(max_instruction_words=3, max_instruction_tokens=5)
    assert both._length_limit_text() == "The new instruction must be at most 3 words and at most 5 tokens."


# --- compaction -----------------------------------------------------------------


def tool_history(result: str) -> dspy.History:
    tool_calls = ToolCalls.from_dict_list([{"name": "search", "args": {"query": "cats"}}])
    results = ToolCallResults.from_tool_calls_and_values(tool_calls, [result])
    tool_calls = tool_calls.model_copy(update={"tool_call_results": results})
    return dspy.History(messages=[{"question": "cats?", "next_thought": "Search first.", "tool_calls": tool_calls}])


def test_compact_history_cuts_long_tool_results_and_leaves_the_input_alone():
    long_result = "r" * 1_200
    history = tool_history(long_result)

    compacted = compact_history(history)

    [message] = compacted.messages
    assert message["question"] == "cats?"
    assert message["next_thought"] == "Search first."
    call = message["tool_calls"].tool_calls[0]
    assert (call.name, call.args) == ("search", {"query": "cats"})
    [result] = message["tool_calls"].tool_call_results.tool_call_results
    assert result.value == "r" * 500 + " [700 of 1,200 characters cut]"
    assert result.name == "search"
    assert result.is_error is False

    # The input is untouched: ReActV2 appends to its History in place and the trace shares that object.
    assert history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].value == long_result
    assert compacted is not history
    assert compacted.messages is not history.messages

    short = compact_history(tool_history("short"))
    assert short.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].value == "short"


def test_compact_history_treats_field_markers_in_tool_results_as_plain_text():
    history = tool_history("[[ ## thought_1 ## ]]\n" + "z" * 1_000)
    [result] = compact_history(history).messages[0]["tool_calls"].tool_call_results.tool_call_results
    assert result.value.startswith("[[ ## thought_1 ## ]]\n" + "z" * 478)
    assert result.value.endswith(" [522 of 1,022 characters cut]")


def test_compact_history_returns_a_plain_chat_history_unchanged():
    history = dspy.History(messages=[{"question": "a" * 2_000, "answer": "b" * 2_000}, {"question": "c"}])
    assert compact_history(history) == history


def test_compact_repl_history_lowers_max_output_chars_and_never_raises_it():
    history = REPLHistory().append(reasoning="look", code="print(x)", output="o" * 3_000)
    compacted = compact_repl_history(history)
    assert compacted.max_output_chars == 500
    assert compacted.entries == history.entries
    assert history.max_output_chars == 10_000
    assert "(2,500 characters omitted)" in compacted.format()

    small = REPLHistory(max_output_chars=100)
    assert compact_repl_history(small).max_output_chars == 100


def test_without_compaction_history_and_repl_history_render_as_on_main():
    history = tool_history("r" * 1_200)
    repl = REPLHistory().append(reasoning="look", code="print(x)", output="o" * 3_000)
    examples = [{"Inputs": {"Context": history, "repl": repl}, "Generated Outputs": {"a": "b"}, "Feedback": "ok"}]

    lm = json_lm("New instruction.")
    propose(InstructionProposer(), lm, examples=examples)
    _, user = last_messages(lm)
    assert "### Context\n" + format_history_for_reflection(history) + "\n\n### repl\n" + str(repl) + "\n\n" in user
    assert "characters cut" not in user
    assert "characters omitted" not in user

    lm = json_lm("New instruction.")
    propose(InstructionProposer(compaction=True), lm, examples=examples)
    _, user = last_messages(lm)
    assert "r" * 500 + " [700 of 1,200 characters cut]" in user
    assert "r" * 501 not in user
    assert "(2,500 characters omitted)" in user


# --- GEPA end to end -------------------------------------------------------------


def no_reflection_errors(caplog) -> bool:
    return "Exception during reflection/proposal" not in caplog.text


def test_gepa_default_proposer_on_a_text_program(caplog):
    student = dspy.Predict("input -> output")
    trainset = [
        dspy.Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        dspy.Example(input="What does the fox say?", output="ring").with_inputs("input"),
    ]
    task_lm = DummyLM([{"output": "unsure"}] * 30)
    reflection_lm = DummyLM([{"new_instruction": "Answer in one word."}] * 5, adapter=dspy.JSONAdapter())

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=float(gold.output == pred.output), feedback="Wrong answer.")

    with dspy.context(lm=task_lm), caplog.at_level(logging.INFO, logger="dspy"):
        optimized = dspy.GEPA(metric=metric, reflection_lm=reflection_lm, max_metric_calls=5).compile(
            student, trainset=trainset, valset=trainset
        )

    assert no_reflection_errors(caplog)
    assert reflection_lm.history, "The reflection LM should have been called"
    system, user = last_messages(reflection_lm)
    assert GEPA_PARAGRAPH_1 in system
    assert "[[ ## current_instruction ## ]]\n" + student.signature.instructions in user
    assert "[[ ## examples_with_feedback ## ]]\n# Example 1\n## Inputs\n### input\n" in user
    assert "## Feedback\nWrong answer." in user
    assert "Proposed new text for self: Answer in one word." in caplog.text
    assert optimized is not None


LONG_TOOL_OUTPUT = "cat facts " * 300  # 3,000 characters


def react_program_and_lm():
    def lookup(query: str) -> str:
        return LONG_TOOL_OUTPUT

    task_lm = DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "cats"}}]),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "cats purr"}}]),
            },
        ]
        * 20
    )
    return dspy.ReActV2("question -> answer", tools=[lookup]), task_lm


def react_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=0.3, feedback="Answer with more detail.")


def test_gepa_compaction_on_a_react_program_and_custom_proposers_see_the_context_string(caplog):
    trainset = [dspy.Example(question="cats?", answer="cats purr loudly").with_inputs("question")]

    # The built-in proposer with compaction: the reflection prompt omits the long tool output.
    program, task_lm = react_program_and_lm()
    reflection_lm = DummyLM([{"new_instruction": "Search, then answer in detail."}] * 5, adapter=dspy.JSONAdapter())
    with dspy.context(lm=task_lm, adapter=dspy.ChatAdapter()), caplog.at_level(logging.INFO, logger="dspy"):
        dspy.GEPA(
            metric=react_metric,
            reflection_lm=reflection_lm,
            instruction_proposer=InstructionProposer(compaction=True),
            max_metric_calls=4,
        ).compile(program, trainset=trainset, valset=trainset)

    assert no_reflection_errors(caplog)
    _, user = last_messages(reflection_lm)
    assert "### Context\n```json\n  0: {" in user
    assert "'next_thought': 'I should look this up.'" in user
    assert "characters cut]" in user
    assert LONG_TOOL_OUTPUT not in user

    # A custom proposer receives the Context string exactly as the bridge built it on main.
    seen: list[dict[str, Any]] = []

    def custom_proposer(candidate, reflective_dataset, components_to_update):
        seen.extend(reflective_dataset[name][0] for name in components_to_update)
        return dict.fromkeys(components_to_update, "Custom instruction.")

    program, task_lm = react_program_and_lm()
    with dspy.context(lm=task_lm, adapter=dspy.ChatAdapter()):
        dspy.GEPA(metric=react_metric, instruction_proposer=custom_proposer, max_metric_calls=4).compile(
            program, trainset=trainset, valset=trainset
        )

    [example] = seen
    context = example["Inputs"]["Context"]
    assert isinstance(context, str)
    assert context.startswith("```json\n  0: {")
    assert context.endswith("```")
    assert LONG_TOOL_OUTPUT in context
    assert "characters cut" not in context
    assert list(example["Inputs"]) == ["Context", "tools"]
    assert isinstance(example["Inputs"]["tools"], str)


def test_explicit_builtin_proposer_with_a_flex_submodule_logs_no_custom_proposer_warning(caplog):
    class Echo(dspy.Signature):
        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.flex = dspy.Flex(Echo)
            self.sibling = dspy.Predict("x -> y")

        def forward(self, **kwargs):
            return self.flex(**kwargs)

    prog = Prog()
    reflection_lm = DummyLM(
        {
            "current_source": {"revised_source": prog.flex.module_src},
            "current_instruction": {"new_instruction": "new instruction"},
        },
        adapter=dspy.JSONAdapter(),
    )
    adapter = DspyAdapter(
        student_module=prog,
        metric_fn=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.0,
        feedback_map={},
        reflection_lm=reflection_lm,
        custom_instruction_proposer=InstructionProposer(additional_instructions="Be brief."),
    )
    assert adapter._builtin_instruction_proposer
    candidate = {"flex": prog.flex.module_src, "sibling": "old instruction"}
    reflective = {"flex": [], "sibling": EXAMPLES}

    with caplog.at_level(logging.WARNING), dspy.context(adapter=dspy.JSONAdapter()):
        out = adapter.propose_new_texts(candidate, reflective, ["flex", "sibling"])

    assert out["sibling"] == "new instruction"
    assert not any("custom instruction_proposer" in r.message for r in caplog.records)
