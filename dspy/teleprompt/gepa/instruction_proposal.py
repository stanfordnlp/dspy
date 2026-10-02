import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa.core.adapter import ProposalFn

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import History
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.primitives.repl_types import REPLHistory
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample, format_history_for_reflection
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


# --------------------------------------------------------------------------- #
# Skills: reference material for the reflection LM
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Skill:
    name: str
    content: str
    description: str | None = None


class _SkillLoader:
    """Resolve a skill source (a directory, a file, or inline text) into a `_Skill`.

    Resolution order for `load`:

    1. A `Path` is always a path (`~` is expanded). A missing path raises `FileNotFoundError`.
    2. A `str` naming an existing file or directory (after `~` expansion) is a path.
    3. A `str` that does not exist but looks like a path raises `FileNotFoundError`. Text spanning
       more than one line never looks like a path. A single line looks like a path when it contains
       a path separator and no whitespace, starts with `.` or `~` and contains no whitespace, starts
       like a path (`./`, `../`, `~/`, `/`, or a drive letter), ends with `.md`, `.markdown`, or
       `.txt`, or names an entry inside a directory that exists (so `./skills/my skill` and
       `skills/my skill` are caught even though they contain a space).
    4. Any other `str` is inline skill content. Empty content raises `ValueError`.

    A directory must contain `SKILL.md` (the Agent Skills layout); only that file is read. A file is
    read as UTF-8. A leading YAML frontmatter block supplies `name` and `description` and is
    stripped from the content. The fallback name is the directory name, the file stem, or the first
    line of an inline skill.

    Note: a single line of inline content can be mistaken for a path when it matches one of the rules
    above; write it on more than one line, or pass a `Path` for strict path checking.
    """

    SKILL_FILE = "SKILL.md"
    FILE_SUFFIXES = (".md", ".markdown", ".txt")
    # `./`, `../`, `~/`, `/`, `\`, or a Windows drive letter, with either separator.
    _PATH_PREFIX_RE = re.compile(r"^(?:\.{1,2}[\\/]|~[\\/]|[\\/]|[A-Za-z]:[\\/])")

    @classmethod
    def load(cls, source: "str | Path") -> _Skill:
        if isinstance(source, Path):
            return cls._from_path(source.expanduser())
        if not isinstance(source, str):
            raise TypeError(f"A skill must be a str or a Path, not {type(source).__name__}.")

        if not source.strip():
            raise ValueError("Empty skill content.")
        path = cls._existing_path(source)
        if path is not None:
            return cls._from_path(path)
        if cls._looks_like_path(source):
            raise FileNotFoundError(
                f"Skill {source!r} looks like a path, but no such file or directory exists. "
                "Pass the path to an existing skill file or directory. Inline skill content that "
                "resembles a path must span more than one line."
            )
        return cls._from_text(source, fallback_name=None)

    @classmethod
    def _from_path(cls, path: Path) -> _Skill:
        if not path.exists():
            raise FileNotFoundError(f"Skill path {path} does not exist.")
        if path.is_dir():
            skill_md = path / cls.SKILL_FILE
            if not skill_md.exists():
                raise FileNotFoundError(f"Skill directory {path} has no {cls.SKILL_FILE}.")
            return cls._from_text(skill_md.read_text(encoding="utf-8"), fallback_name=path.name)
        return cls._from_text(path.read_text(encoding="utf-8"), fallback_name=path.stem)

    @classmethod
    def _from_text(cls, text: str, fallback_name: str | None) -> _Skill:
        text = text.strip()
        if not text:
            raise ValueError("Empty skill content.")

        meta, content = cls._parse_frontmatter(text)
        content = content.strip()

        if fallback_name is None:
            # The first non-empty line doubles as a display name for inline skills.
            first_line = content.splitlines()[0].lstrip("# ").strip() if content else ""
            fallback_name = ((first_line[:60] + "…") if len(first_line) > 60 else first_line) or "inline-skill"

        return _Skill(name=meta.get("name") or fallback_name, content=content, description=meta.get("description"))

    @staticmethod
    def _existing_path(source: str) -> Path | None:
        try:
            path = Path(source).expanduser()
            return path if path.exists() else None
        except (OSError, ValueError):  # e.g. an inline string too long to be a valid path
            return None

    @classmethod
    def _looks_like_path(cls, source: str) -> bool:
        """Whether a string that names nothing on disk was meant as a path rather than inline content."""
        if "\n" in source or "\r" in source:
            return False
        source = source.strip()
        has_separator = "/" in source or "\\" in source
        if not re.search(r"\s", source):
            return has_separator or source.startswith((".", "~")) or source.lower().endswith(cls.FILE_SUFFIXES)
        # Paths may contain spaces ("./skills/my skill"), so a single line with whitespace is still a path
        # when it carries a stronger signal than a bare separator.
        if cls._PATH_PREFIX_RE.match(source) or source.lower().endswith(cls.FILE_SUFFIXES):
            return True
        return has_separator and cls._parent_exists(source)

    @staticmethod
    def _parent_exists(source: str) -> bool:
        try:
            return Path(source).expanduser().parent.exists()
        except (OSError, ValueError):
            return False

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """Split a leading `--- ... ---` block into (metadata, body).

        Only flat, unindented `key: value` lines are read; nested YAML is ignored. Returns `({}, text)`
        when there is no well-formed block.
        """
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text
        for end, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                meta: dict[str, str] = {}
                for raw in lines[1:end]:
                    if raw.startswith((" ", "\t")) or ":" not in raw:
                        continue
                    key, _, value = raw.partition(":")
                    meta[key.strip()] = value.strip().strip("'\"")
                return meta, "\n".join(lines[end + 1 :])
        return {}, text


def _render_skill(skill: _Skill) -> str:
    """Render a skill as one `<skill>` block for the reflection prompt."""
    attrs = f"name={skill.name!r}"
    if skill.description:
        attrs += f" description={skill.description!r}"
    return f"<skill {attrs}>\n{skill.content.strip()}\n</skill>"


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #


class ProposeInstruction(dspy.Signature):
    """I provided an assistant with instructions to perform a task for me. You are given those instructions, along with examples of different task inputs provided to the assistant, the assistant's response for each of them, and some feedback on how the assistant's response could be better.

    Your task is to write a new instruction for the assistant.

    Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

    Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well."""

    current_instruction: str = dspy.InputField(desc="The instructions I provided to the assistant.")
    examples_with_feedback: str = dspy.InputField(
        desc="Task inputs provided to the assistant, the assistant's response for each, "
        "and feedback on how the response could be better."
    )
    new_instruction: str = dspy.OutputField(desc="The new instruction for the assistant.")


class CompressInstruction(dspy.Signature):
    """Shorten the instruction to fit within the stated limit. Preserve every
    strategy, rule, and constraint; cut redundancy and verbosity. Do not add
    new content. Output only the shortened instruction."""

    instruction: str = dspy.InputField()
    length_limit: str = dspy.InputField()
    shortened_instruction: str = dspy.OutputField()


# --------------------------------------------------------------------------- #
# Compaction of long inputs
# --------------------------------------------------------------------------- #

_COMPACTED_CHARS = 500


def compact_history(history: History, max_chars: int = _COMPACTED_CHARS) -> History:
    """Return a copy of `history` with every tool result longer than `max_chars` cut to that length.

    Only the `value` of each `ToolCallResult` attached to a `ToolCalls` message value is shortened; a cut
    value ends with a marker naming how many characters were removed. Thoughts, tool names, tool
    arguments, user inputs, and error flags are untouched. The input is never mutated: `dspy.ReActV2`
    appends to its `History` in place and the trace shares that object.
    """
    messages = []
    for message in history.messages:
        new_message = dict(message)
        for key, value in message.items():
            if isinstance(value, ToolCalls) and isinstance(value.tool_call_results, ToolCallResults):
                new_message[key] = _compact_tool_calls(value, max_chars)
        messages.append(new_message)
    return history.model_copy(update={"messages": messages})


def _compact_tool_calls(tool_calls: ToolCalls, max_chars: int) -> ToolCalls:
    results = tool_calls.tool_call_results
    new_results = []
    for result in results.tool_call_results:
        text = result.value if isinstance(result.value, str) else str(result.value)
        if len(text) > max_chars:
            cut = len(text) - max_chars
            marker = f" [{cut:,} of {len(text):,} characters cut]"
            result = result.model_copy(update={"value": text[:max_chars] + marker})
        new_results.append(result)
    new_results = results.model_copy(update={"tool_call_results": new_results})
    return tool_calls.model_copy(update={"tool_call_results": new_results})


def compact_repl_history(history: REPLHistory, max_chars: int = _COMPACTED_CHARS) -> REPLHistory:
    """Return a copy of `history` whose rendered REPL outputs are capped at `max_chars` characters.

    `REPLHistory.format()` already cuts each output to its head and tail and reports the omitted count, so
    only `max_output_chars` changes, and only downwards.
    """
    return history.model_copy(update={"max_output_chars": min(history.max_output_chars, max_chars)})


# --------------------------------------------------------------------------- #
# The proposer
# --------------------------------------------------------------------------- #


@experimental(version="3.4.1")
class InstructionProposer(ProposalFn):
    """GEPA's default instruction proposer.

    `dspy.GEPA` builds `InstructionProposer()` when no `instruction_proposer` is passed. The proposer
    renders each component's reflective examples as markdown and asks the reflection LM, through
    `dspy.Predict` and a `JSONAdapter`, for a new instruction. Inputs that are `dspy.Type` instances
    (for example `dspy.Image`) reach the reflection LM as structured content.

    Args:
        skills: Reference material for the reflection LM, each a path to a skill file, a path to a
            directory holding `SKILL.md`, or an inline string. Loaded once, at construction; a path
            that does not exist raises.
        additional_instructions: Guidance applied to every proposal, for example "Write instructions in
            imperative voice."
        base_instructions: Replaces the proposal prompt (the `ProposeInstruction` docstring). The input and
            output fields are unchanged.
        max_instruction_words: Cap on the length of each proposed instruction, in words.
        max_instruction_tokens: Cap on the length of each proposed instruction, in tokens, counted with
            `litellm.token_counter` for the reflection LM's model.
        compaction: When True, long tool results inside `dspy.History` inputs and long outputs inside
            `REPLHistory` inputs are cut before the examples are rendered. Nothing else is shortened.
        adapter: The adapter for the proposer's own LM calls. Defaults to `JSONAdapter()`.

    When a cap is set and a proposal exceeds it, the proposer makes one call asking the reflection LM to
    shorten the draft. A draft that is still over the cap is kept and a warning is logged; the proposer
    never truncates an instruction. Exceptions from LM calls propagate to GEPA, which retries the
    proposal once and then skips the iteration.

    Example:
        ```python
        from dspy.teleprompt.gepa import InstructionProposer

        dspy.GEPA(
            metric=metric,
            reflection_lm=lm,
            instruction_proposer=InstructionProposer(
                skills=["./skills/prompt-engineering"],
                additional_instructions="Write instructions in imperative voice.",
                max_instruction_words=300,
                compaction=True,
            ),
            auto="medium",
        )
        ```
    """

    def __init__(
        self,
        skills: Sequence[str | Path] | None = None,
        additional_instructions: str | None = None,
        base_instructions: str | None = None,
        max_instruction_words: int | None = None,
        max_instruction_tokens: int | None = None,
        compaction: bool = False,
        adapter: Adapter | None = None,
    ):
        for label, cap in (
            ("max_instruction_words", max_instruction_words),
            ("max_instruction_tokens", max_instruction_tokens),
        ):
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
                raise ValueError(f"{label} must be a positive int or None, got {cap!r}.")
        if not isinstance(compaction, bool):
            raise TypeError(f"compaction must be a bool, got {type(compaction).__name__}.")
        if adapter is not None and not isinstance(adapter, Adapter):
            raise TypeError(f"adapter must be a dspy.Adapter or None, got {type(adapter).__name__}.")

        self.skills: tuple[_Skill, ...] = tuple(_SkillLoader.load(skill) for skill in (skills or ()))
        self.additional_instructions = _clean_text("additional_instructions", additional_instructions)
        self.base_instructions = _clean_text("base_instructions", base_instructions)
        self.max_instruction_words = max_instruction_words
        self.max_instruction_tokens = max_instruction_tokens
        self.compaction = compaction
        self.adapter = adapter if adapter is not None else JSONAdapter()

        # Static fields come first so provider-side prompt caching can reuse them across calls. Each
        # prepend lands at index 0, so they are added in reverse order of their final position.
        signature = ProposeInstruction
        if self._length_limit_text():
            signature = signature.prepend(
                "length_limit", dspy.InputField(desc="Hard limit on the length of the new instruction."), str
            )
        if self.additional_instructions:
            signature = signature.prepend(
                "additional_instructions",
                dspy.InputField(desc="Additional requirements from the user for the new instruction. Follow them."),
                str,
            )
        if self.skills:
            signature = signature.prepend(
                "reference_skills",
                dspy.InputField(
                    desc="Trusted reference material chosen by the user. Draw on it when writing the new instruction."
                ),
                str,
            )
        if self.base_instructions:
            signature = signature.with_instructions(self.base_instructions)

        self.propose = dspy.Predict(signature)
        self.compress = dspy.Predict(CompressInstruction) if self._length_limit_text() else None

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose a new instruction for each component in `components_to_update`.

        A component missing from `candidate` or from `reflective_dataset` is left out of the result.
        """
        results: dict[str, str] = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue

            kwargs: dict[str, Any] = {}
            if self.skills:
                kwargs["reference_skills"] = "\n\n".join(_render_skill(skill) for skill in self.skills)
            if self.additional_instructions:
                kwargs["additional_instructions"] = self.additional_instructions
            if self.compress is not None:
                kwargs["length_limit"] = self._length_limit_text()
            kwargs["current_instruction"] = candidate[name]
            kwargs["examples_with_feedback"] = self._render_examples(reflective_dataset[name])

            with dspy.context(adapter=self.adapter):
                pred = self.propose(**kwargs)

            draft = pred.new_instruction.strip()
            if not draft:
                raise ValueError(f"The reflection LM returned an empty instruction for component {name!r}.")

            results[name] = self._enforce_length(name, draft)

        return results

    # -- Rendering ----------------------------------------------------------

    def _render_examples(self, examples: Sequence[Mapping[str, Any]]) -> str:
        """Render reflective examples as markdown, in the layout of gepa's stock proposer."""
        if not examples:
            return "No examples were provided."

        def render_value(value: Any, level: int = 3) -> str:
            if isinstance(value, History):
                if self.compaction:
                    value = compact_history(value)
                return f"{format_history_for_reflection(value)}\n\n"
            if isinstance(value, REPLHistory):
                text = compact_repl_history(value).format() if self.compaction else str(value)
                return f"{text.strip()}\n\n"
            if isinstance(value, Type):
                # str() emits DSPy's custom-type marker; the adapter turns it back into structured content.
                return f"{value!s}\n\n"
            if isinstance(value, Mapping):
                out = ""
                for k, v in value.items():
                    out += f"{'#' * level} {k}\n{render_value(v, min(level + 1, 6))}"
                return out or "\n"
            if isinstance(value, (list, tuple)):
                out = ""
                for i, item in enumerate(value, 1):
                    out += f"{'#' * level} Item {i}\n{render_value(item, min(level + 1, 6))}"
                return out or "\n"
            return f"{str(value).strip()}\n\n"

        blocks = []
        for i, example in enumerate(examples, 1):
            block = f"# Example {i}\n"
            for key, value in example.items():
                block += f"## {key}\n{render_value(value)}"
            blocks.append(block)
        return "\n\n".join(blocks)

    # -- Length caps ---------------------------------------------------------

    def _length_limit_text(self) -> str:
        parts = []
        if self.max_instruction_words is not None:
            parts.append(f"at most {self.max_instruction_words} words")
        if self.max_instruction_tokens is not None:
            parts.append(f"at most {self.max_instruction_tokens} tokens")
        return "The new instruction must be " + " and ".join(parts) + "." if parts else ""

    def _measure(self, text: str) -> dict[str, int]:
        """Measure `text` in every unit that has a cap."""
        measured = {}
        if self.max_instruction_words is not None:
            measured["words"] = len(_WORD_RE.findall(text))
        if self.max_instruction_tokens is not None:
            measured["tokens"] = _count_tokens(text, getattr(dspy.settings.lm, "model", None))
        return measured

    def _caps(self) -> dict[str, int]:
        caps = {}
        if self.max_instruction_words is not None:
            caps["words"] = self.max_instruction_words
        if self.max_instruction_tokens is not None:
            caps["tokens"] = self.max_instruction_tokens
        return caps

    def _enforce_length(self, name: str, draft: str) -> str:
        """Ask the reflection LM once to shorten an over-length draft. Never truncates.

        A compressed draft that still exceeds a cap is kept when it makes progress on at least one
        over-cap unit without going over, or further over, the cap in any unit. A unit that was and
        stays within its cap may change freely. Otherwise the original draft is kept.
        """
        if self.compress is None:
            return draft
        caps = self._caps()
        draft_size = self._measure(draft)
        if all(draft_size[unit] <= caps[unit] for unit in caps):
            return draft

        with dspy.context(adapter=self.adapter):
            shortened = self.compress(instruction=draft, length_limit=self._length_limit_text()).shortened_instruction
        shortened = shortened.strip()

        if shortened:
            shortened_size = self._measure(shortened)
            if all(shortened_size[unit] <= caps[unit] for unit in caps):
                return shortened
            progressed = any(draft_size[unit] > caps[unit] and shortened_size[unit] < draft_size[unit] for unit in caps)
            regressed = any(
                shortened_size[unit] > caps[unit] and shortened_size[unit] > draft_size[unit] for unit in caps
            )
            if progressed and not regressed:
                logger.warning(
                    "The proposed instruction for component %r is %s after compression (limit: %s). "
                    "Using the compressed instruction as is.",
                    name,
                    _describe(shortened_size),
                    _describe(caps),
                )
                return shortened

        logger.warning(
            "The proposed instruction for component %r is %s (limit: %s), and compression did not shorten it. "
            "Using the original proposal as is.",
            name,
            _describe(draft_size),
            _describe(caps),
        )
        return draft


def _clean_text(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a str or None, got {type(value).__name__}.")
    return value.strip() or None


def _describe(sizes: Mapping[str, int]) -> str:
    return ", ".join(f"{count} {unit}" for unit, count in sizes.items())


def _count_tokens(text: str, model: str | None) -> int:
    try:
        import litellm

        return litellm.token_counter(model=model, text=text)
    except Exception:
        return max(1, round(len(text) / 4))  # rough fallback: ~4 characters per token


# --------------------------------------------------------------------------- #
# Multimodal proposer
# --------------------------------------------------------------------------- #


class GenerateEnhancedMultimodalInstructionFromFeedback(dspy.Signature):
    """I provided an assistant with instructions to perform a task involving visual content, but the assistant's performance needs improvement based on the examples and feedback below.

    Your task is to write a better instruction for the assistant that addresses the specific issues identified in the feedback, with particular attention to how visual and textual information should be analyzed and integrated.

    ## Analysis Steps:
    1. **Read the inputs carefully** and identify both the visual and textual input formats, understanding how they work together
    2. **Read all the assistant responses and corresponding feedback** to understand what went wrong with visual analysis, text processing, or their integration
    3. **Identify visual analysis patterns** - what visual features, relationships, or details are important for this task
    4. **Identify domain-specific knowledge** about both visual and textual aspects, as this information may not be available to the assistant in the future
    5. **Look for successful visual-textual integration strategies** and include these patterns in the instruction
    6. **Address specific visual analysis issues** mentioned in the feedback

    ## Instruction Requirements:
    - **Clear task definition** explaining how to process both visual and textual inputs
    - **Visual analysis guidance** specific to this task (what to look for, how to describe, what features matter)
    - **Integration strategies** for combining visual observations with textual information
    - **Domain-specific knowledge** about visual concepts, terminology, or relationships
    - **Error prevention guidance** for common visual analysis mistakes shown in the feedback
    - **Precise, actionable language** for both visual and textual processing

    Focus on creating an instruction that helps the assistant properly analyze visual content, integrate it with textual information, and avoid the specific visual analysis mistakes shown in the examples."""

    current_instruction = dspy.InputField(
        desc="The current instruction that was provided to the assistant to perform the multimodal task"
    )
    examples_with_feedback = dspy.InputField(
        desc="Task examples with visual content showing inputs, assistant outputs, and feedback. "
        "Pay special attention to feedback about visual analysis accuracy, visual-textual integration, "
        "and any domain-specific visual knowledge that the assistant missed."
    )

    improved_instruction = dspy.OutputField(
        desc="A better instruction for the assistant that addresses visual analysis issues, provides "
        "clear guidance on how to process and integrate visual and textual information, includes "
        "necessary visual domain knowledge, and prevents the visual analysis mistakes shown in the examples."
    )


class SingleComponentMultiModalProposer(dspy.Module):
    """
    dspy.Module for proposing improved instructions based on feedback.
    """

    def __init__(self):
        super().__init__()
        self.propose_instruction = dspy.Predict(GenerateEnhancedMultimodalInstructionFromFeedback)

    def forward(self, current_instruction: str, reflective_dataset: list[ReflectiveExample]) -> str:
        """
        Generate an improved instruction based on current instruction and feedback examples.

        Args:
            current_instruction: The current instruction that needs improvement
            reflective_dataset: List of examples with inputs, outputs, and feedback
                               May contain dspy.Image objects in inputs

        Returns:
            str: Improved instruction text
        """
        # Format examples with enhanced pattern recognition
        formatted_examples, image_map = self._format_examples_with_pattern_analysis(reflective_dataset)

        # Build kwargs for the prediction call
        predict_kwargs = {
            "current_instruction": current_instruction,
            "examples_with_feedback": formatted_examples,
        }

        # Create a rich multimodal examples_with_feedback that includes both text and images
        predict_kwargs["examples_with_feedback"] = self._create_multimodal_examples(formatted_examples, image_map)

        # Use current dspy LM settings (GEPA will pass reflection_lm via context)
        result = self.propose_instruction(**predict_kwargs)

        return result.improved_instruction

    def _format_examples_with_pattern_analysis(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:
        """
        Format examples with pattern analysis and feedback categorization.

        Returns:
            tuple: (formatted_text_with_patterns, image_map)
        """
        # First, use the existing proven formatting approach
        formatted_examples, image_map = self._format_examples_for_instruction_generation(reflective_dataset)

        # Enhanced analysis: categorize feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(reflective_dataset)

        # Add pattern analysis to the formatted examples
        if feedback_analysis["summary"]:
            pattern_summary = self._create_pattern_summary(feedback_analysis)
            enhanced_examples = f"{pattern_summary}\n\n{formatted_examples}"
            return enhanced_examples, image_map

        return formatted_examples, image_map

    def _analyze_feedback_patterns(self, reflective_dataset: list[ReflectiveExample]) -> dict[str, Any]:
        """
        Analyze feedback patterns to provide better context for instruction generation.

        Categorizes feedback into:
        - Error patterns: Common mistakes and their types
        - Success patterns: What worked well and should be preserved/emphasized
        - Domain knowledge gaps: Missing information that should be included
        - Task-specific guidance: Specific requirements or edge cases
        """
        analysis = {
            "error_patterns": [],
            "success_patterns": [],
            "domain_knowledge_gaps": [],
            "task_specific_guidance": [],
            "summary": "",
        }

        # Simple pattern recognition - could be enhanced further
        for example in reflective_dataset:
            feedback = example.get("Feedback", "").lower()

            # Identify error patterns
            if any(error_word in feedback for error_word in ["incorrect", "wrong", "error", "failed", "missing"]):
                analysis["error_patterns"].append(feedback)

            # Identify success patterns
            if any(
                success_word in feedback for success_word in ["correct", "good", "accurate", "well", "successfully"]
            ):
                analysis["success_patterns"].append(feedback)

            # Identify domain knowledge needs
            if any(
                knowledge_word in feedback
                for knowledge_word in ["should know", "domain", "specific", "context", "background"]
            ):
                analysis["domain_knowledge_gaps"].append(feedback)

        # Create summary if patterns were found
        if any(analysis[key] for key in ["error_patterns", "success_patterns", "domain_knowledge_gaps"]):
            analysis["summary"] = (
                f"Patterns identified: {len(analysis['error_patterns'])} error(s), {len(analysis['success_patterns'])} success(es), {len(analysis['domain_knowledge_gaps'])} knowledge gap(s)"
            )

        return analysis

    def _create_pattern_summary(self, feedback_analysis: dict[str, Any]) -> str:
        """Create a summary of feedback patterns to help guide instruction generation."""

        summary_parts = ["## Feedback Pattern Analysis\n"]

        if feedback_analysis["error_patterns"]:
            summary_parts.append(f"**Common Issues Found ({len(feedback_analysis['error_patterns'])} examples):**")
            summary_parts.append("Focus on preventing these types of mistakes in the new instruction.\n")

        if feedback_analysis["success_patterns"]:
            summary_parts.append(
                f"**Successful Approaches Found ({len(feedback_analysis['success_patterns'])} examples):**"
            )
            summary_parts.append("Build on these successful strategies in the new instruction.\n")

        if feedback_analysis["domain_knowledge_gaps"]:
            summary_parts.append(
                f"**Domain Knowledge Needs Identified ({len(feedback_analysis['domain_knowledge_gaps'])} examples):**"
            )
            summary_parts.append("Include this specialized knowledge in the new instruction.\n")

        return "\n".join(summary_parts)

    def _format_examples_for_instruction_generation(
        self, reflective_dataset: list[ReflectiveExample]
    ) -> tuple[str, dict[int, list[Type]]]:
        """
        Format examples using GEPA's markdown structure while preserving image objects.

        Returns:
            tuple: (formatted_text, image_map) where image_map maps example_index -> list[images]
        """

        def render_value_with_images(value, level=3, example_images=None):
            if example_images is None:
                example_images = []

            if isinstance(value, Type):
                image_idx = len(example_images) + 1
                example_images.append(value)
                return f"[IMAGE-{image_idx} - see visual content]\n\n"
            elif isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value_with_images(v, min(level + 1, 6), example_images)
                if not value:
                    s += "\n"
                return s
            elif isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value_with_images(item, min(level + 1, 6), example_images)
                if not value:
                    s += "\n"
                return s
            else:
                return f"{str(value).strip()}\n\n"

        def convert_sample_to_markdown_with_images(sample, example_num):
            example_images = []
            s = f"# Example {example_num}\n"

            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value_with_images(val, level=3, example_images=example_images)

            return s, example_images

        formatted_parts = []
        image_map = {}

        for i, example_data in enumerate(reflective_dataset):
            formatted_example, example_images = convert_sample_to_markdown_with_images(example_data, i + 1)
            formatted_parts.append(formatted_example)

            if example_images:
                image_map[i] = example_images

        formatted_text = "\n\n".join(formatted_parts)

        if image_map:
            total_images = sum(len(imgs) for imgs in image_map.values())
            formatted_text = (
                f"The examples below include visual content ({total_images} images total). "
                "Please analyze both the text and visual elements when suggesting improvements.\n\n" + formatted_text
            )

        return formatted_text, image_map

    def _create_multimodal_examples(self, formatted_text: str, image_map: dict[int, list[Type]]) -> Any:
        """
        Create a multimodal input that contains both text and images for the reflection LM.

        Args:
            formatted_text: The formatted text with image placeholders
            image_map: Dictionary mapping example_index -> list[images] for structured access
        """
        if not image_map:
            return formatted_text

        # Collect all images from all examples
        all_images = []
        for example_images in image_map.values():
            all_images.extend(example_images)

        multimodal_content = [formatted_text]
        multimodal_content.extend(all_images)
        return multimodal_content


# TODO: InstructionProposer now delivers images to the reflection LM as structured content.
# Review this class for deprecation once the two are compared on a vision task.
class MultiModalInstructionProposer(ProposalFn):
    """GEPA-compatible multimodal instruction proposer.

    This class handles multimodal inputs (like dspy.Image) during GEPA optimization by using
    a single-component proposer for each component that needs to be updated.
    """

    def __init__(self):
        self.single_proposer = SingleComponentMultiModalProposer()

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[ReflectiveExample]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """GEPA-compatible proposal function.

        Args:
            candidate: Current component name -> instruction mapping
            reflective_dataset: Component name -> list of reflective examples
            components_to_update: List of component names to update

        Returns:
            dict: Component name -> new instruction mapping
        """
        updated_components = {}

        for component_name in components_to_update:
            if component_name in candidate and component_name in reflective_dataset:
                current_instruction = candidate[component_name]
                component_reflective_data = reflective_dataset[component_name]

                # Call the single-instruction proposer.
                #
                # In the future, proposals could consider multiple components instructions,
                # instead of just the current instruction, for more holistic instruction proposals.
                new_instruction = self.single_proposer(
                    current_instruction=current_instruction, reflective_dataset=component_reflective_data
                )

                updated_components[component_name] = new_instruction

        return updated_components
