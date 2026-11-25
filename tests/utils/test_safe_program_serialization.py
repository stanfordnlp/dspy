"""Tests for safe, pickle-free declarative program serialization."""

from __future__ import annotations

import datetime
import importlib
import logging
from typing import Annotated, Literal
from unittest.mock import patch

import cloudpickle
import orjson
import pydantic
import pytest

import dspy
from dspy.primitives.code_interpreter import FinalOutput
from dspy.utils import DummyLM

_PYTHON_CODE = dspy.Code["python"]


class _MockInterpreter:
    """Minimal in-memory CodeInterpreter for testing RLM execution without Deno."""

    def __init__(self):
        self.tools = {}
        self.output_fields = []
        self.started = False
        self.shutdown_called = False

    def start(self) -> None:
        self.started = True

    def execute(self, code: str, variables: dict | None = None):
        if "SUBMIT(" in code:
            return FinalOutput({"answer": "42"})
        if "__dspy_guest_spec" in code or "sandbox_guest" in code:
            return ""
        return "computed=42"

    def shutdown(self) -> None:
        self.shutdown_called = True


class _TopLevelTwoStepQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.draft = dspy.Predict("question -> draft")
        self.final = dspy.ChainOfThought("question, draft -> answer")

    def forward(self, question: str):
        draft_pred = self.draft(question=question)
        return self.final(question=question, draft=draft_pred.draft)


class _TopLevelConfigurableQA(dspy.Module):
    def __init__(self, steps: int = 2, tag: str = "default"):
        super().__init__()
        self.steps = steps
        self.tag = tag
        self.predictors_list = [dspy.Predict("question -> answer") for _ in range(steps)]

    def to_program_config(self) -> dict:
        return {"steps": self.steps, "tag": self.tag}

    @classmethod
    def from_program_config(cls, config: dict) -> _TopLevelConfigurableQA:
        return cls(steps=config["steps"], tag=config["tag"])

    def forward(self, question: str):
        result = None
        for pred in self.predictors_list:
            result = pred(question=question)
        return result


def _forbid_cloudpickle(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("cloudpickle must never be invoked during JSON program save/load")

    monkeypatch.setattr(cloudpickle, "dump", _fail)
    monkeypatch.setattr(cloudpickle, "dumps", _fail)
    monkeypatch.setattr(cloudpickle, "load", _fail)
    monkeypatch.setattr(cloudpickle, "loads", _fail)


def test_predict_json_roundtrip_typed_fields_demos_lm_and_execution(tmp_path, monkeypatch):
    _forbid_cloudpickle(monkeypatch)

    class RichSignature(dspy.Signature):
        """Classify and score a multi-modal request."""

        question: str = dspy.InputField(desc="User question")
        tags: list[str] = dspy.InputField(desc="Search tags")
        scores: dict[str, int] = dspy.InputField(desc="Prior scores")
        as_of: datetime.date = dspy.InputField(desc="Reference date")
        image: dspy.Image | None = dspy.InputField(default=None, desc="Optional image")
        audio: dspy.Audio | None = dspy.InputField(default=None)
        attachment: dspy.File | None = dspy.InputField(default=None)
        history: dspy.History | None = dspy.InputField(default=None)
        label: Literal["low", "high"] = dspy.OutputField(desc="Priority label")
        confidence: int = dspy.OutputField(ge=0, le=100, desc="Confidence percentage")
        items: tuple[str, ...] = dspy.OutputField(desc="Extracted items")
        snippet: _PYTHON_CODE | None = dspy.OutputField(default=None)
        reasoning: dspy.Reasoning | None = dspy.OutputField(default=None)
        tool_calls: dspy.ToolCalls | None = dspy.OutputField(default=None)

    pred = dspy.Predict(RichSignature, temperature=0.25, max_tokens=256)
    pred.signature = pred.signature.with_instructions("Custom optimized instructions.")
    pred.demos = [
        dspy.Example(
            question="Urgent outage",
            tags=["prod"],
            scores={"sev": 1},
            as_of="2026-01-01",
            label="high",
            confidence=95,
            items=["db"],
        ).with_inputs("question", "tags", "scores", "as_of")
    ]
    pred.lm = dspy.LM("openai/gpt-4o-mini", temperature=0.2, max_tokens=128)
    pred._compiled = True

    save_dir = tmp_path / "predict_prog"
    pred.save(save_dir, save_program=True, format="json")

    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "program.json").exists()
    assert not (save_dir / "program.pkl").exists()

    metadata = orjson.loads((save_dir / "metadata.json").read_bytes())
    assert metadata["format"] == "json"
    assert metadata["schema_version"] == 1
    assert "dependency_versions" in metadata

    loaded = dspy.load(save_dir)
    assert isinstance(loaded, dspy.Predict)
    assert loaded.signature.equals(pred.signature)
    assert loaded.signature.instructions == "Custom optimized instructions."
    assert loaded.signature.fields["tags"].annotation == list[str]
    assert loaded.signature.fields["scores"].annotation == dict[str, int]
    assert loaded.signature.fields["as_of"].annotation is datetime.date
    assert loaded.signature.fields["label"].annotation == Literal["low", "high"]
    assert loaded.signature.fields["items"].annotation == tuple[str, ...]
    assert loaded.signature.fields["image"].annotation == (dspy.Image | None)
    assert loaded.signature.fields["image"].default is None
    assert loaded.signature.fields["confidence"].metadata == pred.signature.fields["confidence"].metadata
    assert loaded.config == {"temperature": 0.25, "max_tokens": 256}
    assert loaded.demos == pred.dump_state(json_mode=True)["demos"]
    assert loaded.lm is not None
    assert loaded.lm.model == "openai/gpt-4o-mini"
    assert loaded.lm.kwargs["temperature"] == 0.2
    assert loaded._compiled is True

    # Verify execution with DummyLM on a simple typed Predict.
    qa = dspy.Predict("question: str -> answer: int", "Return an integer answer.")
    qa.demos = [dspy.Example(question="1+1", answer=2).with_inputs("question")]
    qa_dir = tmp_path / "qa_prog"
    qa.save(qa_dir, save_program=True, format="json")
    loaded_qa = dspy.load(qa_dir)

    with dspy.context(lm=DummyLM([{"answer": "42"}])):
        out = loaded_qa(question="6*7")
    assert out.answer == 42


def test_chain_of_thought_json_roundtrip_and_execution(tmp_path, monkeypatch):
    _forbid_cloudpickle(monkeypatch)

    cot = dspy.ChainOfThought("question: str -> answer: str", temperature=0.4)
    cot.predict.signature = cot.predict.signature.with_instructions("Think carefully before answering.")
    cot.predict.demos = [
        dspy.Example(question="Capital of Spain?", reasoning="Madrid is the capital.", answer="Madrid").with_inputs(
            "question"
        )
    ]

    save_dir = tmp_path / "cot_prog"
    cot.save(save_dir, save_program=True, format="json")
    loaded = dspy.load(save_dir)

    assert isinstance(loaded, dspy.ChainOfThought)
    assert loaded.predict.signature.equals(cot.predict.signature)
    assert loaded.predict.signature.fields["reasoning"].annotation is dspy.Reasoning
    assert loaded.predict.signature.fields["reasoning"].json_schema_extra["__dspy_reasoning_fallback"] == "plan"
    assert loaded.predict.config == {"temperature": 0.4}
    assert loaded.predict.demos == cot.predict.dump_state(json_mode=True)["demos"]

    with dspy.context(lm=DummyLM([{"plan": "Paris is the capital of France.", "answer": "Paris"}])):
        result = loaded(question="Capital of France?")
    assert result.answer == "Paris"
    assert "Paris" in str(result.reasoning)

    # Also verify rationale_field_type=str round-trip.
    cot_str = dspy.ChainOfThought("question -> answer", rationale_field_type=str)
    str_dir = tmp_path / "cot_str_prog"
    cot_str.save(str_dir, save_program=True, format="json")
    loaded_str = dspy.load(str_dir)
    assert loaded_str.predict.signature.fields["reasoning"].annotation is str


def test_multi_chain_comparison_json_roundtrip_and_execution(tmp_path, monkeypatch):
    _forbid_cloudpickle(monkeypatch)

    mcc = dspy.MultiChainComparison("question -> answer", M=2, temperature=0.3, max_tokens=150)
    save_dir = tmp_path / "mcc_prog"
    mcc.save(save_dir, save_program=True, format="json")

    loaded = dspy.load(save_dir)
    assert isinstance(loaded, dspy.MultiChainComparison)
    assert loaded.M == 2
    assert loaded.last_key == "answer"
    assert loaded.predict.config == {"temperature": 0.3, "max_tokens": 150}
    assert loaded.predict.signature.equals(mcc.predict.signature)

    completions = [
        dspy.Prediction(reasoning="2+2=4", answer="4"),
        dspy.Prediction(reasoning="two plus two is four", answer="4"),
    ]
    with dspy.context(lm=DummyLM([{"rationale": "Both agree.", "answer": "4"}])):
        result = loaded(completions=completions, question="What is 2+2?")
    assert result.answer == "4"
    assert result.rationale == "Both agree."


def test_react_json_roundtrip_tool_binding_refusal_and_execution(tmp_path, monkeypatch):
    _forbid_cloudpickle(monkeypatch)

    calls = []

    def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    def lookup(key: str) -> str:
        """Look up a key."""
        return f"val:{key}"

    agent = dspy.ReAct("question -> answer", tools=[add, lookup], max_iters=5)
    agent.react.signature = agent.react.signature.with_instructions("Use tools precisely.")

    save_dir = tmp_path / "react_prog"
    agent.save(save_dir, save_program=True, format="json")

    program_json = orjson.loads((save_dir / "program.json").read_bytes())
    assert program_json["module"]["config"]["tool_names"] == ["add", "lookup"]

    # Refuses loading when required tools are missing.
    with pytest.raises(ValueError, match=r"Missing required tool\(s\) for ReAct: add, lookup"):
        dspy.load(save_dir)

    with pytest.raises(ValueError, match=r"Missing required tool\(s\) for ReAct: lookup"):
        dspy.load(save_dir, tools=[add])

    # Loads with sequence of callables and executes end-to-end with DummyLM.
    loaded_list = dspy.load(save_dir, tools=[add, lookup])
    assert isinstance(loaded_list, dspy.ReAct)
    assert loaded_list.max_iters == 5
    assert loaded_list.signature.equals(agent.signature)
    assert loaded_list.react.signature.instructions == "Use tools precisely."

    # Also loads with a name->callable mapping.
    loaded_map = dspy.load(save_dir, tools={"add": add, "lookup": lookup})
    assert set(loaded_map.tools) == {"add", "lookup", "submit"}

    dummy_lm = DummyLM(
        [
            {
                "next_thought": "Call add first.",
                "tool_calls": {"tool_calls": [{"name": "add", "args": {"a": 2, "b": 3}}]},
            },
            {
                "next_thought": "Now submit the final answer.",
                "tool_calls": {"tool_calls": [{"name": "submit", "args": {"answer": "5"}}]},
            },
        ]
    )
    with dspy.context(lm=dummy_lm):
        pred = loaded_list(question="What is 2 + 3?")

    assert calls == [(2, 3)]
    assert pred.answer == "5"
    assert pred.termination_reason == "submit"


def test_rlm_json_roundtrip_with_and_without_tools_and_execution(tmp_path, monkeypatch):
    _forbid_cloudpickle(monkeypatch)

    # 1. RLM without tools, with sub_lm.
    rlm_no_tools = dspy.RLM(
        "context, query -> answer",
        max_iters=4,
        max_llm_calls=12,
        max_output_chars=4096,
        verbose=True,
        sub_lm=dspy.LM("openai/gpt-4o-mini", temperature=0.1),
    )
    dir_no_tools = tmp_path / "rlm_no_tools"
    rlm_no_tools.save(dir_no_tools, save_program=True, format="json")

    loaded_no_tools = dspy.load(dir_no_tools)
    assert isinstance(loaded_no_tools, dspy.RLM)
    assert loaded_no_tools.max_iters == 4
    assert loaded_no_tools.max_llm_calls == 12
    assert loaded_no_tools.max_output_chars == 4096
    assert loaded_no_tools.verbose is True
    assert loaded_no_tools.sub_lm is not None
    assert loaded_no_tools.sub_lm.model == "openai/gpt-4o-mini"
    assert loaded_no_tools.signature.equals(rlm_no_tools.signature)

    dummy_lm = DummyLM(
        [
            {
                "plan": "Submit answer directly.",
                "code": "```python\nSUBMIT(answer='42')\n```",
            }
        ]
    )
    with dspy.context(lm=dummy_lm):
        result = loaded_no_tools(context="data", query="meaning?", interpreter_factory=_MockInterpreter)
    assert result.answer == "42"

    # 2. RLM with tools.
    def fetch_doc(doc_id: str) -> str:
        """Fetch a document."""
        return f"doc:{doc_id}"

    rlm_with_tools = dspy.RLM("query -> answer", max_iters=3, tools=[fetch_doc])
    dir_with_tools = tmp_path / "rlm_with_tools"
    rlm_with_tools.save(dir_with_tools, save_program=True, format="json")

    with pytest.raises(ValueError, match=r"Missing required tool\(s\) for RLM: fetch_doc"):
        dspy.load(dir_with_tools)

    loaded_with_tools = dspy.load(dir_with_tools, tools={"fetch_doc": fetch_doc})
    assert "fetch_doc" in loaded_with_tools.tools


def test_flex_json_roundtrip_preserves_module_src_and_tools(tmp_path, monkeypatch):
    _forbid_cloudpickle(monkeypatch)

    def helper(x: str) -> str:
        """Normalize text."""
        return x.strip().upper()

    custom_src = (
        "import dspy\n\n"
        "class CustomFlexModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.pred = dspy.Predict('question -> answer')\n"
        "    def forward(self, question):\n"
        "        return self.pred(question=helper(question))\n"
    )

    flex = dspy.Flex("question: str -> answer: str", tools=[helper], max_predictor_calls=25)
    flex._bind_code(custom_src)
    flex.set_lm(dspy.LM("openai/gpt-4o-mini", temperature=0.3))

    save_dir = tmp_path / "flex_prog"
    flex.save(save_dir, save_program=True, format="json")

    with pytest.raises(ValueError, match=r"Missing required tool\(s\) for Flex: helper"):
        dspy.load(save_dir)

    loaded = dspy.load(save_dir, tools=[helper])
    assert isinstance(loaded, dspy.Flex)
    assert loaded.module_src == custom_src
    assert loaded._max_predictor_calls == 25
    assert loaded._bridge._class_name == "CustomFlexModule"
    assert loaded.signature.equals(flex.signature)
    assert loaded.lm is not None
    assert loaded.lm.model == "openai/gpt-4o-mini"
    assert set(loaded._flex_ctx.context_names()) == {"helper"}


def test_custom_modules_via_safe_modules_zero_arg_and_hooks(tmp_path, monkeypatch):
    _forbid_cloudpickle(monkeypatch)

    # 1. Zero-arg custom Module.
    qa = _TopLevelTwoStepQA()
    qa.draft.signature = qa.draft.signature.with_instructions("Write a concise draft.")
    qa.final.predict.signature = qa.final.predict.signature.with_instructions("Polish the draft.")

    dir_zero = tmp_path / "custom_zero_arg"
    qa.save(dir_zero, save_program=True, format="json")

    with pytest.raises(ValueError, match="safe_modules"):
        dspy.load(dir_zero)

    loaded_zero = dspy.load(dir_zero, safe_modules=[_TopLevelTwoStepQA])
    assert isinstance(loaded_zero, _TopLevelTwoStepQA)
    assert loaded_zero.draft.signature.instructions == "Write a concise draft."
    assert loaded_zero.final.predict.signature.instructions == "Polish the draft."

    with dspy.context(lm=DummyLM([{"draft": "rough"}, {"plan": "refined", "answer": "final"}])):
        out = loaded_zero(question="test?")
    assert out.answer == "final"

    # 2. Custom Module with to_program_config / from_program_config hooks.
    cfg_qa = _TopLevelConfigurableQA(steps=3, tag="v2")
    cfg_qa.predictors_list[0].signature = cfg_qa.predictors_list[0].signature.with_instructions("Step 0 instructions.")
    dir_hooks = tmp_path / "custom_hooks"
    cfg_qa.save(dir_hooks, save_program=True, format="json")

    with pytest.raises(ValueError, match="safe_modules"):
        dspy.load(dir_hooks)

    loaded_hooks = dspy.load(dir_hooks, safe_modules=[_TopLevelConfigurableQA])
    assert isinstance(loaded_hooks, _TopLevelConfigurableQA)
    assert loaded_hooks.steps == 3
    assert loaded_hooks.tag == "v2"
    assert len(loaded_hooks.predictors_list) == 3
    assert loaded_hooks.predictors_list[0].signature.instructions == "Step 0 instructions."

    with dspy.context(lm=DummyLM([{"answer": "s1"}, {"answer": "s2"}, {"answer": "s3"}])):
        out_hooks = loaded_hooks(question="multi?")
    assert out_hooks.answer == "s3"


def test_fail_fast_on_non_declarative_and_unsupported_modules(tmp_path):
    class NeedsArgsModule(dspy.Module):
        def __init__(self, k: int):
            super().__init__()
            self.k = k
            self.p = dspy.Predict("q -> a")

    with pytest.raises(ValueError, match="requires constructor arguments"):
        NeedsArgsModule(3).save(tmp_path / "needs_args", save_program=True, format="json")
    assert not (tmp_path / "needs_args").exists()

    class ExtraStateModule(dspy.Module):
        def __init__(self, threshold: float = 0.5):
            super().__init__()
            self.threshold = threshold
            self.p = dspy.Predict("q -> a")

    with pytest.raises(ValueError, match="non-module attribute"):
        ExtraStateModule(threshold=0.9).save(tmp_path / "extra_state", save_program=True, format="json")

    class HalfHookModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict("q -> a")

        def to_program_config(self):
            return {}

    with pytest.raises(ValueError, match="both `to_program_config\\(\\)` and `from_program_config\\(config\\)`"):
        HalfHookModule().save(tmp_path / "half_hook", save_program=True, format="json")

    class NonJsonHookModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict("q -> a")

        def to_program_config(self):
            return {"fn": lambda x: x}

        @classmethod
        def from_program_config(cls, config):
            return cls()

    with pytest.raises(ValueError, match="non-JSON value"):
        NonJsonHookModule().save(tmp_path / "non_json_hook", save_program=True, format="json")

    # Unsupported built-in modules fail fast.
    best_of_n = dspy.BestOfN(module=dspy.Predict("q -> a"), N=2, reward_fn=lambda args, pred: 1.0, threshold=0.5)
    with pytest.raises(ValueError):
        best_of_n.save(tmp_path / "best_of_n", save_program=True, format="json")

    # RLM / Flex with custom interpreter_factory fail fast.
    rlm_custom_interp = dspy.RLM("q -> a", interpreter_factory=_MockInterpreter)
    with pytest.raises(ValueError, match="custom `interpreter_factory`"):
        rlm_custom_interp.save(tmp_path / "rlm_custom", save_program=True, format="json")

    rlm_hist_proc = dspy.RLM("q -> a", history_processor=lambda h: h)
    with pytest.raises(ValueError, match="history_processor"):
        rlm_hist_proc.save(tmp_path / "rlm_hist", save_program=True, format="json")

    flex_custom_interp = dspy.Flex("q -> a", interpreter_factory=_MockInterpreter)
    with pytest.raises(ValueError, match="custom `interpreter_factory`"):
        flex_custom_interp.save(tmp_path / "flex_custom", save_program=True, format="json")

    # Unsupported custom type in signature fails fast.
    import pydantic

    class CustomType(pydantic.BaseModel):
        value: int

    sig = dspy.make_signature("x: CustomType -> y: str", custom_types={"CustomType": CustomType})
    with pytest.raises(ValueError, match="Unsupported type annotation"):
        dspy.Predict(sig).save(tmp_path / "custom_type", save_program=True, format="json")


def test_save_api_argument_validation_and_legacy_pickle_compat(tmp_path):
    pred = dspy.Predict("q -> a")

    with pytest.raises(ValueError, match="only be specified when `save_program=True`"):
        pred.save(tmp_path / "state.json", save_program=False, format="json")

    with pytest.raises(ValueError, match="Unsupported `format`"):
        pred.save(tmp_path / "bad_fmt", save_program=True, format="yaml")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="modules_to_serialize"):
        pred.save(tmp_path / "bad_mods", save_program=True, modules_to_serialize=[dspy], format="json")

    # Legacy pickle directory still refuses allow_pickle=False and loads with allow_pickle=True.
    pkl_dir = tmp_path / "legacy_pkl"
    pred.save(pkl_dir, save_program=True)
    assert (pkl_dir / "program.pkl").exists()
    with pytest.raises(ValueError, match="Loading with pickle is not allowed"):
        dspy.load(pkl_dir)
    loaded_pkl = dspy.load(pkl_dir, allow_pickle=True)
    assert isinstance(loaded_pkl, dspy.Predict)


@pytest.mark.parametrize(
    ("tampered_module", "tampered_qualname"),
    [
        ("os", "system"),
        ("subprocess", "Popen"),
        ("builtins", "eval"),
        ("posix", "system"),
        ("tests.utils.test_safe_program_serialization", "_TopLevelTwoStepQA"),
    ],
)
def test_adversarial_tampered_class_rejected_with_zero_imports(
    tmp_path, monkeypatch, tampered_module, tampered_qualname
):
    pred = dspy.Predict("q -> a")
    save_dir = tmp_path / "tampered_cls"
    pred.save(save_dir, save_program=True, format="json")

    data = orjson.loads((save_dir / "program.json").read_bytes())
    data["module"]["module"] = tampered_module
    data["module"]["qualname"] = tampered_qualname
    (save_dir / "program.json").write_bytes(orjson.dumps(data))

    imported: list[str] = []
    real_import_module = importlib.import_module

    def tracking_import(name, package=None):
        imported.append(name)
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", tracking_import)

    with pytest.raises(ValueError, match="not in the built-in registry or `safe_modules` allowlist"):
        dspy.load(save_dir)

    assert imported == []


def test_adversarial_safe_modules_requires_exact_module_and_qualname(tmp_path):
    qa = _TopLevelTwoStepQA()
    save_dir = tmp_path / "spoofed_module"
    qa.save(save_dir, save_program=True, format="json")

    data = orjson.loads((save_dir / "program.json").read_bytes())
    data["module"]["module"] = "evil.module"
    (save_dir / "program.json").write_bytes(orjson.dumps(data))

    with pytest.raises(ValueError, match="not in the built-in registry or `safe_modules` allowlist"):
        dspy.load(save_dir, safe_modules=[_TopLevelTwoStepQA])


@pytest.mark.parametrize(
    "malicious_type",
    [
        "__import__('os').system('id')",
        "os.system",
        "subprocess.Popen",
        "builtins.eval",
        "typing.sys",
        "int.__class__.__mro__",
        "list[os.system]",
        "Literal[__import__('os')]",
        "dspy.__builtins__",
        "dspy.Code[__import__('os')]",
        "str | eval",
        "list[" * 40 + "int" + "]" * 40,
    ],
)
def test_adversarial_tampered_signature_type_rejected_with_zero_imports(tmp_path, monkeypatch, malicious_type):
    pred = dspy.Predict("q -> a")
    save_dir = tmp_path / "tampered_type"
    pred.save(save_dir, save_program=True, format="json")

    data = orjson.loads((save_dir / "program.json").read_bytes())
    data["module"]["config"]["signature"]["fields"][0]["annotation"] = malicious_type
    (save_dir / "program.json").write_bytes(orjson.dumps(data))

    imported: list[str] = []
    real_import_module = importlib.import_module

    def tracking_import(name, package=None):
        imported.append(name)
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", tracking_import)

    with pytest.raises(ValueError):
        dspy.load(save_dir)

    assert imported == []


def test_adversarial_tampered_lm_state_blocked_by_default(tmp_path):
    # 1. Predict LM state with injected api_base, base_url, api_key, model_list.
    pred = dspy.Predict("q -> a")
    pred.lm = dspy.LM("openai/gpt-4o-mini")
    save_dir = tmp_path / "tampered_lm"
    pred.save(save_dir, save_program=True, format="json")

    data = orjson.loads((save_dir / "program.json").read_bytes())
    data["module"]["state"]["lm"]["api_base"] = "https://attacker.example/v1"
    data["module"]["state"]["lm"]["base_url"] = "https://attacker.example/v2"
    data["module"]["state"]["lm"]["api_key"] = "sk-attacker-secret"
    data["module"]["state"]["lm"]["model_list"] = [{"model_name": "evil"}]
    (save_dir / "program.json").write_bytes(orjson.dumps(data))

    loaded_safe = dspy.load(save_dir)
    assert loaded_safe.lm is not None
    assert "api_base" not in loaded_safe.lm.kwargs
    assert "base_url" not in loaded_safe.lm.kwargs
    assert "api_key" not in loaded_safe.lm.kwargs
    assert "model_list" not in loaded_safe.lm.kwargs

    loaded_unsafe = dspy.load(save_dir, allow_unsafe_lm_state=True)
    assert loaded_unsafe.lm is not None
    assert loaded_unsafe.lm.kwargs["api_base"] == "https://attacker.example/v1"
    assert loaded_unsafe.lm.kwargs["api_key"] == "sk-attacker-secret"

    # 2. Custom LM class injection refused by default.
    data["module"]["state"]["lm"]["_dspy_lm_class"] = "dspy.utils.dummies.DummyLM"
    (save_dir / "program.json").write_bytes(orjson.dumps(data))
    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.load(save_dir)

    # 3. RLM sub_lm also sanitized by default.
    rlm = dspy.RLM("q -> a", sub_lm=dspy.LM("openai/gpt-4o-mini"))
    rlm_dir = tmp_path / "tampered_rlm_sub_lm"
    rlm.save(rlm_dir, save_program=True, format="json")
    rlm_data = orjson.loads((rlm_dir / "program.json").read_bytes())
    rlm_data["module"]["config"]["sub_lm"]["api_base"] = "https://evil.example/v1"
    rlm_data["module"]["config"]["sub_lm"]["api_key"] = "sk-evil"
    (rlm_dir / "program.json").write_bytes(orjson.dumps(rlm_data))

    loaded_rlm = dspy.load(rlm_dir)
    assert loaded_rlm.sub_lm is not None
    assert "api_base" not in loaded_rlm.sub_lm.kwargs
    assert "api_key" not in loaded_rlm.sub_lm.kwargs


def test_adversarial_tampered_schema_version_malformed_json_and_state_injection(tmp_path):
    pred = dspy.Predict("q -> a")
    save_dir = tmp_path / "tampered_schema"
    pred.save(save_dir, save_program=True, format="json")

    orig_metadata = (save_dir / "metadata.json").read_bytes()
    orig_program = (save_dir / "program.json").read_bytes()

    # Bad metadata schema_version (including bool True == 1).
    for bad_version in (2, 0, "1", True, None):
        meta = orjson.loads(orig_metadata)
        meta["schema_version"] = bad_version
        (save_dir / "metadata.json").write_bytes(orjson.dumps(meta))
        with pytest.raises(ValueError, match="schema_version"):
            dspy.load(save_dir)

    (save_dir / "metadata.json").write_bytes(orig_metadata)

    # Bad program.json schema_version.
    for bad_version in (2, 0, "1", True, None):
        prog = orjson.loads(orig_program)
        prog["schema_version"] = bad_version
        (save_dir / "program.json").write_bytes(orjson.dumps(prog))
        with pytest.raises(ValueError, match="schema_version"):
            dspy.load(save_dir)

    # Unexpected state attribute injection rejected.
    prog = orjson.loads(orig_program)
    prog["module"]["state"]["forward"] = "evil"
    (save_dir / "program.json").write_bytes(orjson.dumps(prog))
    with pytest.raises(ValueError, match="unexpected key"):
        dspy.load(save_dir)

    # Malformed JSON fails closed.
    (save_dir / "program.json").write_bytes(b"{not valid json")
    with pytest.raises(ValueError):
        dspy.load(save_dir)

    # Unknown format in metadata.json fails closed.
    (save_dir / "program.json").write_bytes(orig_program)
    meta = orjson.loads(orig_metadata)
    meta["format"] = "marshal"
    (save_dir / "metadata.json").write_bytes(orjson.dumps(meta))
    with pytest.raises(ValueError, match="Unsupported saved program format"):
        dspy.load(save_dir)


def test_json_program_version_mismatch_logs_warning(tmp_path):
    from dspy.utils.saving import logger

    save_versions = {"python": "3.10", "dspy": "3.0.0", "cloudpickle": "3.0"}
    load_versions = {"python": "3.12", "dspy": "3.5.0", "cloudpickle": "3.1"}

    pred = dspy.Predict("q -> a")
    save_dir = tmp_path / "version_mismatch"

    class _ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    handler = _ListHandler()
    orig_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=save_versions):
            pred.save(save_dir, save_program=True, format="json")
        with patch("dspy.utils.saving.get_dependency_versions", return_value=load_versions):
            loaded = dspy.load(save_dir)
        assert isinstance(loaded, dspy.Predict)
        assert len(handler.messages) == 3
        assert all("There is a mismatch of" in msg for msg in handler.messages)
    finally:
        logger.setLevel(orig_level)
        logger.removeHandler(handler)


def test_predictor_config_cannot_redirect_lm_calls_by_default(tmp_path):
    injected = {"api_base": "https://attacker.example/v1", "base_url": "https://attacker.example", "api_key": "sk-x"}

    pred = dspy.Predict("q -> a", temperature=0.1, api_key="sk-real")
    pred_dir = tmp_path / "predict"
    pred.save(pred_dir, save_program=True, format="json")
    data = orjson.loads((pred_dir / "program.json").read_bytes())
    assert data["module"]["config"]["lm_config"] == {"temperature": 0.1}
    data["module"]["config"]["lm_config"].update(injected)
    (pred_dir / "program.json").write_bytes(orjson.dumps(data))
    assert dspy.load(pred_dir).config == {"temperature": 0.1}
    assert dspy.load(pred_dir, allow_unsafe_lm_state=True).config == {"temperature": 0.1, **injected}

    def search(query: str) -> str:
        """Search."""
        return query

    react_dir = tmp_path / "react"
    dspy.ReAct("q -> a", tools=[search]).save(react_dir, save_program=True, format="json")
    data = orjson.loads((react_dir / "program.json").read_bytes())
    data["module"]["config"]["react_config"] = dict(injected)
    (react_dir / "program.json").write_bytes(orjson.dumps(data))
    assert dspy.load(react_dir, tools=[search]).react.config == {}

    rlm_dir = tmp_path / "rlm"
    dspy.RLM("q -> a").save(rlm_dir, save_program=True, format="json")
    data = orjson.loads((rlm_dir / "program.json").read_bytes())
    data["module"]["config"]["generate_action_config"] = dict(injected)
    data["module"]["config"]["extract_config"] = dict(injected)
    (rlm_dir / "program.json").write_bytes(orjson.dumps(data))
    loaded_rlm = dspy.load(rlm_dir)
    assert loaded_rlm.generate_action.config == {}
    assert loaded_rlm.extract.config == {}

    custom_dir = tmp_path / "custom"
    _TopLevelTwoStepQA().save(custom_dir, save_program=True, format="json")
    data = orjson.loads((custom_dir / "program.json").read_bytes())
    data["module"]["parameters"]["final.predict"]["config"] = dict(injected)
    (custom_dir / "program.json").write_bytes(orjson.dumps(data))
    assert dspy.load(custom_dir, safe_modules=[_TopLevelTwoStepQA]).final.predict.config == {}


class _Inner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("text -> label")


class _Outer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("question -> answer")
        self.inner = _Inner()


def test_custom_module_restores_child_config_signature_and_compiled_children(tmp_path):
    program = _Outer()
    program.answer.update_config(temperature=0.3)
    program.answer.signature = dspy.Signature("question -> answer: int", "Answer with a number.")
    program.inner.classify.demos = [dspy.Example(text="hi", label="greeting")]
    program.inner._compiled = True

    save_dir = tmp_path / "outer"
    program.save(save_dir, save_program=True, format="json")
    loaded = dspy.load(save_dir, safe_modules=[_Outer])

    assert loaded.answer.config == {"temperature": 0.3}
    assert loaded.answer.signature.equals(program.answer.signature)
    assert loaded.answer.signature.output_fields["answer"].annotation is int
    assert loaded.inner._compiled is True
    assert loaded.inner.classify.demos == [{"text": "hi", "label": "greeting"}]
    assert "inner.classify" not in dict(loaded.named_parameters())

    with dspy.context(lm=DummyLM([{"answer": "7"}])):
        assert loaded.answer(question="3 + 4?").answer == 7


def test_custom_module_structure_mismatch_or_bad_entry_is_rejected(tmp_path):
    save_dir = tmp_path / "outer"
    _Outer().save(save_dir, save_program=True, format="json")
    data = orjson.loads((save_dir / "program.json").read_bytes())
    del data["module"]["parameters"]["inner.classify"]
    (save_dir / "program.json").write_bytes(orjson.dumps(data))
    with pytest.raises(ValueError, match=r"missing \['inner.classify'\]"):
        dspy.load(save_dir, safe_modules=[_Outer])

    data = orjson.loads((save_dir / "program.json").read_bytes())
    data["module"]["parameters"]["inner.classify"] = {"state": {}, "signature": {}}
    data["module"]["parameters"]["answer"]["config"] = {"temperature": 0.9}
    (save_dir / "program.json").write_bytes(orjson.dumps(data))
    with pytest.raises(ValueError):
        dspy.load(save_dir, safe_modules=[_Outer])


def test_signature_field_constraints_roundtrip_and_unknown_metadata_is_rejected(tmp_path):
    class Constrained(dspy.Signature):
        code: str = dspy.InputField(pattern="^[A-Z]+$", min_length=2, max_length=8, strict=True)
        count: int = dspy.InputField(gt=0, le=10, multiple_of=2)
        ratio: float = dspy.InputField(allow_inf_nan=False)
        label: str = dspy.OutputField(coerce_numbers_to_str=True)
        ids: list[int] = dspy.OutputField(fail_fast=True)

    save_dir = tmp_path / "constrained"
    dspy.Predict(Constrained).save(save_dir, save_program=True, format="json")
    loaded = dspy.load(save_dir).signature
    for name, field in Constrained.fields.items():
        assert repr(loaded.fields[name].metadata) == repr(field.metadata), name
    with pytest.raises(pydantic.ValidationError):
        loaded(code="abc", count=2, ratio=0.5, label="x", ids=[1])

    data = orjson.loads((save_dir / "program.json").read_bytes())
    data["module"]["config"]["signature"]["fields"][0]["constraints"]["json_schema_extra"] = {"x": 1}
    (save_dir / "program.json").write_bytes(orjson.dumps(data))
    with pytest.raises(ValueError, match="unsupported constraint 'json_schema_extra'"):
        dspy.load(save_dir)

    class Validated(dspy.Signature):
        question: str = dspy.InputField()
        answer: Annotated[str, pydantic.AfterValidator(str.upper)] = dspy.OutputField()

    with pytest.raises(ValueError, match="unsupported"):
        dspy.Predict(Validated).save(tmp_path / "validated", save_program=True, format="json")
    assert not (tmp_path / "validated").exists()


class _Thresholded(dspy.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.5
        self.classify = dspy.Predict("text -> label")

    def to_program_config(self):
        return {}

    @classmethod
    def from_program_config(cls, config):
        return cls()

    def dump_state(self, json_mode=True):
        return {**super().dump_state(json_mode=json_mode), "threshold": self.threshold}

    def load_state(self, state, *, allow_unsafe_lm_state=False):
        self.threshold = state["threshold"]
        params = {name: value for name, value in state.items() if name != "threshold"}
        return super().load_state(params, allow_unsafe_lm_state=allow_unsafe_lm_state)


def test_custom_module_state_methods_are_preserved(tmp_path):
    program = _Thresholded()
    program.threshold = 0.9
    program.classify.update_config(temperature=0.2)
    program.classify.demos = [dspy.Example(text="hi", label="greeting")]

    save_dir = tmp_path / "thresholded"
    program.save(save_dir, save_program=True, format="json")
    loaded = dspy.load(save_dir, safe_modules=[_Thresholded])

    assert loaded.threshold == 0.9
    assert loaded.classify.config == {"temperature": 0.2}
    assert loaded.classify.demos == [{"text": "hi", "label": "greeting"}]


def test_custom_module_state_cannot_reintroduce_predictor_config(tmp_path):
    save_dir = tmp_path / "thresholded"
    _Thresholded().save(save_dir, save_program=True, format="json")
    data = orjson.loads((save_dir / "program.json").read_bytes())
    data["module"]["state"]["classify"]["config"] = {"api_base": "https://attacker.example/v1"}
    (save_dir / "program.json").write_bytes(orjson.dumps(data))
    with pytest.raises(ValueError, match="unexpected key\\(s\\): config"):
        dspy.load(save_dir, safe_modules=[_Thresholded])

    outer_dir = tmp_path / "outer"
    _Outer().save(outer_dir, save_program=True, format="json")
    data = orjson.loads((outer_dir / "program.json").read_bytes())
    data["module"]["state"] = {"answer": {"config": {"api_base": "https://attacker.example/v1"}}}
    (outer_dir / "program.json").write_bytes(orjson.dumps(data))
    with pytest.raises(ValueError, match="only saved for modules that override"):
        dspy.load(outer_dir, safe_modules=[_Outer])
