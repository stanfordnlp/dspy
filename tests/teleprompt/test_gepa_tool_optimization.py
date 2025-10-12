from types import SimpleNamespace

import dspy
from dspy import Example
from dspy.teleprompt.gepa import gepa_utils
from dspy.utils.dummies import DummyLM


def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception:
        return "Error"


def search(query: str) -> str:
    return f"Results for: {query}"


def simple_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    pred_str = str(prediction.answer).strip()
    expected = str(example.answer).strip()
    score = 1.0 if pred_str == expected else 0.0
    return dspy.Prediction(score=score, feedback="Correct" if score == 1.0 else "Wrong")


def make_example(question: str, answer: str) -> Example:
    return Example(question=question, answer=answer).with_inputs("question")


def make_reflection_entry(question: str, answer: str, feedback: str, score: float = 1.0) -> dict:
    return {
        "Inputs": {"question": question},
        "Generated Outputs": {"answer": answer},
        "Feedback": f"Score: {score}.\n{feedback}",
    }


def make_react_module(tool_specs, *, max_iters: int = 3):
    class SimpleReact(dspy.Module):
        def __init__(self):
            super().__init__()
            tools = [dspy.Tool(fn, name=name, desc=desc) for name, desc, fn in tool_specs]
            self.agent = dspy.ReAct(
                "question -> answer",
                tools=tools,
                max_iters=max_iters,
            )

        def forward(self, question: str):
            return self.agent(question=question)

    return SimpleReact()


def make_nested_react_module(main_tool_specs, *, nested_tool_specs, max_iters: int = 3):
    class NestedReact(dspy.Module):
        def __init__(self):
            super().__init__()
            nested_tools = [dspy.Tool(fn, name=name, desc=desc) for name, desc, fn in nested_tool_specs]
            self.subagent = dspy.ReAct(
                "task -> result",
                tools=nested_tools,
                max_iters=max_iters,
            )

            def spawn_subagent(task: str) -> str:
                return self.subagent(task=task).result

            spawn_tool = dspy.Tool(spawn_subagent, name="spawn_subagent", desc="Spawns helper agent.")
            main_tools = [dspy.Tool(fn, name=name, desc=desc) for name, desc, fn in main_tool_specs]
            self.agent = dspy.ReAct(
                "question -> answer",
                tools=[spawn_tool, *main_tools],
                max_iters=max_iters,
            )

        def forward(self, question: str):
            return self.agent(question=question)

    return NestedReact()


def build_adapter_for_program(
    program,
    *,
    custom_instruction_proposer=None,
    reflection_lm=None,
    optimize_tool_descriptions: bool = True,
):
    predictor_names = sorted(name for name, _ in program.named_predictors())
    if not predictor_names:
        raise ValueError("program must expose at least one predictor")

    def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=1.0, feedback="ok")

    feedback_map = {}
    for name in predictor_names:
        feedback_map[name] = lambda *args, _name=name, **kwargs: dspy.Prediction(
            score=1.0, feedback=f"{_name}-fb"
        )

    adapter = gepa_utils.DspyAdapter(
        student_module=program,
        metric_fn=metric_fn,
        feedback_map=feedback_map,
        failure_score=0.0,
        reflection_lm=reflection_lm,
        custom_instruction_proposer=custom_instruction_proposer,
        optimize_tool_descriptions=optimize_tool_descriptions,
    )

    return adapter, predictor_names


def stub_optimize(monkeypatch, *, new_descs, captured_seed):
    def fake_optimize(*, seed_candidate, **kwargs):
        captured_seed.update(seed_candidate)
        best_candidate = dict(seed_candidate)
        for tool_name, desc in new_descs.items():
            best_candidate[f"tool:{tool_name}"] = desc
        return SimpleNamespace(best_candidate=best_candidate)

    monkeypatch.setattr("gepa.optimize", fake_optimize)


def test_gepa_updates_nested_agent_tools(monkeypatch):
    program = make_nested_react_module(
        main_tool_specs=[("calculator", "Does math", calculator)],
        nested_tool_specs=[("search", "Searches", search)],
        max_iters=1,
    )

    original_descs = {
        "calculator": program.agent.tools["calculator"].desc,
        "spawn_subagent": program.agent.tools["spawn_subagent"].desc,
        "search": program.subagent.tools["search"].desc,
    }

    new_descs = {
        "calculator": "Clarify how to perform arithmetic precisely.",
        "spawn_subagent": "Explain when to spawn a helper agent.",
        "search": "Improve how search guidance is presented.",
    }

    captured_seed: dict[str, str] = {}
    dspy.settings.configure(lm=DummyLM([{"q": "question", "a": "answer"}]))
    reflection_lm = DummyLM([{"improved_instruction": "unused"}])

    stub_optimize(monkeypatch, new_descs=new_descs, captured_seed=captured_seed)
    optimizer = dspy.GEPA(
        metric=simple_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=3,
        optimize_tool_descriptions=True,
    )
    trainset = [
        make_example("What is 2 + 2?", "4"),
    ]
    optimized = optimizer.compile(program, trainset=trainset)

    for tool_name, original in original_descs.items():
        assert captured_seed[f"tool:{tool_name}"] == original

    assert optimized.agent.tools["calculator"].desc == new_descs["calculator"]
    assert optimized.agent.tools["spawn_subagent"].desc == new_descs["spawn_subagent"]
    assert optimized.subagent.tools["search"].desc == new_descs["search"]


def test_reflective_dataset_shares_feedback_across_shared_tools():
    shared_tool = dspy.Tool(calculator, name="calculator", desc="Original calculator guidance")

    class DualReact(dspy.Module):
        def __init__(self):
            super().__init__()
            self.agent_a = dspy.ReAct("question -> answer", tools=[shared_tool], max_iters=1)
            self.agent_b = dspy.ReAct("question -> answer", tools=[shared_tool], max_iters=1)

        def forward(self, question: str):
            return dspy.Prediction(answer="unused")

    program = DualReact()
    adapter, predictor_names = build_adapter_for_program(
        program,
        reflection_lm=DummyLM([{"improved_instruction": "Better"}]),
    )

    candidate = {}
    for name in predictor_names:
        candidate[name] = f"{name}-instruction"
    candidate["tool:calculator"] = shared_tool.desc

    program = adapter.build_program(candidate)
    predictor_lookup = {name: pred for name, pred in program.named_predictors()}

    trajectories: list[dict] = []
    for index, name in enumerate(predictor_names):
        predictor = predictor_lookup[name]
        trace_entry = (
            predictor,
            {"question": f"Request {index + 1}"},
            dspy.Prediction(answer=f"Response {index + 1}"),
        )
        trajectories.append(
            {
                "trace": [trace_entry],
                "example": make_example(
                    f"Request {index + 1}",
                    f"Response {index + 1}",
                ),
                "prediction": dspy.Prediction(answer=f"Response {index + 1}"),
                "score": 1.0,
            }
        )

    eval_batch = SimpleNamespace(outputs=[], scores=[], trajectories=trajectories)
    components_to_update = [*predictor_names, "tool:calculator"]

    reflective_dataset = adapter.make_reflective_dataset(candidate, eval_batch, components_to_update)

    for name in predictor_names:
        assert name in reflective_dataset
    assert "tool:calculator" in reflective_dataset
    assert len(reflective_dataset["tool:calculator"]) == len(predictor_names)

    feedback_texts = [item["Feedback"] for item in reflective_dataset["tool:calculator"]]
    for name in predictor_names:
        assert any(name in feedback for feedback in feedback_texts)


def test_dspy_adapter_uses_custom_instruction_and_tool_proposers(monkeypatch):
    program = make_react_module([("toolA", "Original tool desc", lambda arg: arg)])

    tool_calls: list[tuple[dict, list[str]]] = []

    class MockToolProposer:
        def __call__(self, *, candidate, reflective_dataset, components_to_update):
            tool_calls.append((dict(candidate), list(components_to_update)))
            return {component: f"tool-new-{component}" for component in components_to_update}

    monkeypatch.setattr(
        "dspy.teleprompt.gepa.instruction_proposal.ToolProposer",
        MockToolProposer,
    )

    class MockInstructionProposer:
        def __init__(self):
            self.calls: list[list[str]] = []

        def __call__(self, *, candidate, reflective_dataset, components_to_update):
            self.calls.append(list(components_to_update))
            return {name: f"instr-new-{name}" for name in components_to_update}

    instruction_proposer = MockInstructionProposer()

    adapter, predictor_names = build_adapter_for_program(
        program,
        custom_instruction_proposer=instruction_proposer,
        reflection_lm=DummyLM([{"improved_instruction": "Better"}]),
    )

    predictor_name = predictor_names[0]
    tool_key = "tool:toolA"
    candidate = {
        predictor_name: "Base instruction",
        tool_key: program.agent.tools["toolA"].desc,
    }
    reflective_dataset = {
        predictor_name: [
            make_reflection_entry(
                "When should I ask for help?",
                "Use toolA when delegation unblocks progress.",
                "Clarify the decision boundary.",
            )
        ],
        tool_key: [
            make_reflection_entry(
                "When should I ask for help?",
                "Use toolA when delegation unblocks progress.",
                "Highlight the tool's specialty.",
            )
        ],
    }

    updated = adapter.propose_new_texts(candidate, reflective_dataset, [predictor_name, tool_key])

    assert instruction_proposer.calls == [[predictor_name]]
    assert tool_calls == [(candidate, [tool_key])]
    assert updated[predictor_name] == f"instr-new-{predictor_name}"
    assert updated[tool_key] == f"tool-new-{tool_key}"


def test_gepa_overwrites_single_react_tool_description(monkeypatch):
    program = make_react_module([("calculator", "Does math", calculator)], max_iters=1)
    original_desc = program.agent.tools["calculator"].desc

    new_descs = {"calculator": "Clarify how to perform arithmetic precisely."}
    captured_seed: dict[str, str] = {}

    dspy.settings.configure(lm=DummyLM([{"q": "question", "a": "answer"}]))
    reflection_lm = DummyLM([{"improved_instruction": "unused"}])

    stub_optimize(monkeypatch, new_descs=new_descs, captured_seed=captured_seed)
    optimizer = dspy.GEPA(
        metric=simple_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=3,
        optimize_tool_descriptions=True,
    )
    trainset = [
        make_example("Compute 3 + 5.", "8"),
    ]
    optimized = optimizer.compile(program, trainset=trainset)

    assert captured_seed["tool:calculator"] == original_desc
    assert optimized.agent.tools["calculator"].desc == new_descs["calculator"]
    assert optimized.agent.tools["calculator"].desc != original_desc
