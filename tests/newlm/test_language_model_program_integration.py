import re

import pytest

import dspy

dspy.settings.experimental = True

class CapitalQA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class CapitalProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(CapitalQA)

    def forward(self, question: str):
        return self.answer(question=question)


class ScriptedCapitalLM(dspy.BaseLM):
    def __init__(self, model="test/scripted-capitals", cache=False, require_demo: bool = False, **kwargs):
        super().__init__(model=model, cache=cache, **kwargs)
        self.require_demo = require_demo
        self.requests = []

    def dump_state(self) -> dict:
        state = super().dump_state()
        if self.require_demo:
            state["require_demo"] = self.require_demo
        return state

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        prompt = _request_text(request)
        question = _extract_field(prompt, "question") or ""
        has_demo = any(
            marker in prompt
            for marker in (
                "[[ ## answer ## ]]\nParis",
                "[[ ## answer ## ]]\nBerlin",
                "[[ ## answer ## ]]\nTokyo",
            )
        )
        answer = "unknown"
        if "capital expert" in prompt.lower() and (has_demo or not self.require_demo):
            answer = _capital_answer(question)
        return dspy.LMResponse.from_text(
            f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]",
            model=request.model,
        )


class ScriptedMIPROPromptLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="test/scripted-mipro-prompt", cache=False)
        self.requests = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        prompt = _request_text(request)
        if "[[ ## proposed_instruction ## ]]" in prompt or "`[[ ## proposed_instruction ## ]]`" in prompt:
            text = "You are a capital expert. Return only the city name for capital-city questions."
            field = "proposed_instruction"
        elif "[[ ## observations ## ]]" in prompt or "`[[ ## observations ## ]]`" in prompt:
            text = "The dataset contains country capital questions with short city-name answers."
            field = "observations"
        elif "[[ ## summary ## ]]" in prompt or "`[[ ## summary ## ]]`" in prompt:
            text = "Country questions should be answered with their capitals."
            field = "summary"
        elif "[[ ## program_description ## ]]" in prompt or "`[[ ## program_description ## ]]`" in prompt:
            text = "The program answers capital-city questions."
            field = "program_description"
        elif "[[ ## module_description ## ]]" in prompt or "`[[ ## module_description ## ]]`" in prompt:
            text = "This module maps one country question to one city answer."
            field = "module_description"
        else:
            text = "Generated helper text."
            field = "answer"
        return dspy.LMResponse.from_text(f"[[ ## {field} ## ]]\n{text}\n\n[[ ## completed ## ]]", model=request.model)


def test_predict_program_runs_end_to_end_with_language_model():
    lm = ScriptedCapitalLM()
    program = CapitalProgram()

    with dspy.context(lm=lm):
        baseline = program(question="What is the capital of France?")

    assert baseline.answer == "unknown"
    assert isinstance(lm.requests[0], dspy.LMRequest)

    optimized = program.deepcopy()
    optimized.answer.signature = optimized.answer.signature.with_instructions(
        "You are a capital expert. Return only the city name for capital-city questions."
    )

    with dspy.context(lm=lm):
        prediction = optimized(question="What is the capital of France?")

    assert prediction.answer == "Paris"


def test_language_model_program_save_and_load_round_trips_lm_and_instruction(tmp_path):
    program = CapitalProgram()
    program.answer.lm = ScriptedCapitalLM()
    program.answer.signature = program.answer.signature.with_instructions(
        "You are a capital expert. Return only the city name for capital-city questions."
    )

    path = tmp_path / "capital_program.json"
    program.save(path)

    loaded = CapitalProgram()
    loaded.load(path)

    assert isinstance(loaded.answer.lm, ScriptedCapitalLM)
    assert "capital expert" in loaded.answer.signature.instructions
    prediction = loaded(question="What is the capital of Germany?")
    assert prediction.answer == "Berlin"


def test_tiny_miprov2_run_with_language_models_changes_instruction_and_demos(tmp_path):
    pytest.importorskip("optuna", reason="MIPROv2 requires the optional `optuna` dependency")
    task_lm = ScriptedCapitalLM(require_demo=True)
    prompt_lm = ScriptedMIPROPromptLM()
    trainset = [
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        dspy.Example(question="What is the capital of Germany?", answer="Berlin").with_inputs("question"),
        dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
    ]
    valset = [
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        dspy.Example(question="What is the capital of Germany?", answer="Berlin").with_inputs("question"),
    ]

    def metric(example, pred, trace=None):
        return pred.answer == example.answer

    program = CapitalProgram()
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto=None,
        num_candidates=3,
        max_bootstrapped_demos=0,
        max_labeled_demos=1,
        num_threads=1,
        max_errors=10,
        verbose=False,
    )

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
        num_trials=6,
        minibatch=False,
        program_aware_proposer=False,
        data_aware_proposer=False,
        tip_aware_proposer=False,
        fewshot_aware_proposer=False,
    )

    assert optimized.answer.signature.instructions != program.answer.signature.instructions
    assert "capital expert" in optimized.answer.signature.instructions
    assert len(optimized.answer.demos) > 0

    with dspy.context(lm=task_lm):
        prediction = optimized(question="What is the capital of Japan?")
    assert prediction.answer == "Tokyo"

    path = tmp_path / "optimized_capital_program.json"
    optimized.save(path)
    loaded = CapitalProgram()
    loaded.load(path)

    assert "capital expert" in loaded.answer.signature.instructions
    assert len(loaded.answer.demos) > 0
    with dspy.context(lm=task_lm):
        loaded_prediction = loaded(question="What is the capital of France?")
    assert loaded_prediction.answer == "Paris"


def _request_text(request: dspy.LMRequest) -> str:
    return "\n\n".join(message.text or "" for message in request.messages)


def _extract_field(text: str, field: str) -> str | None:
    matches = list(re.finditer(rf"\[\[ ## {field} ## \]\]\s*(.*?)(?=\n\s*\[\[ ##|\Z)", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _capital_answer(question: str) -> str:
    question = question.lower()
    if "france" in question:
        return "Paris"
    if "germany" in question:
        return "Berlin"
    if "japan" in question:
        return "Tokyo"
    return "unknown"
