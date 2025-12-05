"""Tests for GEPA's tool optimization (ReAct modules).

Test categories:
1. Detection - Compile-time detection of dspy.ReAct modules
2. Application - build_program applies optimized instructions and tool descriptions

DSPy ReAct Design Note:
    DSPy's ReAct uses two predictors:
    - react: reasoning/acting loop
    - extract: structured output synthesis

    We optimize extract.predict as it's called once with the complete trajectory
    and produces all output fields.
"""

import json

import gepa
from gepa import optimize as gepa_optimize

import dspy
from dspy.teleprompt.gepa.gepa_utils import TOOL_MODULE_PREFIX, DspyAdapter
from dspy.utils.dummies import DummyLM


# Test tool fixtures
def search(query: str) -> str:
    """Test search tool."""
    return f"Search: {query}"


def calculate(expr: str) -> str:
    """Test calculator tool."""
    return str(eval(expr))


def analyze(data: str) -> str:
    """Test analyzer tool."""
    return f"Analysis: {data}"


def setup_seed_candidate_capture(monkeypatch):
    """Capture seed_candidate dict passed to gepa.optimize."""
    captured = {}

    def capture_optimize(seed_candidate, **kwargs):
        captured.update(seed_candidate)
        return gepa_optimize(seed_candidate=seed_candidate, **kwargs)

    monkeypatch.setattr(gepa, "optimize", capture_optimize)
    return captured


def create_optimizer(task_responses, reflection_responses):
    """Create GEPA optimizer with explicit LM responses.

    Args:
        task_responses: List of dicts for task LM (e.g., [{"answer": "test"}])
        reflection_responses: List of dicts for reflection LM

    Returns:
        tuple: (optimizer, trainset)
    """
    task_lm = DummyLM(task_responses)
    reflection_lm = DummyLM(reflection_responses)

    dspy.settings.configure(lm=task_lm)

    optimizer = dspy.GEPA(
        metric=lambda example, pred, trace=None, pred_name=None, pred_trace=None: dspy.Prediction(score=0.5, feedback="ok"),
        reflection_lm=reflection_lm,
        max_metric_calls=2,
        enable_tool_optimization=True,
    )

    trainset = [dspy.Example(query="test", answer="test").with_inputs("query")]
    return optimizer, trainset


def get_predictor_name(program, predictor):
    """Find predictor name by object identity in named_predictors().

    Args:
        program: DSPy module
        predictor: Predictor object to find

    Returns:
        str: Predictor name (e.g., "pred", "agent.pred")
    """
    for name, pred in program.named_predictors():
        if pred is predictor:
            return name
    raise ValueError(f"Predictor not found: {predictor}")


def test_skip_predictor_without_tools(monkeypatch):
    """Skip predictors without Tool annotations."""
    seed_candidate = setup_seed_candidate_capture(monkeypatch)

    class PlainSignature(dspy.Signature):
        """Answer questions."""
        query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class PlainAgent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict(PlainSignature)

        def forward(self, query):
            return self.pred(query=query)

    program = PlainAgent()
    optimizer, trainset = create_optimizer(
        task_responses=[{"answer": "test"}] * 20,  # Repeat for GEPA iterations
        reflection_responses=[{"improved_instruction": "optimized"}] * 20  # Repeat for GEPA iterations
    )
    optimizer.compile(program, trainset=trainset, valset=trainset)

    predictor_name = get_predictor_name(program, program.pred)
    assert predictor_name in seed_candidate

    # Should be plain string instruction, not JSON config
    instruction = seed_candidate[predictor_name]
    assert isinstance(instruction, str)


def test_detect_react_module(monkeypatch):
    """Detect ReAct module with tools."""
    seed_candidate = setup_seed_candidate_capture(monkeypatch)

    program = dspy.ReAct("question -> answer", tools=[search])
    optimizer, trainset = create_optimizer(
        task_responses=[
            {"next_thought": "I should search", "next_tool_name": "search", "next_tool_args": {"query": "test"}},
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Based on search", "answer": "test"},
        ] * 20,  # Repeat for GEPA iterations
        reflection_responses=[
            {
                "improved_predictor_instruction": "optimized react",
                "improved_extract_instruction": "optimized extract",
                "improved_tool_search_desc": "optimized search desc",
                "improved_tool_search_arg_query_desc": "optimized query desc"
            }
        ] * 20  # Repeat for GEPA iterations
    )
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Verify detection - use extract.predict as primary (for tracing)
    extract_name = get_predictor_name(program, program.extract.predict)
    component_key = f"{TOOL_MODULE_PREFIX}:{extract_name}"
    assert component_key in seed_candidate

    tool_config = json.loads(seed_candidate[component_key])
    assert "tools" in tool_config


def test_detect_multiple_react_modules(monkeypatch):
    """Detect multiple ReAct modules in workflow."""
    seed_candidate = setup_seed_candidate_capture(monkeypatch)

    class Workflow(dspy.Module):
        def __init__(self):
            super().__init__()
            self.searcher = dspy.ReAct("query -> results", tools=[search])
            self.analyzer = dspy.ReAct("data -> analysis", tools=[analyze])

        def forward(self, query):
            results = self.searcher(query=query)
            return self.analyzer(data=results.results)

    program = Workflow()
    optimizer, trainset = create_optimizer(
        task_responses=[
            {"next_thought": "Searching", "next_tool_name": "search", "next_tool_args": {"query": "test"}},
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Found results", "results": "data"},
            {"next_thought": "Analyzing", "next_tool_name": "analyze", "next_tool_args": {"data": "test"}},
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Analyzed", "analysis": "result"},
        ] * 20,  # Repeat for GEPA iterations
        reflection_responses=[
            {
                "improved_predictor_instruction": "opt react search",
                "improved_extract_instruction": "opt extract search",
                "improved_tool_search_desc": "opt search desc",
                "improved_tool_search_arg_query_desc": "opt query desc"
            },
            {
                "improved_predictor_instruction": "opt react analyze",
                "improved_extract_instruction": "opt extract analyze",
                "improved_tool_analyze_desc": "opt analyze desc",
                "improved_tool_analyze_arg_data_desc": "opt data desc"
            }
        ] * 20  # Repeat for GEPA iterations
    )
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Verify both detected - use extract.predict as primary (for tracing)
    searcher_name = get_predictor_name(program, program.searcher.extract.predict)
    analyzer_name = get_predictor_name(program, program.analyzer.extract.predict)

    searcher_key = f"{TOOL_MODULE_PREFIX}:{searcher_name}"
    analyzer_key = f"{TOOL_MODULE_PREFIX}:{analyzer_name}"

    assert searcher_key in seed_candidate
    assert analyzer_key in seed_candidate


def test_apply_optimized_react_descriptions():
    """Apply optimized tool descriptions to ReAct modules."""

    program = dspy.ReAct("question -> answer", tools=[search])

    # Create mock optimized candidate - use extract.predict as primary (for tracing)
    react_name = get_predictor_name(program, program.react)
    extract_predict_name = get_predictor_name(program, program.extract.predict)

    component_key = f"{TOOL_MODULE_PREFIX}:{extract_predict_name}"

    optimized_candidate = {
        component_key: json.dumps({
            react_name: "OPTIMIZED: React instruction",
            extract_predict_name: "OPTIMIZED: Extract instruction",
            "tools": {
                "search": {
                    "desc": "OPTIMIZED: Search tool",
                    "args": {"query": {"type": "string"}},
                }
            }
        })
    }

    # Apply optimizations
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=lambda example, pred, trace=None: 0.5,
        feedback_map={},
        enable_tool_optimization=True,
    )
    rebuilt = adapter.build_program(optimized_candidate)

    # Verify instructions updated
    assert rebuilt.react.signature.instructions == "OPTIMIZED: React instruction"
    assert rebuilt.extract.predict.signature.instructions == "OPTIMIZED: Extract instruction"

    # Verify tool updated
    assert rebuilt.tools["search"].desc == "OPTIMIZED: Search tool"


def test_detect_nested_react_modules(monkeypatch):
    """Detect ReAct modules in nested program structure."""
    seed_candidate = setup_seed_candidate_capture(monkeypatch)

    class Worker(dspy.Module):
        def __init__(self):
            super().__init__()
            self.react = dspy.ReAct("task -> result", tools=[analyze])

        def forward(self, task):
            return self.react(task=task)

    class Orchestrator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.searcher = dspy.ReAct("query -> results", tools=[search])
            self.worker = Worker()

        def forward(self, query):
            results = self.searcher(query=query)
            return self.worker(task=results.results)

    program = Orchestrator()
    optimizer, trainset = create_optimizer(
        task_responses=[
            {"next_thought": "Search", "next_tool_name": "search", "next_tool_args": {"query": "test"}},
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Found", "results": "data"},
            {"next_thought": "Analyze", "next_tool_name": "analyze", "next_tool_args": {"data": "test"}},
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Analyzed", "result": "final"},
        ] * 20,  # Repeat for GEPA iterations
        reflection_responses=[
            {
                "improved_predictor_instruction": "opt react search",
                "improved_extract_instruction": "opt extract search",
                "improved_tool_search_desc": "opt search desc",
                "improved_tool_search_arg_query_desc": "opt query desc"
            },
            {
                "improved_predictor_instruction": "opt react analyze",
                "improved_extract_instruction": "opt extract analyze",
                "improved_tool_analyze_desc": "opt analyze desc",
                "improved_tool_analyze_arg_data_desc": "opt data desc"
            }
        ] * 20  # Repeat for GEPA iterations
    )
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Verify nested modules detected with full paths - use extract.predict as primary (for tracing)
    searcher_name = get_predictor_name(program, program.searcher.extract.predict)
    worker_extract_name = get_predictor_name(program, program.worker.react.extract.predict)

    searcher_key = f"{TOOL_MODULE_PREFIX}:{searcher_name}"
    worker_key = f"{TOOL_MODULE_PREFIX}:{worker_extract_name}"

    assert searcher_key in seed_candidate
    assert worker_key in seed_candidate

    # Verify full paths preserved (not truncated)
    assert "searcher" in searcher_name  # Contains parent path
    assert "worker" in worker_extract_name  # Contains nested path


def test_selective_optimization_with_none_returns():
    """Verify selective optimization when reflection LM returns None for some fields."""

    program = dspy.ReAct("question -> answer", tools=[search, calculate])

    react_name = get_predictor_name(program, program.react)
    extract_name = get_predictor_name(program, program.extract.predict)
    component_key = f"{TOOL_MODULE_PREFIX}:{extract_name}"

    # Mock selective optimization (only react instruction and search tool updated)
    optimized_candidate = {
        component_key: json.dumps({
            react_name: "OPTIMIZED: React instruction",
            extract_name: program.extract.predict.signature.instructions,
            "tools": {
                "search": {
                    "desc": "OPTIMIZED: Search tool",
                    "args": {"query": {"type": "string"}},
                }
            }
        })
    }

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=lambda example, pred, trace=None: 0.5,
        feedback_map={},
        enable_tool_optimization=True,
    )
    rebuilt = adapter.build_program(optimized_candidate)

    # Verify selective updates
    assert rebuilt.react.signature.instructions == "OPTIMIZED: React instruction"
    assert rebuilt.extract.predict.signature.instructions == program.extract.predict.signature.instructions
    assert rebuilt.tools["search"].desc == "OPTIMIZED: Search tool"

    # Original unchanged (calculate not in optimized candidate)
    assert rebuilt.tools["calculate"].desc == program.tools["calculate"].desc
