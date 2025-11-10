"""Tests for GEPA's generic tool optimization (beyond ReAct modules).

Tests the new generic tool optimization pathway that detects and optimizes
ANY dspy.Module using dspy.Tool, not just dspy.ReAct modules.

What we test:
1. Detection: Verify predictors with Tool-typed input fields are detected at compile time
   - JSON config structure is created (vs plain string for non-tool predictors)
   - Config contains "predictor" and "tools" fields
2. Reconstruction: Verify build_program applies optimized tool descriptions
   - Predictor instructions are updated
   - Tool descriptions and arg_desc are updated

What we DON'T test:
- Exact tool extraction from runtime traces (that's internal GEPA behavior)
- We only verify the compile-time detection creates the right structure

Requirements:
- Signatures MUST use class-based definitions with type annotations
- String signatures like "query, tools -> answer" are NOT supported (lose type info)
- Detection is based on INPUT types only (output types don't matter)
"""

import json

import pytest

import dspy
from dspy import Example
from dspy.teleprompt.gepa.gepa_utils import TOOL_MODULE_PREFIX
from dspy.utils.dummies import DummyLM


def setup_capture_for_base_program(monkeypatch):
    """Capture base_program snapshot at compile time."""
    captured_base_program = {}

    from gepa import optimize as original_optimize

    def capture_optimize(seed_candidate, **kwargs):
        captured_base_program.update(seed_candidate)
        return original_optimize(seed_candidate=seed_candidate, **kwargs)

    import gepa
    monkeypatch.setattr(gepa, "optimize", capture_optimize)

    return captured_base_program


def simple_metric_for_detection(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Simple metric for GEPA detection tests."""
    return dspy.Prediction(score=0.5, feedback="ok")


def mock_optimized_tool_module(optimized_candidate, pred_key, predictor_instruction, tool_descriptions):
    """Helper to mock an optimized tool module in the candidate dict.
        
    Args:
        optimized_candidate: The candidate dict to modify
        pred_key: Predictor key from captured_base_program (e.g., "tool_module:pred")
        predictor_instruction: New predictor instruction
        tool_descriptions: Dict of {tool_name: {"desc": desc, "arg_desc": {arg: desc}}}
    """
    # Parse existing config
    config = json.loads(optimized_candidate[pred_key])

    # Modify predictor instruction
    config["predictor"] = predictor_instruction

    # Modify tool descriptions
    for tool_name, tool_desc in tool_descriptions.items():
        if tool_name not in config["tools"]:
            config["tools"][tool_name] = {"args": {}}

        if "desc" in tool_desc:
            config["tools"][tool_name]["desc"] = tool_desc["desc"]
        if "arg_desc" in tool_desc:
            config["tools"][tool_name]["arg_desc"] = tool_desc["arg_desc"]

    # Serialize back
    optimized_candidate[pred_key] = json.dumps(config)


def create_gepa_optimizer_for_tool_detection():
    """Create GEPA optimizer configured for tool optimization."""
    task_lm = DummyLM([
        {"answer": "test answer"},
    ] * 20)

    reflection_lm = DummyLM([
        {"improved_instruction": "optimized instruction"},
        {"improved_desc": "optimized tool description", "improved_args": "optimized args"},
    ] * 20)

    dspy.settings.configure(lm=task_lm)

    optimizer = dspy.GEPA(
        metric=simple_metric_for_detection,
        reflection_lm=reflection_lm,
        max_metric_calls=2,
        enable_tool_optimization=True,
    )

    trainset = [Example(query="test", answer="test").with_inputs("query")]

    return optimizer, trainset


def test_detect_single_tool(monkeypatch):
    """Detect predictor with single Tool input field.
    
    Tests that GEPA detects a custom module with a single tool at compile time.
    We verify the JSON structure is created, but don't check exact tools
    (those are extracted at runtime from traces).
    """
    captured_base_program = setup_capture_for_base_program(monkeypatch)

    # Create module with single tool (MUST use class signature!)
    class AgentSignature(dspy.Signature):
        """Answer questions using tools."""
        query: str = dspy.InputField()
        tool: dspy.Tool = dspy.InputField()
        answer: str = dspy.OutputField()

    class SimpleAgent(dspy.Module):
        def __init__(self):
            super().__init__()

            def search_web(query: str) -> str:
                """Search the internet."""
                return f"Results for: {query}"

            self.tool = dspy.Tool(search_web, name="search", desc="Search tool")
            self.pred = dspy.Predict(AgentSignature)

        def forward(self, query):
            return self.pred(query=query, tool=self.tool)

    program = SimpleAgent()
    optimizer, trainset = create_gepa_optimizer_for_tool_detection()

    # Run GEPA - should detect tool-using predictor
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Verify compile-time detection created JSON config

    pred_key = f"{TOOL_MODULE_PREFIX}:pred"
    assert pred_key in captured_base_program, f"Expected '{pred_key}' to be detected"

    config = json.loads(captured_base_program[pred_key])

    # Check JSON structure (proves detection worked)
    assert "predictor" in config, "Should have predictor instruction"
    assert isinstance(config["predictor"], str), "Predictor should be string"
    assert "tools" in config, "Should have tools field"
    assert isinstance(config["tools"], dict), "Tools should be dict"
    # Don't check exact tools - that's runtime extraction


def test_detect_tool_list(monkeypatch):
    """Detect predictor with list of Tools.
    
    Tests that GEPA detects a predictor using multiple tools at compile time.
    We verify the JSON structure is created for tool-using predictors.
    """
    captured_base_program = setup_capture_for_base_program(monkeypatch)

    # Create module with tool list (MUST use class signature!)
    class AgentSignature(dspy.Signature):
        """Answer questions using multiple tools."""
        query: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()

    class MultiToolAgent(dspy.Module):
        def __init__(self):
            super().__init__()

            def search_web(query: str) -> str:
                return f"Search: {query}"

            def calculate(expr: str) -> str:
                return f"Calc: {expr}"

            self.tools = [
                dspy.Tool(search_web, name="search", desc="Search tool"),
                dspy.Tool(calculate, name="calc", desc="Calculator tool"),
            ]
            self.pred = dspy.Predict(AgentSignature)

        def forward(self, query):
            return self.pred(query=query, tools=self.tools)

    program = MultiToolAgent()
    optimizer, trainset = create_gepa_optimizer_for_tool_detection()

    # Run GEPA - should detect tool-using predictor
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Verify compile-time detection created JSON config

    pred_key = f"{TOOL_MODULE_PREFIX}:pred"
    assert pred_key in captured_base_program, f"Expected '{pred_key}' to be detected"

    config = json.loads(captured_base_program[pred_key])

    # Check JSON structure
    assert "predictor" in config, "Should have predictor instruction"
    assert "tools" in config, "Should have tools field"
    assert isinstance(config["tools"], dict), "Tools should be dict"


def test_skip_predictor_without_tools(monkeypatch):
    """Negative case: Predictors without Tool annotations should be skipped.
    
    Tests that regular predictors (no Tool fields) get normal string optimization,
    not JSON tool optimization.
    """
    captured_base_program = setup_capture_for_base_program(monkeypatch)

    # Create plain module without tools
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
    optimizer, trainset = create_gepa_optimizer_for_tool_detection()

    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Verify predictor detected as plain string (not JSON)
    pred_key = "pred"
    assert pred_key in captured_base_program, f"Expected '{pred_key}' to be detected"

    pred_config = captured_base_program[pred_key]

    # Should be plain string, not JSON
    assert isinstance(pred_config, str), "Should be string instruction"

    # Verify it's NOT a JSON structure
    try:
        json.loads(pred_config)
        assert False, "Plain predictor should not have JSON config"
    except json.JSONDecodeError:
        pass  # Expected - proves it's a plain string


@pytest.mark.skip(reason="Tool module reconstruction not yet implemented in build_program")
def test_update_tool_and_predictor(monkeypatch):
    """Rebuild program with updated tool descriptions and predictor instructions.
    
    Tests that DspyAdapter.build_program applies optimized tool metadata.
    Follows the same pattern as ReAct test_build_program_single_react.
    
    TODO: Implement tool module reconstruction in DspyAdapter.build_program
    """
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    captured_base_program = setup_capture_for_base_program(monkeypatch)

    # Create module with tool
    class AgentSignature(dspy.Signature):
        """Answer using tools."""
        query: str = dspy.InputField()
        tool: dspy.Tool = dspy.InputField()
        answer: str = dspy.OutputField()

    class Agent(dspy.Module):
        def __init__(self):
            super().__init__()

            def search_web(query: str) -> str:
                return f"Search: {query}"

            self.tool = dspy.Tool(search_web, name="search", desc="Original desc")
            self.pred = dspy.Predict(AgentSignature)

        def forward(self, query):
            return self.pred(query=query, tool=self.tool)

    program = Agent()
    optimizer, trainset = create_gepa_optimizer_for_tool_detection()

    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Mock optimized candidate

    pred_key = f"{TOOL_MODULE_PREFIX}:pred"
    assert pred_key in captured_base_program, f"Expected '{pred_key}' to be detected"

    optimized_candidate = dict(captured_base_program)
    mock_optimized_tool_module(
        optimized_candidate=optimized_candidate,
        pred_key=pred_key,
        predictor_instruction="OPTIMIZED: Answer using tools",
        tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search description",
                "arg_desc": {"query": "OPTIMIZED: Search query param"}
            }
        }
    )

    # Build program with optimizations
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_detection,
        feedback_map={},
        enable_tool_optimization=True,
    )
    rebuilt_program = adapter.build_program(optimized_candidate)

    # Verify predictor instruction was updated
    assert rebuilt_program.pred.signature.instructions == "OPTIMIZED: Answer using tools"

    # Verify tool description was updated
    assert rebuilt_program.tool.desc == "OPTIMIZED: Search description"
    assert rebuilt_program.tool.args["query"]["description"] == "OPTIMIZED: Search query param"

    # Verify original program unchanged
    assert program.pred.signature.instructions != "OPTIMIZED: Answer using tools"
    assert program.tool.desc == "Original desc"
