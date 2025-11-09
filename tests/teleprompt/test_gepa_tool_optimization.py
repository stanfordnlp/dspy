"""Tests for GEPA's generic tool optimization (beyond ReAct modules).

Tests the new generic tool optimization pathway that detects and optimizes
ANY dspy.Module using dspy.Tool, not just dspy.ReAct modules.

What we test:
1. Detection: Identify predictors with Tool-typed input fields
2. Extraction: Capture tool metadata from traces
3. Optimization: Route to ReActModuleProposer for joint predictor+tool optimization
4. Reconstruction: Apply optimized tool descriptions via traversal

Requirements:
- Signatures MUST use class-based definitions with type annotations
- String signatures like "query, tools -> answer" are NOT supported (lose type info)
- Detection is based on INPUT types only (output types don't matter)
"""

import json

import dspy
from dspy import Example
from dspy.utils.dummies import DummyLM


def setup_capture_for_base_program(monkeypatch):
    """Capture base_program passed to gepa.optimize."""
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
    
    Tests that GEPA detects a custom module with a single tool and captures:
    - Predictor instruction
    - Tool name, description, and arg descriptions
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

    # Assert predictor detected with tool config (JSON, not plain string)
    assert "pred" in captured_base_program, "Expected 'pred' to be detected"

    pred_config = captured_base_program["pred"]
    config = json.loads(pred_config)  # Will fail if not JSON

    # Should have predictor instruction
    assert "predictor" in config, "Should have predictor instruction"
    assert isinstance(config["predictor"], str), "Predictor should be string"

    # Should have tool config
    assert "tools" in config, "Should have tools"
    assert "search" in config["tools"], "Should have search tool"

    tool = config["tools"]["search"]
    assert "desc" in tool, "Tool should have desc"
    assert tool["desc"] == "Search tool", f"Tool desc should match, got: {tool['desc']}"
    assert "arg_desc" in tool, "Tool should have arg_desc"


def test_detect_tool_list(monkeypatch):
    """Detect predictor with list of Tools.
    
    Tests that GEPA detects multiple tools and preserves ordering.
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

    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Assert predictor detected with both tools
    assert "pred" in captured_base_program

    pred_config = captured_base_program["pred"]
    config = json.loads(pred_config)

    assert "tools" in config
    assert "search" in config["tools"]
    assert "calc" in config["tools"]

    # Verify tool descriptions
    assert config["tools"]["search"]["desc"] == "Search tool"
    assert config["tools"]["calc"]["desc"] == "Calculator tool"


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

    # Assert predictor detected as plain string (not JSON with tools)
    assert "pred" in captured_base_program

    pred_config = captured_base_program["pred"]
    assert isinstance(pred_config, str), "Should be string instruction"

    # Plain predictors get string instructions, not JSON
    # This is the current behavior - will stay the same after implementation


def test_update_tool_and_predictor(monkeypatch):
    """Rebuild program with updated tool descriptions and predictor instructions.
    
    Tests that DspyAdapter.build_program applies optimized tool metadata.
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

    # Mock optimized candidate with updated tool metadata
    optimized_candidate = dict(captured_base_program)

    # Assuming JSON format (will fail until implemented)
    pred_config = json.loads(optimized_candidate["pred"])
    pred_config["predictor"] = "OPTIMIZED: Answer using tools"
    pred_config["tools"]["search"]["desc"] = "OPTIMIZED: Search description"
    pred_config["tools"]["search"]["arg_desc"] = {"query": "OPTIMIZED: Search query param"}
    optimized_candidate["pred"] = json.dumps(pred_config)

    # Build program with optimizations
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_detection,
        feedback_map={},
        enable_tool_optimization=True,
    )
    rebuilt_program = adapter.build_program(optimized_candidate)

    # Assert predictor instruction updated
    assert rebuilt_program.pred.signature.instructions == "OPTIMIZED: Answer using tools"

    # Assert tool description updated
    assert rebuilt_program.tool.desc == "OPTIMIZED: Search description"
    assert rebuilt_program.tool.args["query"]["description"] == "OPTIMIZED: Search query param"

    # Verify original unchanged
    assert program.pred.signature.instructions != "OPTIMIZED: Answer using tools"
    assert program.tool.desc == "Original desc"
