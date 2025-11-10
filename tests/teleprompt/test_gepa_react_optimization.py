"""Tests for GEPA's unified ReAct module optimization with full path preservation.

Tests the critical bug fix where ReAct module paths must be preserved in full
(e.g., "multi_agent.orchestrator") instead of being truncated (e.g., "multi_agent").
This ensures correct module identification in multi-agent systems.

What we test:
1. Detection: GEPA correctly identifies ReAct modules with full paths
2. Reconstruction: build_program applies optimizations using full paths
3. Reflective dataset: make_reflective_dataset captures complete trajectories

Bug fixed: Path truncation in gepa.py and gepa_utils.py caused:
- Wrong module detection in nested structures
- Incorrect trajectory capture in multi-agent systems
- Optimization applied to wrong modules
"""

import json

import gepa
from gepa import optimize as gepa_optimize

import dspy
from dspy import Example
from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX, DspyAdapter
from dspy.utils.dummies import DummyLM


def setup_capture_for_base_program(monkeypatch):
    """Capture base_program passed to gepa.optimize."""
    captured_base_program = {}

    def capture_optimize(seed_candidate, **kwargs):
        captured_base_program.update(seed_candidate)
        return gepa_optimize(seed_candidate=seed_candidate, **kwargs)

    monkeypatch.setattr(gepa, "optimize", capture_optimize)

    return captured_base_program


def simple_metric_for_detection(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Simple metric for GEPA detection tests."""
    return dspy.Prediction(score=0.5, feedback="ok")


def get_predictor_name(program, predictor_obj):
    """Get predictor name by finding it via object identity in named_predictors().
    
    Args:
        program: DSPy program
        predictor_obj: The predictor object to find (e.g., program.react_module)
    
    Returns:
        str: Predictor name (e.g., "react_module", "agent.react", etc.)
    """
    for name, pred in program.named_predictors():
        if pred is predictor_obj:
            return name
    raise ValueError(f"Predictor not found in program: {predictor_obj}")


def simple_metric_for_reconstruction(example, pred, trace=None):
    """Simple metric for adapter reconstruction tests."""
    return 0.5


def simple_feedback(*args, **kwargs):
    """Generic feedback function for reflective dataset tests."""
    return {"score": 1.0, "feedback": "Good"}


def create_gepa_optimizer_for_detection():
    """Create GEPA optimizer with standard test configuration."""
    task_lm = DummyLM([
        {"next_thought": "I should use a tool", "next_tool_name": "search", "next_tool_args": {"query": "test"}},
        {"next_thought": "I have enough information", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "Based on the tool results", "answer": "test answer"},
    ] * 20)

    reflection_lm = DummyLM([
        {"improved_instruction": "optimized instruction"},
        {"react": "optimized react", "extract": "optimized extract", "tools": None},  # For ReActModuleProposer
    ] * 20)

    dspy.settings.configure(lm=task_lm)

    optimizer = dspy.GEPA(
        metric=simple_metric_for_detection,
        reflection_lm=reflection_lm,
        max_metric_calls=2,
        enable_tool_optimization=True,
    )

    trainset = [Example(question="test", answer="test").with_inputs("question")]

    return optimizer, trainset


def assert_react_module_detected(captured_base_program, predictor_name, expected_tools):
    """Assert that a ReAct module was detected with all components.
    
    Args:
        predictor_name: Name of extract.predict from named_predictors() 
                       (e.g., "extract.predict", "workflow.coordinator.extract.predict")
    """
    module_key = f"{REACT_MODULE_PREFIX}:{predictor_name}"

    assert module_key in captured_base_program, f"Expected '{module_key}' to be detected"

    config = json.loads(captured_base_program[module_key])

    # Check structure: should have predictor instructions and tools
    assert "tools" in config, f"{module_key} should have tools"

    for tool_name, expected_desc in expected_tools.items():
        assert tool_name in config["tools"], f"{module_key} should have '{tool_name}' tool"
        tool = config["tools"][tool_name]
        assert "desc" in tool, f"{tool_name} should have desc"
        assert tool["desc"] == expected_desc, f"{tool_name} desc should match"
        assert "arg_desc" in tool, f"{tool_name} should have arg_desc"

    return config


def assert_regular_module_detected(captured_base_program, module_key):
    """Assert that a non-ReAct module was detected."""
    assert module_key in captured_base_program, f"Expected '{module_key}' to be detected"
    instruction = captured_base_program[module_key]
    assert isinstance(instruction, str), f"{module_key} should be string instruction, not JSON"
    return instruction


def assert_react_module_updated(react_module, expected_react_instruction, expected_extract_instruction, expected_tool_descriptions):
    """Assert that a ReAct module was properly updated with optimized instructions.

    Args:
        react_module: The ReAct module instance to check
        expected_react_instruction: Expected react instruction text
        expected_extract_instruction: Expected extract instruction text
        expected_tool_descriptions: Dict of {tool_name: {"desc": desc, "arg_desc": {arg: desc}}}
    """
    assert react_module.react.signature.instructions == expected_react_instruction, \
        f"React instruction mismatch: got {react_module.react.signature.instructions}"

    assert react_module.extract.predict.signature.instructions == expected_extract_instruction, \
        f"Extract instruction mismatch: got {react_module.extract.predict.signature.instructions}"

    for tool_name, tool_desc in expected_tool_descriptions.items():
        tool = react_module.tools[tool_name]

        if "desc" in tool_desc:
            assert tool.desc == tool_desc["desc"], \
                f"Tool '{tool_name}' desc mismatch: got {tool.desc}"

        if "arg_desc" in tool_desc:
            for arg_name, expected_arg_desc in tool_desc["arg_desc"].items():
                # Verify arg_desc propagated to tool.args (rendered in prompts)
                assert arg_name in tool.args, \
                    f"Tool '{tool_name}' arg_desc has '{arg_name}' but args schema doesn't"
                assert tool.args[arg_name].get("description") == expected_arg_desc, \
                    f"Tool '{tool_name}' args['{arg_name}']['description'] should match arg_desc (got {tool.args[arg_name].get('description')!r}, expected {expected_arg_desc!r})"


def assert_regular_module_updated(predictor, expected_instruction):
    """Assert that a regular (non-ReAct) predictor was updated with optimized instruction."""
    assert predictor.signature.instructions == expected_instruction, \
        f"Instruction mismatch: expected '{expected_instruction}', got '{predictor.signature.instructions}'"


def mock_optimized_react_module(program, optimized_candidate, react_instruction, extract_instruction, tool_descriptions, react_module=None):
    """Helper to mock an optimized ReAct module in the candidate dict.

    Args:
        program: The DSPy program (to find predictor names)
        optimized_candidate: The candidate dict to modify
        react_instruction: New react instruction
        extract_instruction: New extract instruction
        tool_descriptions: Dict of {tool_name: {"desc": desc, "arg_desc": {arg: desc}}}
        react_module: Optional specific ReAct module to update (for multi-module programs)
    """
    # Find the ReAct module's predictors via object identity
    if react_module is None:
        react_module = program if isinstance(program, dspy.ReAct) else None
        if not react_module:
            for _, module in program.named_sub_modules():
                if isinstance(module, dspy.ReAct):
                    react_module = module
                    break

        if not react_module:
            raise ValueError("No ReAct module found in program")

    # Get predictor names dynamically
    expected_react_name = get_predictor_name(program, react_module.react)
    expected_extract_name = get_predictor_name(program, react_module.extract.predict)

    module_key = f"{REACT_MODULE_PREFIX}:{expected_extract_name}"
    config = json.loads(optimized_candidate[module_key])

    # Update instructions using actual predictor names
    config[expected_react_name] = react_instruction
    config[expected_extract_name] = extract_instruction

    for tool_name, tool_desc in tool_descriptions.items():
        if "desc" in tool_desc:
            config["tools"][tool_name]["desc"] = tool_desc["desc"]
        if "arg_desc" in tool_desc:
            config["tools"][tool_name]["arg_desc"] = tool_desc["arg_desc"]

    optimized_candidate[module_key] = json.dumps(config)


def create_single_react_program():
    """Create a simple single ReAct module program."""
    def search_tool(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    def calculate_tool(expr: str) -> str:
        """Calculate math expression."""
        return "42"

    return dspy.ReAct(
        "question -> answer",
        tools=[
            dspy.Tool(search_tool, name="search", desc="Search the web"),
            dspy.Tool(calculate_tool, name="calc", desc="Calculate math"),
        ],
        max_iters=3
    )


def create_multi_react_workflow_program():
    """Create a mixed workflow program with 2 ReAct + 1 ChainOfThought."""
    class ResearchWorkflow(dspy.Module):
        def __init__(self):
            super().__init__()

            def search_papers(query: str) -> str:
                return f"Papers: {query}"

            def analyze_data(data: str) -> str:
                return f"Analysis: {data}"

            self.coordinator = dspy.ReAct(
                "task -> plan",
                tools=[dspy.Tool(search_papers, name="search", desc="Search tool")],
                max_iters=2
            )

            self.researcher = dspy.ReAct(
                "plan -> findings",
                tools=[dspy.Tool(analyze_data, name="analyze", desc="Analysis tool")],
                max_iters=2
            )

            self.summarizer = dspy.ChainOfThought("findings -> summary")

        def forward(self, question):
            plan = self.coordinator(task=question)
            findings = self.researcher(plan=plan.plan)
            summary = self.summarizer(findings=findings.findings)
            return dspy.Prediction(answer=summary.summary)

    class MixedWorkflowSystem(dspy.Module):
        def __init__(self):
            super().__init__()
            self.workflow = ResearchWorkflow()

        def forward(self, question):
            return self.workflow(question=question)

    return MixedWorkflowSystem()


def create_orchestrator_with_workers_program():
    """Create orchestrator with 2 worker ReAct modules as tools."""
    class OrchestratorWorkerSystem(dspy.Module):
        def __init__(self):
            super().__init__()

            def search_web(query: str) -> str:
                return f"Search results: {query}"

            def analyze_data(data: str) -> str:
                return f"Analysis: {data}"

            def research_topic(topic: str) -> str:
                return f"Research: {topic}"

            self.analyst = dspy.ReAct(
                "data -> analysis",
                tools=[dspy.Tool(analyze_data, name="analyze", desc="Analyze data")],
                max_iters=2
            )

            self.researcher = dspy.ReAct(
                "topic -> findings",
                tools=[dspy.Tool(research_topic, name="research", desc="Research topic")],
                max_iters=2
            )

            def use_analyst(data: str) -> str:
                result = self.analyst(data=data)
                return str(result.analysis) if hasattr(result, "analysis") else str(result)

            def use_researcher(topic: str) -> str:
                result = self.researcher(topic=topic)
                return str(result.findings) if hasattr(result, "findings") else str(result)

            self.orchestrator = dspy.ReAct(
                "question -> answer",
                tools=[
                    dspy.Tool(search_web, name="search", desc="Search tool"),
                    dspy.Tool(use_analyst, name="analyst", desc="Use analyst"),
                    dspy.Tool(use_researcher, name="researcher", desc="Use researcher"),
                ],
                max_iters=3
            )

        def forward(self, question):
            result = self.orchestrator(question=question)
            return dspy.Prediction(answer=result.answer)

    class MultiAgentSystem(dspy.Module):
        def __init__(self):
            super().__init__()
            self.multi_agent = OrchestratorWorkerSystem()

        def forward(self, question):
            return self.multi_agent(question=question)

    return MultiAgentSystem()


def test_single_react_module_detection(monkeypatch):
    """Test GEPA detects a single top-level ReAct module with all components.

    Tests:
    - ReAct module detected as REACT_MODULE_PREFIX (no path suffix)
    - react instruction captured
    - extract instruction captured
    - All tools with descriptions captured
    """

    captured_base_program = setup_capture_for_base_program(monkeypatch)
    program = create_single_react_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    # DummyLM now properly configured - compile should succeed
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Get predictor name dynamically via object identity
    expected_predictor_name = get_predictor_name(program, program.extract.predict)

    assert_react_module_detected(
        captured_base_program=captured_base_program,
        predictor_name=expected_predictor_name,
        expected_tools={"search": "Search the web", "calc": "Calculate math"}
    )


def test_multi_react_workflow_detection(monkeypatch):
    """Test GEPA detects multiple ReAct modules with FULL paths preserved.

    PRIMARY BUG FIX TEST: Validates paths are NOT truncated.

    Tests:
    - workflow.coordinator detected as "react_module:workflow.coordinator" (NOT "react_module:workflow")
    - workflow.researcher detected as "react_module:workflow.researcher" (NOT "react_module:workflow")
    - Both ReAct modules detected separately (not merged)
    - Non-ReAct module (summarizer) detected correctly

    Before fix: Paths truncated at first dot → wrong module matching
    After fix: Full paths preserved → correct module identification
    """

    captured_base_program = setup_capture_for_base_program(monkeypatch)
    program = create_multi_react_workflow_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    # DummyLM now properly configured - compile should succeed
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Get predictor names dynamically via object identity
    expected_coordinator_name = get_predictor_name(program, program.workflow.coordinator.extract.predict)
    expected_researcher_name = get_predictor_name(program, program.workflow.researcher.extract.predict)
    expected_summarizer_name = get_predictor_name(program, program.workflow.summarizer.predict)

    assert f"{REACT_MODULE_PREFIX}:{expected_coordinator_name}" in captured_base_program
    assert f"{REACT_MODULE_PREFIX}:{expected_researcher_name}" in captured_base_program

    react_modules = [k for k in captured_base_program.keys() if k.startswith(REACT_MODULE_PREFIX)]
    assert len(react_modules) == 2, f"Expected 2 ReAct modules, got {len(react_modules)}"

    assert_react_module_detected(
        captured_base_program=captured_base_program,
        predictor_name=expected_coordinator_name,
        expected_tools={"search": "Search tool"}
    )
    assert_react_module_detected(
        captured_base_program=captured_base_program,
        predictor_name=expected_researcher_name,
        expected_tools={"analyze": "Analysis tool"}
    )
    assert_regular_module_detected(
        captured_base_program=captured_base_program,
        module_key=expected_summarizer_name
    )


def test_nested_react_orchestrator_worker_detection(monkeypatch):
    """Test GEPA detects nested multi-agent system with 3 separate ReAct modules.

    Tests complex nested structure:
    - Orchestrator: multi_agent.orchestrator (has analyst + researcher as tools)
    - Analyst worker: multi_agent.analyst (wrapped as tool for orchestrator)
    - Researcher worker: multi_agent.researcher (wrapped as tool for orchestrator)

    Validates:
    - All 3 ReAct modules detected with FULL paths
    - Each module has its own tools detected
    - No path truncation causes module merging
    """

    captured_base_program = setup_capture_for_base_program(monkeypatch)
    program = create_orchestrator_with_workers_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    # DummyLM now properly configured - compile should succeed
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Get predictor names dynamically via object identity
    expected_orchestrator_name = get_predictor_name(program, program.multi_agent.orchestrator.extract.predict)
    expected_analyst_name = get_predictor_name(program, program.multi_agent.analyst.extract.predict)
    expected_researcher_name = get_predictor_name(program, program.multi_agent.researcher.extract.predict)

    assert f"{REACT_MODULE_PREFIX}:{expected_orchestrator_name}" in captured_base_program
    assert f"{REACT_MODULE_PREFIX}:{expected_analyst_name}" in captured_base_program
    assert f"{REACT_MODULE_PREFIX}:{expected_researcher_name}" in captured_base_program

    react_modules = [k for k in captured_base_program.keys() if k.startswith(REACT_MODULE_PREFIX)]
    assert len(react_modules) == 3, f"Expected 3 ReAct modules, got {len(react_modules)}"

    assert_react_module_detected(
        captured_base_program=captured_base_program,
        predictor_name=expected_orchestrator_name,
        expected_tools={"search": "Search tool", "analyst": "Use analyst", "researcher": "Use researcher"}
    )
    assert_react_module_detected(
        captured_base_program=captured_base_program,
        predictor_name=expected_analyst_name,
        expected_tools={"analyze": "Analyze data"}
    )
    assert_react_module_detected(
        captured_base_program=captured_base_program,
        predictor_name=expected_researcher_name,
        expected_tools={"research": "Research topic"}
    )


def test_build_program_single_react(monkeypatch):
    """Test build_program applies optimizations to single top-level ReAct module."""

    captured_base_program = setup_capture_for_base_program(monkeypatch)
    program = create_single_react_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Mock optimized candidate
    optimized_candidate = dict(captured_base_program)
    mock_optimized_react_module(
        program=program,
        optimized_candidate=optimized_candidate,
        react_instruction="OPTIMIZED: React instruction",
        extract_instruction="OPTIMIZED: Extract instruction",
        tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search description",
                "arg_desc": {"query": "OPTIMIZED: Search query param"}
            },
            "calc": {
                "desc": "OPTIMIZED: Calc description",
                "arg_desc": {"expr": "OPTIMIZED: Math expression param"}
            }
        }
    )

    # Build program
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={},
        enable_tool_optimization=True
    )
    rebuilt_program = adapter.build_program(optimized_candidate)

    # Assert updates applied
    assert_react_module_updated(
        react_module=rebuilt_program,
        expected_react_instruction="OPTIMIZED: React instruction",
        expected_extract_instruction="OPTIMIZED: Extract instruction",
        expected_tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search description",
                "arg_desc": {"query": "OPTIMIZED: Search query param"}
            },
            "calc": {
                "desc": "OPTIMIZED: Calc description",
                "arg_desc": {"expr": "OPTIMIZED: Math expression param"}
            }
        }
    )

    # Verify original unchanged
    assert program.react.signature.instructions != "OPTIMIZED: React instruction"


def test_build_program_multi_react_workflow(monkeypatch):
    """Test build_program applies optimizations to mixed ReAct + non-ReAct workflow."""

    captured_base_program = setup_capture_for_base_program(monkeypatch)
    program = create_multi_react_workflow_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    # DummyLM now properly configured - compile should succeed
    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Mock optimized candidate
    optimized_candidate = dict(captured_base_program)

    mock_optimized_react_module(
        program=program,
        optimized_candidate=optimized_candidate,
        react_instruction="OPTIMIZED: Coordinator react",
        extract_instruction="OPTIMIZED: Coordinator extract",
        tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search tool",
                "arg_desc": {"query": "OPTIMIZED: Coordinator search query"}
            }
        },
        react_module=program.workflow.coordinator
    )

    mock_optimized_react_module(
        program=program,
        optimized_candidate=optimized_candidate,
        react_instruction="OPTIMIZED: Researcher react",
        extract_instruction="OPTIMIZED: Researcher extract",
        tool_descriptions={
            "analyze": {
                "desc": "OPTIMIZED: Analyze tool",
                "arg_desc": {"data": "OPTIMIZED: Data to analyze"}
            }
        },
        react_module=program.workflow.researcher
    )

    # Optimize summarizer (non-ReAct ChainOfThought)
    expected_summarizer_name = get_predictor_name(program, program.workflow.summarizer.predict)
    optimized_candidate[expected_summarizer_name] = "OPTIMIZED: Summarizer instruction"

    # Build program
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={},
        enable_tool_optimization=True
    )
    rebuilt_program = adapter.build_program(optimized_candidate)

    # Assert ReAct modules updated
    assert_react_module_updated(
        react_module=rebuilt_program.workflow.coordinator,
        expected_react_instruction="OPTIMIZED: Coordinator react",
        expected_extract_instruction="OPTIMIZED: Coordinator extract",
        expected_tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search tool",
                "arg_desc": {"query": "OPTIMIZED: Coordinator search query"}
            }
        }
    )

    assert_react_module_updated(
        react_module=rebuilt_program.workflow.researcher,
        expected_react_instruction="OPTIMIZED: Researcher react",
        expected_extract_instruction="OPTIMIZED: Researcher extract",
        expected_tool_descriptions={
            "analyze": {
                "desc": "OPTIMIZED: Analyze tool",
                "arg_desc": {"data": "OPTIMIZED: Data to analyze"}
            }
        }
    )

    # Assert non-ReAct module updated
    assert_regular_module_updated(
        predictor=rebuilt_program.workflow.summarizer.predict,
        expected_instruction="OPTIMIZED: Summarizer instruction"
    )

    # Verify original unchanged
    assert program.workflow.coordinator.react.signature.instructions != "OPTIMIZED: Coordinator react"


def test_build_program_orchestrator_with_workers(monkeypatch):
    """Test build_program applies optimizations to orchestrator with worker ReAct modules."""

    captured_base_program = setup_capture_for_base_program(monkeypatch)
    program = create_orchestrator_with_workers_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    optimizer.compile(program, trainset=trainset, valset=trainset)

    # Mock optimized candidate
    optimized_candidate = dict(captured_base_program)

    mock_optimized_react_module(
        program=program,
        optimized_candidate=optimized_candidate,
        react_instruction="OPTIMIZED: Orchestrator react",
        extract_instruction="OPTIMIZED: Orchestrator extract",
        tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search tool",
                "arg_desc": {"query": "OPTIMIZED: Query param"}
            }
        },
        react_module=program.multi_agent.orchestrator
    )

    mock_optimized_react_module(
        program=program,
        optimized_candidate=optimized_candidate,
        react_instruction="OPTIMIZED: Analyst react",
        extract_instruction="OPTIMIZED: Analyst extract",
        tool_descriptions={"analyze": {"desc": "OPTIMIZED: Analyze tool"}},
        react_module=program.multi_agent.analyst
    )

    mock_optimized_react_module(
        program=program,
        optimized_candidate=optimized_candidate,
        react_instruction="OPTIMIZED: Researcher react",
        extract_instruction="OPTIMIZED: Researcher extract",
        tool_descriptions={"research": {"desc": "OPTIMIZED: Research tool"}},
        react_module=program.multi_agent.researcher
    )

    # Build program
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={},
        enable_tool_optimization=True
    )
    rebuilt_program = adapter.build_program(optimized_candidate)

    # Assert all modules updated
    assert_react_module_updated(
        react_module=rebuilt_program.multi_agent.orchestrator,
        expected_react_instruction="OPTIMIZED: Orchestrator react",
        expected_extract_instruction="OPTIMIZED: Orchestrator extract",
        expected_tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search tool",
                "arg_desc": {"query": "OPTIMIZED: Query param"}
            }
        }
    )

    assert_react_module_updated(
        react_module=rebuilt_program.multi_agent.analyst,
        expected_react_instruction="OPTIMIZED: Analyst react",
        expected_extract_instruction="OPTIMIZED: Analyst extract",
        expected_tool_descriptions={"analyze": {"desc": "OPTIMIZED: Analyze tool"}}
    )

    assert_react_module_updated(
        react_module=rebuilt_program.multi_agent.researcher,
        expected_react_instruction="OPTIMIZED: Researcher react",
        expected_extract_instruction="OPTIMIZED: Researcher extract",
        expected_tool_descriptions={"research": {"desc": "OPTIMIZED: Research tool"}}
    )

    # Verify original unchanged
    assert program.multi_agent.orchestrator.react.signature.instructions != "OPTIMIZED: Orchestrator react"


def assert_reflective_example_has_trajectory(actual_example, expected_iterations, answer):
    """Assert reflective dataset captured complete trajectory without duplicates.

    Validates:
    - All iterations present (thought_0, thought_1, ..., thought_N)
    - No duplicate/extra iterations (no thought_(N+1))
    - Expected answer in outputs
    - Works for any signature (question→answer, data→analysis, etc.)

    Catches bugs:
    - Wrong predictor used (react vs extract.predict) → incomplete trajectory
    - Path truncation → wrong module's trajectory captured
    """
    # Should have the three main sections
    assert "Inputs" in actual_example
    assert "Generated Outputs" in actual_example
    assert "Feedback" in actual_example

    # Validate Inputs
    inputs = actual_example["Inputs"]
    # Don't assume "question" - could be "data", "topic", etc depending on module signature
    # Just check trajectory exists
    assert "trajectory" in inputs

    # Validate trajectory has expected structure and values
    trajectory_str = inputs["trajectory"]
    num_iterations = len(expected_iterations)

    # Check all expected thoughts are present
    for i, (thought, _tool_name, _tool_args) in enumerate(expected_iterations):
        assert thought in trajectory_str, f"Trajectory should contain thought_{i}: {thought}"
        assert f"thought_{i}" in trajectory_str
        assert f"tool_name_{i}" in trajectory_str
        assert f"observation_{i}" in trajectory_str

    # NO extra iterations (validates no duplicates)
    assert f"thought_{num_iterations}" not in trajectory_str, \
        f"Should not have duplicate iteration {num_iterations}"

    # Validate Generated Outputs contain the expected answer
    outputs = actual_example["Generated Outputs"]
    # Answer could be in "answer", "analysis", "findings", etc depending on module signature
    # Just check the expected answer value appears somewhere in the outputs
    output_str = str(outputs)
    assert answer in output_str, f"Expected answer '{answer}' not found in outputs: {outputs}"

    # Validate Feedback exists
    assert isinstance(actual_example["Feedback"], str)
    assert len(actual_example["Feedback"]) > 0


def test_make_reflective_dataset_single_react():
    """Test reflective dataset captures complete trajectory for single ReAct module."""

    program = create_single_react_program()

    expected_iterations = [
        ("I should search", "search", {"query": "test"}),
        ("Done", "finish", {})
    ]
    expected_answer = "result"

    lm = DummyLM([
        {"next_thought": "I should search", "next_tool_name": "search", "next_tool_args": {"query": "test"}},
        {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "Based on search", "answer": "result"},
    ] * 10)
    dspy.settings.configure(lm=lm)

    # Get predictor name dynamically
    expected_predictor_name = get_predictor_name(program, program.extract.predict)

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={expected_predictor_name: simple_feedback},
        enable_tool_optimization=True
    )

    trainset = [Example(question="test", answer="result").with_inputs("question")]
    eval_batch = adapter.evaluate(batch=trainset, candidate={}, capture_traces=True)

    result = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=[f"{REACT_MODULE_PREFIX}:{expected_predictor_name}"]
    )

    module_key = f"{REACT_MODULE_PREFIX}:{expected_predictor_name}"
    assert module_key in result
    examples = result[module_key]
    assert len(examples) == 1, f"Should have 1 reflective example, got {len(examples)}"

    assert_reflective_example_has_trajectory(
        actual_example=examples[0],
        expected_iterations=expected_iterations,
        answer=expected_answer
    )

def test_make_reflective_dataset_orchestrator_with_workers():
    """Test reflective dataset for multi-agent system with 3 ReAct modules.

    Tests full path preservation in complex nested system:
    - Orchestrator: multi_agent.orchestrator (3 iterations)
    - Analyst: multi_agent.analyst (2 iterations)
    - Researcher: multi_agent.researcher (2 iterations)

    Validates each module's trajectory captured separately with correct iteration counts.
    """

    program = create_orchestrator_with_workers_program()

    orchestrator_iterations = [
        ("Let me use the analyst", "analyst", {"data": "test"}),
        ("Now let me use the researcher", "researcher", {"topic": "test"}),
        ("Done", "finish", {})
    ]

    analyst_iterations = [
        ("Analyzing the data", "analyze", {"data": "test"}),
        ("Done", "finish", {})
    ]

    researcher_iterations = [
        ("Researching the topic", "research", {"topic": "test"}),
        ("Done", "finish", {})
    ]

    lm = DummyLM([
        {"next_thought": "Let me use the analyst", "next_tool_name": "analyst", "next_tool_args": {"data": "test"}},
        {"next_thought": "Analyzing the data", "next_tool_name": "analyze", "next_tool_args": {"data": "test"}},
        {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "Analysis complete", "analysis": "analyzed_data"},
        {"next_thought": "Now let me use the researcher", "next_tool_name": "researcher", "next_tool_args": {"topic": "test"}},
        {"next_thought": "Researching the topic", "next_tool_name": "research", "next_tool_args": {"topic": "test"}},
        {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "Research complete", "findings": "research_findings"},
        {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
        {"reasoning": "Orchestration complete", "answer": "result"},
    ] * 10)
    dspy.settings.configure(lm=lm)

    # Get predictor names dynamically
    expected_orch_name = get_predictor_name(program, program.multi_agent.orchestrator.extract.predict)
    expected_analyst_name = get_predictor_name(program, program.multi_agent.analyst.extract.predict)
    expected_researcher_name = get_predictor_name(program, program.multi_agent.researcher.extract.predict)

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={
            expected_orch_name: simple_feedback,
            expected_analyst_name: simple_feedback,
            expected_researcher_name: simple_feedback,
        },
        enable_tool_optimization=True
    )

    trainset = [Example(question="test", answer="result").with_inputs("question")]
    eval_batch = adapter.evaluate(batch=trainset, candidate={}, capture_traces=True)

    result = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=[
            f"{REACT_MODULE_PREFIX}:{expected_orch_name}",
            f"{REACT_MODULE_PREFIX}:{expected_analyst_name}",
            f"{REACT_MODULE_PREFIX}:{expected_researcher_name}"
        ]
    )

    orch_key = f"{REACT_MODULE_PREFIX}:{expected_orch_name}"
    analyst_key = f"{REACT_MODULE_PREFIX}:{expected_analyst_name}"
    researcher_key = f"{REACT_MODULE_PREFIX}:{expected_researcher_name}"

    # Verify all 3 modules captured
    assert len(result) == 3
    assert orch_key in result and len(result[orch_key]) == 1
    assert analyst_key in result and len(result[analyst_key]) == 1
    assert researcher_key in result and len(result[researcher_key]) == 1

    # Verify each module's trajectory captured correctly
    assert_reflective_example_has_trajectory(result[orch_key][0], orchestrator_iterations, "result")
    assert_reflective_example_has_trajectory(result[analyst_key][0], analyst_iterations, "analyzed_data")
    assert_reflective_example_has_trajectory(result[researcher_key][0], researcher_iterations, "research_findings")


