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

import dspy
from dspy import Example
from dspy.utils.dummies import DummyLM


def setup_spy_for_base_program(monkeypatch):
    """Setup spy to capture base_program from gepa.optimize."""
    captured_base_program = {}

    from gepa import optimize as original_optimize

    def spy_optimize(seed_candidate, **kwargs):
        captured_base_program.update(seed_candidate)
        return original_optimize(seed_candidate=seed_candidate, **kwargs)

    import gepa
    monkeypatch.setattr(gepa, "optimize", spy_optimize)

    return captured_base_program


def simple_metric_for_detection(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Simple metric for GEPA detection tests."""
    return dspy.Prediction(score=0.5, feedback="ok")


def simple_metric_for_reconstruction(example, pred, trace=None):
    """Simple metric for adapter reconstruction tests."""
    return 0.5


def simple_feedback(*args, **kwargs):
    """Generic feedback function for reflective dataset tests."""
    return {"score": 1.0, "feedback": "Good"}


def create_gepa_optimizer_for_detection():
    """Create GEPA optimizer with standard test configuration."""
    task_lm = DummyLM([{"answer": "test"}] * 10)
    reflection_lm = DummyLM([{"improved_instruction": "optimized"}] * 10)
    dspy.settings.configure(lm=task_lm)

    optimizer = dspy.GEPA(
        metric=simple_metric_for_detection,
        reflection_lm=reflection_lm,
        max_metric_calls=2,
        optimize_react_components=True,
    )

    trainset = [Example(question="test", answer="test").with_inputs("question")]

    return optimizer, trainset


def assert_react_module_detected(captured_base_program, module_path, expected_tools):
    """Assert that a ReAct module was detected with all components."""
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX

    module_key = REACT_MODULE_PREFIX if module_path == "" else f"{REACT_MODULE_PREFIX}:{module_path}"

    assert module_key in captured_base_program, f"Expected '{module_key}' to be detected"

    config = json.loads(captured_base_program[module_key])

    assert "react" in config, f"{module_key} should have react instruction"
    assert "extract" in config, f"{module_key} should have extract instruction"
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


def mock_optimized_react_module(optimized_candidate, module_path, react_instruction, extract_instruction, tool_descriptions):
    """Helper to mock an optimized ReAct module in the candidate dict.

    Args:
        optimized_candidate: The candidate dict to modify
        module_path: Module path (e.g., "multi_agent.orchestrator" or "" for top-level)
        react_instruction: New react instruction
        extract_instruction: New extract instruction
        tool_descriptions: Dict of {tool_name: {"desc": desc, "arg_desc": {arg: desc}}}
    """
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX

    module_key = REACT_MODULE_PREFIX if module_path == "" else f"{REACT_MODULE_PREFIX}:{module_path}"
    config = json.loads(optimized_candidate[module_key])
    config["react"] = react_instruction
    config["extract"] = extract_instruction

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
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX

    captured_base_program = setup_spy_for_base_program(monkeypatch)
    program = create_single_react_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    try:
        optimizer.compile(program, trainset=trainset, valset=trainset)
    except:
        pass

    module_key = REACT_MODULE_PREFIX
    assert module_key in captured_base_program, f"Expected '{module_key}' to be detected"

    assert_react_module_detected(
        captured_base_program=captured_base_program,
        module_path="",
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
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX

    captured_base_program = setup_spy_for_base_program(monkeypatch)
    program = create_multi_react_workflow_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    try:
        optimizer.compile(program, trainset=trainset, valset=trainset)
    except:
        pass

    assert f"{REACT_MODULE_PREFIX}:workflow.coordinator" in captured_base_program
    assert f"{REACT_MODULE_PREFIX}:workflow.researcher" in captured_base_program

    react_modules = [k for k in captured_base_program.keys() if k.startswith(REACT_MODULE_PREFIX)]
    assert len(react_modules) == 2, f"Expected 2 ReAct modules, got {len(react_modules)}"

    assert_react_module_detected(
        captured_base_program=captured_base_program,
        module_path="workflow.coordinator",
        expected_tools={"search": "Search tool"}
    )
    assert_react_module_detected(
        captured_base_program=captured_base_program,
        module_path="workflow.researcher",
        expected_tools={"analyze": "Analysis tool"}
    )
    assert_regular_module_detected(
        captured_base_program=captured_base_program,
        module_key="workflow.summarizer.predict"
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
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX

    captured_base_program = setup_spy_for_base_program(monkeypatch)
    program = create_orchestrator_with_workers_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    try:
        optimizer.compile(program, trainset=trainset, valset=trainset)
    except:
        pass

    assert f"{REACT_MODULE_PREFIX}:multi_agent.orchestrator" in captured_base_program
    assert f"{REACT_MODULE_PREFIX}:multi_agent.analyst" in captured_base_program
    assert f"{REACT_MODULE_PREFIX}:multi_agent.researcher" in captured_base_program

    react_modules = [k for k in captured_base_program.keys() if k.startswith(REACT_MODULE_PREFIX)]
    assert len(react_modules) == 3, f"Expected 3 ReAct modules, got {len(react_modules)}"

    assert_react_module_detected(
        captured_base_program=captured_base_program,
        module_path="multi_agent.orchestrator",
        expected_tools={"search": "Search tool", "analyst": "Use analyst", "researcher": "Use researcher"}
    )
    assert_react_module_detected(
        captured_base_program=captured_base_program,
        module_path="multi_agent.analyst",
        expected_tools={"analyze": "Analyze data"}
    )
    assert_react_module_detected(
        captured_base_program=captured_base_program,
        module_path="multi_agent.researcher",
        expected_tools={"research": "Research topic"}
    )


def test_build_program_single_react(monkeypatch):
    """Test build_program applies optimizations to single top-level ReAct module."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    captured_base_program = setup_spy_for_base_program(monkeypatch)
    program = create_single_react_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    try:
        optimizer.compile(program, trainset=trainset, valset=trainset)
    except:
        pass

    # Mock optimized candidate
    optimized_candidate = dict(captured_base_program)
    mock_optimized_react_module(
        optimized_candidate=optimized_candidate,
        module_path="",
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
        optimize_react_components=True
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
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    captured_base_program = setup_spy_for_base_program(monkeypatch)
    program = create_multi_react_workflow_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    try:
        optimizer.compile(program, trainset=trainset, valset=trainset)
    except:
        pass

    # Mock optimized candidate
    optimized_candidate = dict(captured_base_program)

    mock_optimized_react_module(
        optimized_candidate=optimized_candidate,
        module_path="workflow.coordinator",
        react_instruction="OPTIMIZED: Coordinator react",
        extract_instruction="OPTIMIZED: Coordinator extract",
        tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search tool",
                "arg_desc": {"query": "OPTIMIZED: Coordinator search query"}
            }
        }
    )

    mock_optimized_react_module(
        optimized_candidate=optimized_candidate,
        module_path="workflow.researcher",
        react_instruction="OPTIMIZED: Researcher react",
        extract_instruction="OPTIMIZED: Researcher extract",
        tool_descriptions={
            "analyze": {
                "desc": "OPTIMIZED: Analyze tool",
                "arg_desc": {"data": "OPTIMIZED: Data to analyze"}
            }
        }
    )

    # Optimize summarizer (non-ReAct ChainOfThought)
    optimized_candidate["workflow.summarizer.predict"] = "OPTIMIZED: Summarizer instruction"

    # Build program
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={},
        optimize_react_components=True
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
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    captured_base_program = setup_spy_for_base_program(monkeypatch)
    program = create_orchestrator_with_workers_program()

    optimizer, trainset = create_gepa_optimizer_for_detection()

    try:
        optimizer.compile(program, trainset=trainset, valset=trainset)
    except:
        pass

    # Mock optimized candidate
    optimized_candidate = dict(captured_base_program)

    mock_optimized_react_module(
        optimized_candidate=optimized_candidate,
        module_path="multi_agent.orchestrator",
        react_instruction="OPTIMIZED: Orchestrator react",
        extract_instruction="OPTIMIZED: Orchestrator extract",
        tool_descriptions={
            "search": {
                "desc": "OPTIMIZED: Search tool",
                "arg_desc": {"query": "OPTIMIZED: Query param"}
            }
        }
    )

    mock_optimized_react_module(
        optimized_candidate=optimized_candidate,
        module_path="multi_agent.analyst",
        react_instruction="OPTIMIZED: Analyst react",
        extract_instruction="OPTIMIZED: Analyst extract",
        tool_descriptions={"analyze": {"desc": "OPTIMIZED: Analyze tool"}}
    )

    mock_optimized_react_module(
        optimized_candidate=optimized_candidate,
        module_path="multi_agent.researcher",
        react_instruction="OPTIMIZED: Researcher react",
        extract_instruction="OPTIMIZED: Researcher extract",
        tool_descriptions={"research": {"desc": "OPTIMIZED: Research tool"}}
    )

    # Build program
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={},
        optimize_react_components=True
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
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX, DspyAdapter

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

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={"react": simple_feedback},
        optimize_react_components=True
    )

    trainset = [Example(question="test", answer="result").with_inputs("question")]
    eval_batch = adapter.evaluate(batch=trainset, candidate={}, capture_traces=True)

    result = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=[REACT_MODULE_PREFIX]
    )

    assert REACT_MODULE_PREFIX in result
    examples = result[REACT_MODULE_PREFIX]
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
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX, DspyAdapter

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

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=simple_metric_for_reconstruction,
        feedback_map={
            "multi_agent.orchestrator.react": simple_feedback,
            "multi_agent.analyst.react": simple_feedback,
            "multi_agent.researcher.react": simple_feedback,
        },
        optimize_react_components=True
    )

    trainset = [Example(question="test", answer="result").with_inputs("question")]
    eval_batch = adapter.evaluate(batch=trainset, candidate={}, capture_traces=True)

    result = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=[
            f"{REACT_MODULE_PREFIX}:multi_agent.orchestrator",
            f"{REACT_MODULE_PREFIX}:multi_agent.analyst",
            f"{REACT_MODULE_PREFIX}:multi_agent.researcher"
        ]
    )

    assert f"{REACT_MODULE_PREFIX}:multi_agent.orchestrator" in result
    assert f"{REACT_MODULE_PREFIX}:multi_agent.analyst" in result
    assert f"{REACT_MODULE_PREFIX}:multi_agent.researcher" in result
    assert len(result) == 3
    assert len(result[f"{REACT_MODULE_PREFIX}:multi_agent.orchestrator"]) == 1
    assert len(result[f"{REACT_MODULE_PREFIX}:multi_agent.analyst"]) == 1
    assert len(result[f"{REACT_MODULE_PREFIX}:multi_agent.researcher"]) == 1

    orch_example = result[f"{REACT_MODULE_PREFIX}:multi_agent.orchestrator"][0]
    assert_reflective_example_has_trajectory(orch_example, orchestrator_iterations, "result")
    assert "question" in orch_example["Inputs"]
    assert "answer" in orch_example["Generated Outputs"]
    assert "analyst" in orch_example["Inputs"]["trajectory"]

    analyst_example = result[f"{REACT_MODULE_PREFIX}:multi_agent.analyst"][0]
    assert_reflective_example_has_trajectory(analyst_example, analyst_iterations, "analyzed_data")
    assert "data" in analyst_example["Inputs"]
    assert "analysis" in analyst_example["Generated Outputs"]
    assert "Analysis:" in analyst_example["Inputs"]["trajectory"]

    researcher_example = result[f"{REACT_MODULE_PREFIX}:multi_agent.researcher"][0]
    assert_reflective_example_has_trajectory(researcher_example, researcher_iterations, "research_findings")
    assert "topic" in researcher_example["Inputs"]
    assert "findings" in researcher_example["Generated Outputs"]
    assert "Research:" in researcher_example["Inputs"]["trajectory"]


