"""Tests for GEPA's unified ReAct module optimization.

This tests the new architecture where ReAct modules are optimized as a single
unit (react instruction + extract instruction + tool descriptions together).

NOTE: This test is currently skipped because hash-based fixtures are fragile
across Python versions due to prompt formatting changes.
"""

import hashlib
import json

import pytest

import dspy
from dspy import Example
from dspy.utils.dummies import DummyLM

# Load fixture
with open("tests/teleprompt/gepa_dummy_lm_react_opt.json") as f:
    FIXTURE = json.load(f)


def stable_hash(obj):
    """Create a stable hash that works across Python versions.
    
    Uses JSON serialization with sorted keys for truly stable hashing
    across Python versions. This avoids repr() formatting differences
    and dict ordering issues that can occur between Python versions.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class DictDummyLM(dspy.clients.lm.LM):
    """DummyLM that replays from fixture using stable hashing.
    
    Uses SHA256 instead of Python's built-in hash() to ensure deterministic
    hashing across different Python processes (avoids PYTHONHASHSEED issues).
    """

    def __init__(self, history):
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.history = {}
        # Use stable hash instead of Python's randomized hash()
        for m in history:
            self.history[stable_hash(m["messages"])] = m

    def __call__(self, prompt=None, messages=None, **kwargs):
        key = stable_hash(messages)
        if key not in self.history:
            raise AssertionError(
                "Message not found in fixture. "
                "This usually means the test code doesn't match regenerate_fixture.py exactly. "
                "Check: program structure, metric function, trainset examples."
            )
        return self.history[key]["outputs"]


# Tool definitions (must match regenerate_fixture.py)
EMPLOYEE_DEPARTMENTS = {
    "Alice": "Red",
    "Bob": "Blue",
    "Charlie": "Green",
}

DEPARTMENT_BUDGETS = {
    "Red": "10",
    "Blue": "20",
    "Green": "30",
}

EMPLOYEE_SALARIES = {
    "Alice": "1",
    "Bob": "2",
    "Charlie": "3",
}


def get_employee_department(arg: str) -> str:
    """Get employee's department."""
    return EMPLOYEE_DEPARTMENTS.get(
        arg,
        "Not found. This tool accepts an employee's first name only (e.g., 'Alice', 'Bob', or 'Charlie'), not full queries."
    )


def get_department_budget(arg: str) -> str:
    """Get department's budget."""
    return DEPARTMENT_BUDGETS.get(
        arg,
        "Not found. This tool accepts a department name only (e.g., 'Red', 'Blue', or 'Green'), not full queries."
    )


def get_employee_salary(arg: str) -> str:
    """Get employee's salary."""
    return EMPLOYEE_SALARIES.get(
        arg,
        "Not found. This tool accepts an employee's first name only (e.g., 'Alice', 'Bob', or 'Charlie'), not full queries."
    )


@pytest.mark.skip(reason="Hash-based fixtures break across Python versions - see file docstring")
def test_gepa_optimizes_react_module():
    """Test that GEPA optimizes ReAct module (react + extract + tools)."""

    lm = DictDummyLM(FIXTURE["lm"])
    reflection_lm = DictDummyLM(FIXTURE["reflection_lm"])
    dspy.settings.configure(lm=lm)

    dept_tool = dspy.Tool(get_employee_department, name="toolA", desc="Tool A")
    budget_tool = dspy.Tool(get_department_budget, name="toolB", desc="Tool B")
    salary_tool = dspy.Tool(get_employee_salary, name="toolC", desc="Tool C")

    program = dspy.ReAct(
        "question -> answer",
        tools=[dept_tool, budget_tool, salary_tool],
        max_iters=5
    )

    # Store baseline descriptions
    baseline_react = program.react.signature.instructions
    baseline_extract = program.extract.predict.signature.instructions
    baseline_toolA = program.tools["toolA"].desc
    baseline_toolB = program.tools["toolB"].desc
    baseline_toolC = program.tools["toolC"].desc
    baseline_toolA_arg_desc = program.tools["toolA"].arg_desc
    baseline_toolB_arg_desc = program.tools["toolB"].arg_desc
    baseline_toolC_arg_desc = program.tools["toolC"].arg_desc
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        pred_str = str(getattr(prediction, "answer", prediction)).strip()
        expected = str(example.answer).strip()
        score = 1.0 if pred_str == expected else 0.0
        feedback = "Correct" if score == 1.0 else f"Wrong (got '{pred_str}', expected '{expected}')"
        return dspy.Prediction(score=score, feedback=feedback)

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=5,
        optimize_react_components=True,
    )

    trainset = [
        Example(
            question="What is the budget of Alice's department minus Charlie's salary?",
            answer="7",
        ).with_inputs("question"),
        Example(
            question="How much larger is the budget of Bob's department than Alice's salary?",
            answer="19",
        ).with_inputs("question"),
    ]

    optimized = optimizer.compile(program, trainset=trainset, valset=trainset)

    # Baseline and optimized instructions and descriptions should be different
    assert optimized.react.signature.instructions != baseline_react, \
        "ReAct instruction should be optimized by reflection LM"
    assert optimized.extract.predict.signature.instructions != baseline_extract, \
        "Extract instruction should be optimized by reflection LM"
    assert optimized.tools["toolA"].desc != baseline_toolA, \
        "toolA description should be optimized"
    assert optimized.tools["toolB"].desc != baseline_toolB, \
        "toolB description should be optimized"
    assert optimized.tools["toolC"].desc != baseline_toolC, \
        "toolC description should be optimized"
    assert optimized.tools["toolA"].arg_desc != baseline_toolA_arg_desc, \
        "toolA argument description should be optimized"
    assert optimized.tools["toolB"].arg_desc != baseline_toolB_arg_desc, \
        "toolB argument description should be optimized"
    assert optimized.tools["toolC"].arg_desc != baseline_toolC_arg_desc, \
        "toolC argument description should be optimized"


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
                assert tool.arg_desc.get(arg_name) == expected_arg_desc, \
                    f"Tool '{tool_name}' arg '{arg_name}' desc mismatch"


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
                return str(result.analysis) if hasattr(result, 'analysis') else str(result)
            
            def use_researcher(topic: str) -> str:
                result = self.researcher(topic=topic)
                return str(result.findings) if hasattr(result, 'findings') else str(result)
            
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
    """Test GEPA detects a single top-level ReAct module."""
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
    """Test GEPA detects multiple ReAct modules (tests bug fix for path truncation)."""
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
    """Test GEPA detects orchestrator with 2 worker ReAct modules as tools."""
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
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX, DspyAdapter
    
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
            "search": {"desc": "OPTIMIZED: Search description"},
            "calc": {"desc": "OPTIMIZED: Calc description"}
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
            "search": {"desc": "OPTIMIZED: Search description"},
            "calc": {"desc": "OPTIMIZED: Calc description"}
        }
    )
    
    # Verify original unchanged
    assert program.react.signature.instructions != "OPTIMIZED: React instruction"


def test_build_program_multi_react_workflow(monkeypatch):
    """Test build_program applies optimizations to mixed ReAct + non-ReAct workflow."""
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX, DspyAdapter
    
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
        tool_descriptions={"search": {"desc": "OPTIMIZED: Search tool"}}
    )
    
    mock_optimized_react_module(
        optimized_candidate=optimized_candidate,
        module_path="workflow.researcher",
        react_instruction="OPTIMIZED: Researcher react",
        extract_instruction="OPTIMIZED: Researcher extract",
        tool_descriptions={"analyze": {"desc": "OPTIMIZED: Analyze tool"}}
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
        expected_tool_descriptions={"search": {"desc": "OPTIMIZED: Search tool"}}
    )
    
    assert_react_module_updated(
        react_module=rebuilt_program.workflow.researcher,
        expected_react_instruction="OPTIMIZED: Researcher react",
        expected_extract_instruction="OPTIMIZED: Researcher extract",
        expected_tool_descriptions={"analyze": {"desc": "OPTIMIZED: Analyze tool"}}
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
    from dspy.teleprompt.gepa.gepa_utils import REACT_MODULE_PREFIX, DspyAdapter
    
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
