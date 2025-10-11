import json

import dspy
from dspy import Example
from dspy.utils.dummies import DummyLM


class DictDummyLM(dspy.clients.lm.LM):
    """Dummy LM that replays prerecorded responses based on message hash."""
    
    def __init__(self, history):
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.history = {}
        for m in history:
            self.history[hash(repr(m["messages"]))] = m

    def __call__(self, prompt=None, messages=None, **kwargs):
        assert hash(repr(messages)) in self.history, f"Message {messages} not found in history"
        m = self.history[hash(repr(messages))]
        return m["outputs"]


# Simple multi-hop employee database tools (for main integration test)
def get_employee_department(employee_name: str) -> str:
    """Gets department."""
    employees = {
        "John Smith": "Engineering",
        "Mary Johnson": "Sales",
        "Bob Wilson": "HR",
    }
    return employees.get(employee_name, "Not found")


def get_department_budget(department: str) -> str:
    """Gets budget."""
    budgets = {
        "Engineering": "500000",
        "Sales": "300000",
        "HR": "200000",
    }
    return budgets.get(department, "Not found")


def get_employee_salary(employee_name: str) -> str:
    """Gets salary."""
    salaries = {
        "John Smith": "120000",
        "Mary Johnson": "95000",
        "Bob Wilson": "85000",
    }
    return salaries.get(employee_name, "Not found")


# Helper functions for other tests
def calculator(expression: str) -> str:
    """Calculator for math."""
    try:
        return str(eval(expression))
    except Exception:
        return "Error"


def search(query: str) -> str:
    """Search function."""
    return f"Results for: {query}"


def simple_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    pred_str = str(prediction.answer).strip()
    expected = str(example.answer).strip()
    score = 1.0 if pred_str == expected else 0.0
    return dspy.Prediction(score=score, feedback="Correct" if score == 1.0 else "Wrong")


def test_build_program_applies_tool_descriptions():
    """Test that build_program applies tool descriptions from candidate dict."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    dept_tool = dspy.Tool(get_employee_department, name="get_employee_department", desc="Gets department.")
    react = dspy.ReAct("question -> answer", tools=[dept_tool])

    adapter = DspyAdapter(
        student_module=react,
        metric_fn=simple_metric,
        feedback_map={},
        failure_score=0.0,
        optimize_tool_descriptions=True,
    )

    candidate = {
        "react": "New instruction for ReAct",
        "tool:get_employee_department": "Retrieves the department name for a given employee",
    }

    new_prog = adapter.build_program(candidate)

    assert new_prog.react.signature.instructions == "New instruction for ReAct"
    assert new_prog.tools["get_employee_department"].desc == "Retrieves the department name for a given employee"


def test_gepa_with_tool_optimization_enabled():
    """Test GEPA end-to-end with optimize_tool_descriptions=True using preloaded traces."""
    # Setup ReAct with minimal tool descriptions (as captured in traces)
    dept_tool = dspy.Tool(get_employee_department, name="get_employee_department", desc="Gets department.")
    budget_tool = dspy.Tool(get_department_budget, name="get_department_budget", desc="Gets budget.")
    salary_tool = dspy.Tool(get_employee_salary, name="get_employee_salary", desc="Gets salary.")
    
    react = dspy.ReAct("question -> answer", tools=[dept_tool, budget_tool, salary_tool])

    # Load prerecorded LM traces from real gpt-5-nano run
    with open("tests/teleprompt/gepa_dummy_lm_tool_optimization.json") as f:
        data = json.load(f)
    
    lm = DictDummyLM(data["lm"])
    reflection_lm = DictDummyLM(data["reflection_lm"])

    dspy.settings.configure(lm=lm)

    optimizer = dspy.GEPA(
        metric=simple_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=3,
        optimize_tool_descriptions=True,
    )

    # Use same examples as in trace generation
    trainset = [
        Example(question="What is the budget of John Smith's department?", answer="500000").with_inputs("question"),
        Example(question="How much does Mary Johnson earn?", answer="95000").with_inputs("question"),
        Example(question="What is Bob Wilson's department budget?", answer="200000").with_inputs("question"),
    ]

    optimized = optimizer.compile(react, trainset=trainset)

    # Verify optimization occurred
    assert optimized is not None
    assert hasattr(optimized, "tools")
    assert "get_employee_department" in optimized.tools
    assert "get_department_budget" in optimized.tools
    assert "get_employee_salary" in optimized.tools


def test_gepa_optimizes_multi_agent_system_end_to_end():
    """Test GEPA.compile() optimizes ALL tools from nested multi-agent system."""

    class MultiAgentSystem(dspy.Module):
        def __init__(self):
            super().__init__()
            search_tool = dspy.Tool(search, name="search", desc="Searches")
            self.subagent = dspy.ReAct("task -> result", tools=[search_tool])

            def spawn_subagent(task: str) -> str:
                return self.subagent(task=task).result

            spawn_tool = dspy.Tool(spawn_subagent, name="spawn_subagent", desc="Spawns subagent")
            calc_tool = dspy.Tool(calculator, name="calculator", desc="Does math")
            self.main_agent = dspy.ReAct("q -> a", tools=[spawn_tool, calc_tool])

        def forward(self, question):
            return self.main_agent(q=question)

    system = MultiAgentSystem()

    # Setup LMs
    lm = DummyLM([{"q": "question", "a": "answer"}])
    reflection_lm = DummyLM([{"improved_instruction": "Better"}])
    dspy.settings.configure(lm=lm)

    # Run GEPA optimization
    optimizer = dspy.GEPA(
        metric=simple_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=3,
        optimize_tool_descriptions=True,
    )

    trainset = [Example(question="test", answer="answer").with_inputs("question")]
    optimized = optimizer.compile(system, trainset=trainset)

    # Verify optimized system preserves structure with all tools
    assert "search" in optimized.subagent.tools
    assert "calculator" in optimized.main_agent.tools
    assert "spawn_subagent" in optimized.main_agent.tools


def test_adapter_routes_tools_and_signatures_separately():
    """Test that adapter routes tool components to ToolProposer."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
    
    calc_tool = dspy.Tool(calculator, name="calculator", desc="Original tool")
    agent = dspy.ReAct("question -> answer", tools=[calc_tool])
    
    # Provide reflection_lm with response for tool optimization
    reflection_lm = DummyLM([
        {"improved_tool_description": "Improved calculator tool"},
    ])
    
    adapter = DspyAdapter(
        student_module=agent,
        metric_fn=simple_metric,
        feedback_map={},
        failure_score=0.0,
        optimize_tool_descriptions=True,
        reflection_lm=reflection_lm,
    )
    
    # Verify routing function was created
    assert hasattr(adapter, 'propose_new_texts')
    
    # Test with ONLY tool components (signature optimization requires GEPA's LM interface)
    candidate = {
        "tool:calculator": "Original tool description",
    }
    
    reflective_dataset = {
        "tool:calculator": [{"Inputs": {"expr": "1+1"}, "Generated_Outputs": "2", "Feedback": "good"}],
    }
    
    # Call routing function - should route tool to ToolProposer
    result = adapter.propose_new_texts(candidate, reflective_dataset, ["tool:calculator"])
    
    # Verify tool is in result (routing worked)
    assert "tool:calculator" in result
    # Verify it was optimized
    assert result["tool:calculator"] == "Improved calculator tool"
