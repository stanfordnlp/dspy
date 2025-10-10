import dspy
from dspy import Example
from dspy.utils.dummies import DummyLM


def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception:
        return "Error"


def search(query: str) -> str:
    return f"Search results for: {query}"


def simple_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    score = 1.0 if example.answer in str(prediction.answer) else 0.0
    return dspy.Prediction(score=score, feedback="Correct" if score == 1.0 else "Wrong")


def test_build_program_applies_tool_descriptions():
    """Test that build_program applies tool descriptions from candidate dict."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    calc_tool = dspy.Tool(calculator, name="calculator", desc="Old description")
    react = dspy.ReAct("question -> answer", tools=[calc_tool])

    adapter = DspyAdapter(
        student_module=react,
        metric_fn=simple_metric,
        feedback_map={},
        failure_score=0.0,
        optimize_tool_descriptions=True,
    )

    candidate = {
        "react": "New instruction for ReAct",
        "tool:calculator": "Optimized calculator description",
    }

    new_prog = adapter.build_program(candidate)

    assert new_prog.react.signature.instructions == "New instruction for ReAct"
    assert new_prog.tools["calculator"].desc == "Optimized calculator description"


def test_gepa_with_tool_optimization_enabled():
    """Test GEPA end-to-end with optimize_tool_descriptions=True."""
    calc_tool = dspy.Tool(calculator, name="calculator", desc="Does math")
    react = dspy.ReAct("question -> answer", tools=[calc_tool])

    lm = DummyLM(
        [
            {"next_thought": "Calculate", "next_tool_name": "calculator", "next_tool_args": {"expression": "2+2"}},
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Used calculator", "answer": "4"},
        ]
    )
    reflection_lm = DummyLM([{"improved_instruction": "Better"}])

    dspy.settings.configure(lm=lm)

    optimizer = dspy.GEPA(
        metric=simple_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=3,
        optimize_tool_descriptions=True,
    )

    trainset = [Example(question="What is 2+2?", answer="4").with_inputs("question")]

    optimized = optimizer.compile(react, trainset=trainset)

    assert optimized is not None
    assert hasattr(optimized, "tools")
    assert "calculator" in optimized.tools


def test_gepa_with_multi_agent_architecture():
    """Test that tool optimization discovers tools from nested subagent modules."""

    class MultiAgentSystem(dspy.Module):
        def __init__(self):
            super().__init__()
            # Subagent as module attribute (reuse existing search function)
            search_tool = dspy.Tool(search, name="search", desc="Searches")
            self.subagent = dspy.ReAct("task -> result", tools=[search_tool])

            # Main agent with subagent wrapped as tool
            def spawn_subagent(task: str) -> str:
                return self.subagent(task=task).result

            spawn_tool = dspy.Tool(spawn_subagent, name="spawn_subagent", desc="Spawns subagent")
            calc_tool = dspy.Tool(calculator, name="calculator", desc="Does math")
            self.main_agent = dspy.ReAct("q -> a", tools=[spawn_tool, calc_tool])

    system = MultiAgentSystem()

    # Test extraction using named_sub_modules pattern
    tool_descriptions = {}
    for _, module in system.named_sub_modules():
        if hasattr(module, "tools"):
            for tool_name, tool in module.tools.items():
                tool_key = f"tool:{tool_name}"
                if tool_key not in tool_descriptions:
                    tool_descriptions[tool_key] = tool.desc

    # All tools from all nested agents should be discovered
    assert "tool:calculator" in tool_descriptions
    assert "tool:spawn_subagent" in tool_descriptions
    assert "tool:search" in tool_descriptions
    assert "tool:finish" in tool_descriptions


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
