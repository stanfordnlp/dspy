"""Tests for GEPA's unified ReAct module optimization.

This tests the new architecture where ReAct modules are optimized as a single
unit (react instruction + extract instruction + tool descriptions together).
"""

import hashlib
import json

import dspy
from dspy import Example

# Load fixture
with open("tests/teleprompt/gepa_dummy_lm_react_opt.json") as f:
    FIXTURE = json.load(f)


def stable_hash(obj):
    """Create a stable hash that works across Python processes.
    
    Python's built-in hash() is randomized per process (PYTHONHASHSEED),
    so we use SHA256 for deterministic hashing.
    """
    return hashlib.sha256(repr(obj).encode()).hexdigest()


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
        optimize_tool_descriptions=True,
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
