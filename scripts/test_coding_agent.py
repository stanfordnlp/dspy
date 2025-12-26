import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

interpreter = dspy.PythonInterpreter()

def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    return interpreter.execute(code)

agent = dspy.ReAct("request: str -> result: str", tools=[run_python])

try:
    result = agent(request="Calculate 2+2 and give me the answer")
    print(f"Result: {result.result}")
    print(f"Trajectory: {result.trajectory}")
finally:
    interpreter.shutdown()
