from dspy.primitives.python_interpreter import PythonInterpreter

def test_execute_simple_code():
    interpreter = PythonInterpreter()
    code = "print('Hello, World!')"
    result = interpreter.execute(code)
    assert result == 'Hello, World!\n', "Simple print statement should return 'Hello World!\n'"

def test_import():
    interpreter = PythonInterpreter()
    code = "import math\nresult = math.sqrt(4)\nresult"
    result = interpreter.execute(code)
    assert result == 2, "Should be able to import and use math.sqrt"

def test_user_variable_definitions():
    interpreter = PythonInterpreter()
    code = "result = number + 1\nresult"
    result = interpreter.execute(code, variables={'number': 4})
    assert result == 5, "User variable assignment should work"