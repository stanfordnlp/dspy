import pytest
from dspy.primitives.python_interpreter import PythonInterpreter, TextPrompt, CodePrompt

def test_execute_simple_code():
    interpreter = PythonInterpreter(action_space={'print': print})
    code = "print('Hello, World!')"
    result = interpreter.execute(code)
    assert result is None, "Simple print statement should return None"

def test_action_space_limitation():
    def func(string):
        pass
    interpreter = PythonInterpreter(action_space={})
    code = "func('This should not execute')"
    with pytest.raises(Exception):
        interpreter.execute(code)

def test_import_whitelist():
    interpreter = PythonInterpreter(action_space={}, import_white_list=['math'])
    code = "import math\nresult = math.sqrt(4)"
    result = interpreter.execute(code)
    assert result == 2, "Should be able to import and use math.sqrt"

def test_fuzzy_variable_matching():
    interpreter = PythonInterpreter(action_space={})
    code = "result = number + 1"
    result = interpreter.execute(code, fuzz_state={'number': 4})
    assert result == 5, "Fuzzy variable matching should work"

def test_text_prompt_keyword_extraction():
    prompt = TextPrompt("Hello {name}, how are you?")
    assert 'name' in prompt.key_words, "Keyword 'name' should be extracted"

def test_text_prompt_formatting():
    prompt = TextPrompt("Hello {name}, how are you?")
    formatted = prompt.format(name="Alice")
    assert formatted == "Hello Alice, how are you?", "Should format with provided value"

def test_code_prompt_execution():
    action_space = {'len': len}
    interpreter = PythonInterpreter(action_space=action_space)
    code_prompt = CodePrompt("result = len('hello')")
    result, _ = code_prompt.execute(interpreter)
    assert result == 5, "Code execution should return the length of 'hello'"
