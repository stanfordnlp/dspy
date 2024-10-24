import re
import asyncio
import multiprocessing
import concurrent.futures
from typing import Optional
import os
import sys
import traceback

"""
Note that this code is largely based off of the code here:
https://github.com/HackerCupAI/starter-kits/blob/main/submit_first_solution/01_one_shot.py

by @tcapelle, with some adaptations for this workflow.
"""

def extract_code(code_str):
    # Regex pattern to extract the code between ```python and ```
    pattern = r"```python\s*([\s\S]*?)\s*```"

    # Use re.search to find the code inside the code block
    match = re.search(pattern, code_str)

    if match:
        # Extract the matched group (the code part inside ```python ... ```)
        code = match.group(1).strip()
    else:
        # Fallback: Assume we still need to strip ``` and ```python if match fails
        code = code_str

    # Remove any leading/trailing ``` and ```python manually, in case the input doesn't match exactly
    code = re.sub(r"^```python\s*", "", code)  # Remove starting ```python
    code = re.sub(r"^```", "", code)  # Remove starting ```
    code = re.sub(r"```$", "", code)  # Remove ending ``

    return code.strip()




def run_with_timeout(code: str, input: Optional[str], timeout: int):
    def target_fn(input, return_dict):
        vars = {}
        try:
            # Redirect stdout to silence print statements
            sys.stdout = open(os.devnull, 'w')
            exec(code, vars)
            fn = vars.get("solve", lambda x: x)
            return_dict["result"] = fn(input)
        except Exception as e:
            return_dict["error"] = str(e)
            return_dict["stack_trace"] = traceback.format_exc()
        finally:
            # Restore stdout to its original setting
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    process = multiprocessing.Process(target=target_fn, args=(input, return_dict))
    process.start()
    process.join(timeout)  # Wait for the process to finish or timeout

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "error": f"Execution exceeded the timeout of {timeout} seconds",
            "result": None,
            "stack_trace": None,
        }

    if "error" in return_dict:
        # Return the error and the stack trace for feedback
        return {
            "error": return_dict["error"],
            "result": None,
            "stack_trace": return_dict.get("stack_trace"),
        }

    return {"result": return_dict.get("result"), "error": None, "stack_trace": None}

async def arun(
    code: Optional[str] = None, input: Optional[str] = None, timeout: int = 60
):
    loop = asyncio.get_running_loop()
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, run_with_timeout, code, input, timeout
            )
            result_dict = await asyncio.wait_for(future, timeout=timeout)
        return result_dict
    except asyncio.TimeoutError:
        return {
            "error": f"Function call timed out after {timeout} seconds",
            "result": None,
            "stack_trace": "Function call timed out. Code needs to be more efficient.",
        }
    except Exception as e:
        return {"error": str(e), "result": None}


# Function to run code synchronously
def run(code: Optional[str] = None, input: Optional[str] = None, timeout: int = 5):
    return asyncio.run(arun(code, input, timeout))

# Function to check the solution
def check_solution(expected: str, actual: str) -> dict:
    "Check the solution against the expected output"
    matches = 0
    expected_lines = expected.strip().split("\n")
    actual_lines = actual.strip().split("\n")
    offending_cases = []
    for expected_line, actual_line in zip(expected_lines, actual_lines):
        expected_line = expected_line.strip()
        actual_line = actual_line.strip()

        if expected_line == actual_line:
            matches += 1  # +1 for the whole line match
        else:
            offending_cases.append((expected_line, actual_line))
    return {
        "matches": matches == len(expected_lines),
        "total": len(expected_lines),
        "offending_cases": offending_cases,
    }