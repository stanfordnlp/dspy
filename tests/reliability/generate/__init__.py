import os
from typing import List, Optional

from tests.reliability.generate.utils import (
    GeneratedTestCase,
    generate_test_inputs,
    generate_test_program,
    load_generated_cases,
    load_generated_program,
)


def generate_test_cases(
    dst_path: str,
    num_inputs: int = 1,
    program_instructions: Optional[str] = None,
    input_instructions: Optional[str] = None,
) -> list[GeneratedTestCase]:
    os.makedirs(dst_path, exist_ok=True)
    if _directory_contains_program(dst_path):
        print(f"Found an existing test program at path {dst_path}. Generating new" f" test inputs for this program.")
    else:
        print("Generating a new test program and test inputs")
        generate_test_program(
            dst_path=dst_path,
            additional_instructions=program_instructions,
        )
    generate_test_inputs(
        dst_path=os.path.join(dst_path, "inputs"),
        program_path=os.path.join(dst_path, "program.py"),
        num_inputs=num_inputs,
        additional_instructions=input_instructions,
    )
    return load_generated_cases(dir_path=dst_path)


def _directory_contains_program(dir_path: str) -> bool:
    return any(file == "program.py" for file in os.listdir(dir_path))
