import os

import pytest

from tests.reliability.generate.utils import load_generated_cases, run_generated_case

_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.reliability
@pytest.mark.parametrize(
    "generated_case",
    load_generated_cases(_DIR_PATH),
    ids=lambda case: case.name,
)
def test_generated_cases(generated_case):
    run_generated_case(generated_case)
