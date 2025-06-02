import re

import dspy


def test_metadata():
    assert dspy.__name__ == "dspy"
    assert re.match(r"\d+\.\d+\.\d+", dspy.__version__)
    assert dspy.__author__ == "Omar Khattab"
    assert dspy.__author_email__ == "okhattab@stanford.edu"
    assert dspy.__url__ == "https://github.com/stanfordnlp/dspy"
    assert dspy.__description__ == "DSPy"
