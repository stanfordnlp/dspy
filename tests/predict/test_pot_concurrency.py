import json
from concurrent.futures import ThreadPoolExecutor

import dspy
from dspy.predict.program_of_thought import ProgramOfThought


class DummyLM(dspy.LM):
    def __init__(self):
        super().__init__("dummy")

    def __call__(self, prompt=None, **kwargs):
        # Return a valid JSON response for the adapter
        # This matches the structure ProgramOfThought expects
        return [
            json.dumps(
                {
                    "reasoning": "I will calculate the square.",
                    "generated_code": "```python\nanswer = 42\nprint(answer)\n```",
                    "answer": "42",
                }
            )
        ]


def test_pot_concurrency_no_crash():
    """
    Verify that ProgramOfThought can run in multiple threads without crashing
    due to premature interpreter shutdown.
    """
    # Configure Dummy LM
    lm = DummyLM()
    dspy.configure(lm=lm)

    # Initialize implementation
    pot = ProgramOfThought("question -> answer")

    def run_step(i):
        # This triggers interpreter usage
        return pot(question=f"q{i}")

    # Run in parallel
    # 5 workers, 10 items
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_step, i) for i in range(10)]
        results = [f.result() for f in futures]

    # Verify results
    assert len(results) == 10
    for res in results:
        assert isinstance(res, dspy.Prediction)
        # Check against string "42" (or int depending on extraction, usually string from DummyLM unless cast)
        # DummyLM returns answer="42" string in JSON.
        assert res.answer == "42"


if __name__ == "__main__":
    test_pot_concurrency_no_crash()
