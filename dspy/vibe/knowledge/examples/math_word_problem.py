"""Math word problem — ChainOfThought reasons, Python coerces the numeric answer."""

import dspy

TASK = "Solve a math word problem and return an integer answer."
SIGNATURE = "problem: str -> answer: int"
NOTES = (
    "A short reasoning step helps on word problems, so use ChainOfThought instead of the heavy "
    "RLM baseline. The model's answer can arrive as '5', '5.0', or 'five apples', so forward() "
    "extracts the number and coerces to the declared int rather than trusting the raw string."
)


# === MODULE ===
class MathWordProblemModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("problem: str -> answer: float")

    def forward(self, **inputs):
        import re

        raw = self.solve(problem=inputs["problem"]).answer
        match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
        value = float(match.group()) if match else 0.0
        return dspy.Prediction(answer=round(value))  # round() of a float yields the declared int
