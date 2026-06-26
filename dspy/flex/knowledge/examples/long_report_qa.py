"""Long-document QA — keep RLM for the large field; route cheaply in Python."""

import dspy

TASK = "Answer a question about a report that may be far too long to fit in one prompt."
SIGNATURE = "report: str, question: str -> answer: str"
NOTES = (
    "Here RLM is the RIGHT tool, not an anti-pattern: a long `report` is a large blob that needs "
    "iterative exploration. forward() still does the cheap routing in Python — short reports skip "
    "the expensive REPL and use a direct Predict; only long ones pay for RLM."
)


# === MODULE ===
class LongReportQAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.explore = dspy.RLM("report: str, question: str -> answer: str", max_iterations=10)
        self.direct = dspy.Predict("report: str, question: str -> answer: str")

    def forward(self, **inputs):
        report = inputs["report"]
        question = inputs["question"]
        predictor = self.explore if len(report) > 8000 else self.direct
        return dspy.Prediction(answer=str(predictor(report=report, question=question).answer))
