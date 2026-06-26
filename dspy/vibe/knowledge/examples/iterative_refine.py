"""Iterative refinement — a bounded draft/critique/revise loop controlled in Python."""

import dspy

TASK = "Write a concise summary, refining it until a critic is satisfied."
SIGNATURE = "document: str -> summary: str"
NOTES = (
    "Composes three focused predictors in a BOUNDED loop. Python owns the control flow and the "
    "stopping rule (stop early when the critic approves) instead of asking one prompt to do "
    "everything; the loop is bounded with `for _ in range(N)`."
)

# === PREDICTORS ===
PREDICTORS = {
    "draft": dspy.Predict("document: str -> summary: str"),
    "critique": dspy.ChainOfThought("document: str, summary: str -> approved: bool, issues: str"),
    "revise": dspy.Predict("document: str, current_summary: str, issues: str -> summary: str"),
}


# === FORWARD ===
def forward(self, **inputs):
    document = inputs["document"]
    summary = self.draft(document=document).summary
    for _ in range(3):  # bounded refinement
        review = self.critique(document=document, summary=summary)
        if bool(review.approved):
            break
        summary = self.revise(document=document, current_summary=summary, issues=review.issues).summary
    return dspy.Prediction(summary=str(summary))
