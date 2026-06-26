"""Intent classification into a fixed label set, normalized in Python."""

import dspy

TASK = "Classify a retail-banking customer message into one of a fixed set of intents."
SIGNATURE = "text: str -> intent: str"
NOTES = (
    "The exact allowed labels live in the predictor's instructions so the model always sees "
    "them; forward() normalizes the raw output (strip/lowercase) and falls back to 'other' "
    "when the model returns something off-list."
)

# === PREDICTORS ===
PREDICTORS = {
    "classify": dspy.Predict(
        dspy.Signature(
            "text -> intent",
            "Classify the customer's message into exactly one of these intents: "
            "balance, transfer, dispute, card_lost, other. "
            "Return only the label, lowercase, with no extra words.",
        )
    ),
}


# === FORWARD ===
def forward(self, **inputs):
    allowed = {"balance", "transfer", "dispute", "card_lost", "other"}
    raw = self.classify(text=inputs["text"]).intent
    label = (raw or "").strip().lower().replace(" ", "_")
    return dspy.Prediction(intent=label if label in allowed else "other")
