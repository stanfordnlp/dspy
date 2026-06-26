"""Slugify — a fully deterministic task needs no LM at all."""

import dspy

TASK = "Turn a title into a URL slug."
SIGNATURE = "title: str -> slug: str"
NOTES = (
    "No step needs a language model, so __init__ defines no predictors and forward() is plain "
    "Python with a nested helper. A pure-Python module is the most reliable outcome optimization "
    "can reach."
)


# === MODULE ===
class SlugifyModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        def slugify(text):
            cleaned = "".join(c.lower() if c.isalnum() else "-" for c in text)
            return "-".join(part for part in cleaned.split("-") if part)

        return dspy.Prediction(slug=slugify(inputs["title"]))
