"""Invoice total — the LM extracts line items, Python does the arithmetic."""

import dspy

TASK = "Read a free-text invoice line and compute the grand total in whole cents."
SIGNATURE = "invoice: str -> total_cents: int"
NOTES = (
    "Arithmetic is unreliable in an LM, so the predictor only EXTRACTS the structured "
    "numbers; forward() does the math in plain Python and coerces to the declared int."
)


# === MODULE ===
class InvoiceTotalModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict("invoice: str -> qty: int, unit_price_cents: int, shipping_cents: int")

    def forward(self, **inputs):
        e = self.extract(invoice=inputs["invoice"])
        total = int(e.qty) * int(e.unit_price_cents) + int(e.shipping_cents)
        return dspy.Prediction(total_cents=int(total))
