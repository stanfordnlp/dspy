from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

import dspy
from dspy import FlexGEPA, flex
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()
exec_lm = dspy.LM("anthropic/claude-opus-4-7", max_tokens=500)
reflection_lm = dspy.LM("anthropic/claude-opus-4-7", max_tokens=4000)
dspy.configure(lm=exec_lm)

DEMO_DIR = Path(__file__).parent
FLEX_PATH = DEMO_DIR / "invoice_flex_gen.py"


@flex(persist_to=str(FLEX_PATH))
class InvoiceTotal(dspy.Signature):
    """Read a free-text invoice line and compute the grand total in whole cents."""

    invoice: str = dspy.InputField()
    total_cents: int = dspy.OutputField()


def ex(invoice: str, total_cents: int) -> dspy.Example:
    return dspy.Example(invoice=invoice, total_cents=total_cents).with_inputs("invoice")


trainset = [
    ex("3 widgets at $4.50 each, 2 gadgets at $10 each, plus $5 shipping", 3850),
    ex("Two notebooks for $3 apiece and one pen for $1.25", 725),
    ex("1 keyboard at $30, 1 mouse at $15, $5 off the order", 4000),
    ex("5 stickers at $0.40 each", 200),
]
valset = [
    ex("4 cans of soda at $1.25 each and a $2 deposit", 700),
    ex("1 coffee mug at $7.50", 750),
    ex("8 oranges at $0.75 each", 600),
]
heldout = ex("10 pencils at $0.30 each plus $1.50 shipping", 450)


def metric(example, prediction, trace=None) -> ScoreWithFeedback:
    try:
        got = int(prediction.total_cents)
    except (ValueError, TypeError, AttributeError):
        return ScoreWithFeedback(
            score=0.0,
            feedback="`total_cents` was missing or not an integer. Return whole "
            "cents as an int (e.g. $4.50 -> 450).",
        )

    want = int(example.total_cents)
    if got == want:
        return ScoreWithFeedback(score=1.0, feedback="Correct.")
    fb = f"Wrong total: got {got} cents, expected {want} cents. "
    if got * 100 == want:
        fb += "That looks like DOLLARS — the field wants whole CENTS (x100)."
    else:
        fb += (
            "Don't make the LM do the arithmetic in its head. Have the LM only "
            "EXTRACT each line item's quantity and unit price (and any shipping/"
            "discount adjustments), then SUM them in plain Python inside forward()."
        )
    return ScoreWithFeedback(score=0.0, feedback=fb)


def mean_score(program: dspy.Module, dataset: list[dspy.Example]) -> float:
    total = 0.0
    for example in dataset:
        try:
            pred = program(**example.inputs())
            total += float(metric(example, pred).score)
        except Exception:
            pass  # a crash scores 0, same as the optimizer would see
    return total / len(dataset)


def test_invoice_codegen_then_gepa() -> None:
    program = InvoiceTotal()

    baseline = mean_score(program, valset)
    optimized = FlexGEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=24,
        reflection_minibatch_size=2,
        num_threads=1,
    ).compile(program, trainset=trainset, valset=valset)

    assert mean_score(optimized, valset) >= baseline


def test_invoice_manual_edit() -> None:
    program = InvoiceTotal()
    pred = program(**heldout.inputs())
    print(f"Before manual edit: predicted total_cents = {pred.total_cents}, want {heldout.total_cents}")
