"""End-to-end demo for ``dspy.Flex`` + ``FlexGEPA``: an invoice-total extractor.

The story
=========
You're building an invoice processor. The task sounds trivial — read a
free-text invoice line and return the grand total in whole cents:

    "3 widgets at $4.50 each, 2 gadgets at $10 each, plus $5 shipping"  ->  3850

So you declare a one-line ``dspy.Signature`` and let ``dspy.Flex`` author the
implementation. The obvious first draft is a single LM call: "here's the
invoice, give me the total." On a small model that draft is *shaky* — it has to
parse quantities, multiply, sum, and convert dollars→cents all in its head, and
it gets the arithmetic wrong often enough to matter.

Then you run ``FlexGEPA``. Reading the failures (with a metric that nudges it),
it rewrites the *code itself*: instead of asking the LM to do math, it has the
LM only **extract** the line items and does the arithmetic in plain Python
inside ``forward()``. That's the whole point of Flex — the generated module is
real, readable Python, and the optimizer evolves its structure, not just a
prompt string.

Finally you open the generated file, read the evolved design, and apply one last
hand-polish. Flex detects the manual edit, honors it, and records a new version
in its ledger — you stay in control of the code that ships.

Run it
======
    python tests/flex/demo/test_flex_invoice.py          # narrated story
    pytest tests/flex/demo/test_flex_invoice.py           # same thing, with asserts

Needs an OpenAI key (reads ``.env``). Uses ``gpt-4o-mini`` to keep it cheap.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dotenv import load_dotenv

import dspy
from dspy import FlexGEPA, flex
from dspy.flex import Flex
from dspy.flex.exploration import candidate_id
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from dspy.utils.dummies import DummyLM

load_dotenv()

DEMO_DIR = Path(__file__).parent
FLEX_PATH = DEMO_DIR / "test_flex_invoice.py"

exec_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=600)
reflection_lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=4000)
dspy.configure(lm=exec_lm)


# --------------------------------------------------------------------------- #
# 1. The Signature. One input, one output. The user writes only this.
# --------------------------------------------------------------------------- #
@flex(persist_to=str(FLEX_PATH))
class InvoiceTotal(dspy.Signature):
    """Read a free-text invoice line and compute the grand total in whole cents."""

    invoice: str = dspy.InputField()
    total_cents: int = dspy.OutputField()


# --------------------------------------------------------------------------- #
# 2. Data. Tiny, hand-labeled, with clean integer-cent totals.
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 3. Metric. Returns a score AND feedback — the feedback is what steers GEPA's
#    code rewrite toward "extract, then compute in Python".
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 4. The deliberately naive first draft. A single LM call that has to read the
#    invoice, do the math, and convert to cents all at once. This is the kind of
#    obvious-but-shaky implementation basic codegen lands on; we plant it so the
#    demo's starting point is reproducible.
# --------------------------------------------------------------------------- #
NAIVE_PREDICTORS = """\
PREDICTORS = {
    # One LM call, asked to do everything — including the arithmetic.
    "solve": dspy.ChainOfThought("invoice -> total_cents"),
}"""

NAIVE_FORWARD = """\
def forward(self, **inputs):
    out = self.solve(invoice=inputs["invoice"])
    # Trust whatever the LM returned; coerce it to an int as best we can.
    raw = str(out.total_cents).strip().replace("$", "").replace(",", "")
    try:
        return dspy.Prediction(total_cents=int(float(raw)))
    except ValueError:
        return dspy.Prediction(total_cents=0)"""


# The version a developer might hand-polish to after reading GEPA's output: the
# LM only extracts numbers; Python does the math. Robust and easy to audit.
HAND_POLISHED_PREDICTORS = """\
PREDICTORS = {
    # The LM does only what it's good at: pull the numbers out of the text.
    "extract": dspy.ChainOfThought(
        "invoice -> quantities: list[float], unit_prices_dollars: list[float], "
        "adjustments_dollars: list[float]"
    ),
}"""

HAND_POLISHED_FORWARD = """\
def forward(self, **inputs):
    out = self.extract(invoice=inputs["invoice"])
    quantities = list(out.quantities or [])
    unit_prices = list(out.unit_prices_dollars or [])
    adjustments = list(out.adjustments_dollars or [])
    # Arithmetic lives in Python, not in the LM's head.
    line_total = sum(q * p for q, p in zip(quantities, unit_prices))
    total_dollars = line_total + sum(adjustments)
    total_cents = round(total_dollars * 100)
    return dspy.Prediction(total_cents=max(0, int(total_cents)))"""


_FILE_HEADER = '''\
"""Generated by dspy.Flex (demo). Edit the PREDICTORS dict and forward() body
between the marker comments; dspy.Flex parses those regions back out."""
'''


def write_flex_file(predictors_src: str, forward_src: str, sig_hash: str, body_hash: str) -> None:
    """Write a Flex-format ``.py`` file. ``body_hash`` is written verbatim so we
    can plant a *stale* hash to simulate a hand edit (see ``plant_manual_edit``)."""
    FLEX_PATH.write_text(
        _FILE_HEADER
        + f"\n# __FLEX_SIGNATURE_HASH__: {sig_hash}\n"
        + f"# __FLEX_BODY_HASH__: {body_hash}\n"
        + "# flex_id: InvoiceTotal\n\n"
        + "# __FLEX_PREDICTORS_BEGIN__\n"
        + predictors_src.rstrip("\n")
        + "\n# __FLEX_PREDICTORS_END__\n\n"
        + "# __FLEX_FORWARD_BEGIN__\n"
        + forward_src.rstrip("\n")
        + "\n# __FLEX_FORWARD_END__\n",
        encoding="utf-8",
    )


def reset_demo_state() -> None:
    """Start each run from a clean slate so the narrative is reproducible."""
    if FLEX_PATH.exists():
        FLEX_PATH.unlink()
    shutil.rmtree(DEMO_DIR / ".flex" / "InvoiceTotal", ignore_errors=True)


def signature_hash() -> str:
    """Compute the Signature's content hash WITHOUT a real codegen call (offline
    DummyLM, in-memory only) so we can plant the naive baseline file directly.

    Goes through ``Flex`` directly rather than the decorator factory: the factory
    falls back to the bound ``persist_to`` when passed ``None``, but we want a
    genuinely in-memory probe that writes no files and no ledger entries."""
    probe = Flex(
        InvoiceTotal.signature,
        persist_to=None,
        codegen_lm=DummyLM([{"predictors_src": "PREDICTORS = {}", "forward_src": "def forward(self, **i):\n    return dspy.Prediction()"}]),
    )
    return probe._signature_hash()


def show_predictions(program: dspy.Module, dataset: list[dspy.Example]) -> None:
    for example in dataset:
        try:
            got = program(**example.inputs()).total_cents
        except Exception as e:
            got = f"<crash: {e}>"
        mark = "OK " if str(got) == str(example.total_cents) else "BAD"
        print(f"    [{mark}] want {example.total_cents:>5} | got {got!s:>8} | {example.invoice}")


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> dspy.Module:
    reset_demo_state()

    # --- Beat 1: plant the naive first draft -------------------------------- #
    banner("1. The first draft: one LM call that does the math in its head")
    write_flex_file(NAIVE_PREDICTORS, NAIVE_FORWARD, signature_hash(),
                    candidate_id(NAIVE_PREDICTORS, NAIVE_FORWARD))
    program = InvoiceTotal()  # loads the planted baseline (signature hash matches)
    print(program.forward_src)

    baseline = mean_score(program, valset)
    print(f"\n  Baseline val score: {baseline:.2f}  (the small model fumbles the arithmetic)")
    show_predictions(program, valset)

    # --- Beat 2: let FlexGEPA evolve the CODE -------------------------------- #
    banner("2. FlexGEPA rewrites the implementation from the failures")
    optimized = FlexGEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=24,
        reflection_minibatch_size=2,
        num_threads=1,
    ).compile(program, trainset=trainset, valset=valset)

    gepa_val = mean_score(optimized, valset)
    print("\n  GEPA's evolved forward():\n")
    print(optimized.forward_src)
    print(f"\n  GEPA val score: {gepa_val:.2f}  (was {baseline:.2f})")
    show_predictions(optimized, valset)

    # --- Beat 3: the developer reviews and hand-polishes --------------------- #
    banner("3. You read the evolved code and apply a final hand-polish")
    plant_manual_edit()
    final = InvoiceTotal()  # Flex detects the body changed, honors the hand edit
    print(final.forward_src)

    final_val = mean_score(final, valset)
    held = final(**heldout.inputs()).total_cents
    print(f"\n  Final val score: {final_val:.2f}")
    print(f"  Held-out check: '{heldout.invoice}' -> {held} (want {heldout.total_cents})")
    print("\n  Ledger (every codegen / evaluate / propose / manual_edit event):")
    for entry in final._exploration.get_history()[-6:]:
        print(f"    {entry.get('event'):<11} {entry.get('candidate_id', '')[:12]:<12} "
              f"score={entry.get('score')}")

    return final


def plant_manual_edit() -> None:
    """Overwrite the body with the hand-polished version, keeping the *previous*
    body hash. Flex sees ``candidate_id(new body) != recorded hash`` on the next
    load, treats it as a manual edit, honors it, and refreshes the hash."""
    program = InvoiceTotal()
    stored = program._read_persisted()
    write_flex_file(
        HAND_POLISHED_PREDICTORS, HAND_POLISHED_FORWARD,
        stored["signature_hash"], stored["body_hash"],  # stale body hash on purpose
    )


def test_invoice_flex() -> None:
    """Pytest entry point: the shipped program must read invoices correctly and
    be no worse than the naive baseline."""
    reset_demo_state()
    write_flex_file(NAIVE_PREDICTORS, NAIVE_FORWARD, signature_hash(),
                    candidate_id(NAIVE_PREDICTORS, NAIVE_FORWARD))
    program = InvoiceTotal()
    baseline = mean_score(program, valset)

    optimized = FlexGEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=24,
        reflection_minibatch_size=2,
        num_threads=1,
    ).compile(program, trainset=trainset, valset=valset)

    plant_manual_edit()
    final = InvoiceTotal()

    assert mean_score(final, valset) >= baseline
    assert int(final(**heldout.inputs()).total_cents) == heldout.total_cents


if __name__ == "__main__":
    main()
