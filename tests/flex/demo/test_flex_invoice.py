from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()
exec_lm = dspy.LM("anthropic/claude-opus-4-7", max_tokens=500)
reflection_lm = dspy.LM("anthropic/claude-opus-4-7", max_tokens=4000)
dspy.configure(lm=exec_lm)

DEMO_DIR = Path(__file__).parent
SAVE_PATH = DEMO_DIR / "invoice_flex.json"


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


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
    try:
        got = int(pred.total_cents)
    except (ValueError, TypeError, AttributeError):
        return ScoreWithFeedback(
            score=0.0,
            feedback="`total_cents` was missing or not an integer. Return whole cents as an int (e.g. $4.50 -> 450).",
        )

    want = int(gold.total_cents)
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


def _showcase(program: dspy.Module, label: str) -> None:
    """Print the flexed module's clean dspy.Module source and its flat predictors."""
    print(f"\n===== {label} =====")
    print("predictors on the module:", [n for n, _ in program.named_predictors()])
    print("--- module_src (a normal dspy.Module subclass) ---")
    print(program.module_src)


def test_invoice_baseline_then_gepa() -> None:
    dspy.configure(lm=exec_lm)
    program = dspy.Flex(InvoiceTotal)

    # The fresh baseline is a clean dspy.Module subclass that delegates to one dspy.Predict.
    assert program.module_src.lstrip().startswith("class ")
    assert "dspy.Predict(" in program.module_src
    _showcase(program, "baseline (un-optimized flex)")

    baseline = mean_score(program, valset)
    optimized = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=24,
        reflection_minibatch_size=2,
        num_threads=1,
    ).compile(program, trainset=trainset, valset=valset)

    # GEPA rewrote the whole class; any predictors it introduced are flat on the module.
    _showcase(optimized, "optimized by GEPA")
    print(f"GEPA changed the code: {optimized.module_src != program.module_src}")
    assert mean_score(optimized, valset) >= baseline

    # Persist with the standard Module.save/load — the generated code round-trips.
    optimized.save(str(SAVE_PATH))
    reloaded = dspy.Flex(InvoiceTotal)
    reloaded.load(str(SAVE_PATH))
    assert reloaded.module_src == optimized.module_src
    print(f"saved + reloaded optimized program -> {SAVE_PATH}")


def test_invoice_manual_edit_is_saved_and_reseeds_gepa(tmp_path) -> None:
    """Showcase: edit a Flex module's code in-session, then save/load it through the standard
    Module API — the edited ``module_src`` round-trips and is what a later dspy.GEPA run seeds
    from (not the baseline). The edit, save, and reload here are all LM-free."""
    from dspy.teleprompt.gepa.gepa_flex_utils import enumerate_flex_submodules

    dspy.configure(lm=exec_lm)
    flex = dspy.Flex(InvoiceTotal)
    assert "dspy.Predict(" in flex.module_src  # fresh baseline

    # Edit the implementation (no LM): replace the baseline with a hand-written extraction module.
    edited_class = (
        "class InvoiceTotalModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        '        self.extract = dspy.Predict("invoice: str -> total_cents: int")\n'
        "\n"
        "    def forward(self, **inputs):\n"
        '        e = self.extract(invoice=inputs["invoice"])\n'
        "        return dspy.Prediction(total_cents=int(e.total_cents))\n"
    )
    flex._bind_code(edited_class)

    # Save and reload through Module.save / load — the edited code round-trips as module_src.
    path = tmp_path / "invoice_flex.json"
    flex.save(path)
    reloaded = dspy.Flex(InvoiceTotal)
    reloaded.load(path)
    assert "self.extract" in reloaded.module_src and "self.predict" not in reloaded.module_src
    _showcase(reloaded, "after manual edit (saved + reloaded)")

    # A subsequent dspy.GEPA run seeds from THIS edited module_src — it builds on the edit.
    (seed,) = [v.module_src for v in enumerate_flex_submodules(reloaded).values()]
    assert "self.extract" in seed
