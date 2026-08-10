"""Acceptance test for the reserved-name collision issue (stanfordnlp/dspy#658).

Intent: a field name that collides with a DSPy accessor must never leave the user
misled. Either the field works, or DSPy raises an error that NAMES THE COLLISION.
What must not happen: an error that falsely blames the user's declaration, or a
value that is silently swallowed before reaching the adapter.
"""

import pytest

import dspy
from dspy.utils.dummies import DummyLM

SHADOWED = ["instructions", "fields", "signature", "input_fields", "output_fields"]
RESERVED_KWARGS = ["demos", "config", "lm"]


def _make_sig(name, kind="output"):
    field = dspy.OutputField() if kind == "output" else dspy.InputField()
    ann = {"question": str, name: str} if kind == "output" else {name: str, "answer": str}
    ns = {"__annotations__": ann, "__doc__": "Doc here."}
    if kind == "output":
        ns["question"] = dspy.InputField()
        ns[name] = field
    else:
        ns[name] = field
        ns["answer"] = dspy.OutputField()
    return type("ReservedSig", (dspy.Signature,), ns)


@pytest.mark.parametrize("name", SHADOWED)
def test_shadowing_field_name_is_not_misdiagnosed(name):
    """Declaring a field that collides with a metaclass property must not produce
    the misleading 'must be declared with InputField or OutputField' error."""
    try:
        sig = _make_sig(name, kind="output")
    except Exception as e:
        msg = str(e)
        assert "json_schema_extra=None" not in msg, (
            f"Field `{name}` WAS declared with OutputField, but DSPy reports it was not. "
            f"The message blames the user's declaration instead of naming the collision "
            f"with the reserved Signature attribute `{name}`. Got: {msg}"
        )
        # An error is acceptable only if it identifies the collision.
        assert name in msg and any(k in msg.lower() for k in ("reserved", "shadow", "collide", "conflict")), (
            f"Error for `{name}` does not identify the name collision. Got: {msg}"
        )
        return
    # Otherwise the field must actually register.
    assert name in sig.output_fields


@pytest.mark.parametrize("name", RESERVED_KWARGS)
def test_reserved_kwarg_is_not_silently_stolen(name):
    """An input field named `demos`/`config`/`lm` must not have its value eaten by
    Predict._forward_preprocess before it reaches the adapter."""
    seen = {}

    class SpyAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs, **kw):
            seen.clear()
            seen.update(inputs)
            return [{"answer": "ok"}]

    with dspy.context(lm=DummyLM([{"answer": "ok"}] * 3), adapter=SpyAdapter()):
        sig = _make_sig(name, kind="input")
        assert name in sig.input_fields  # signature-level validation permits these names
        # The collision may legitimately be reported either when the Predict is constructed
        # (fail-fast) or when it is called, so both are inside the `try`.
        try:
            predict = dspy.Predict(sig)
            predict(**{name: "MY-VALUE"})
        except Exception as e:
            msg = str(e)
            assert name in msg and any(k in msg.lower() for k in ("reserved", "shadow", "collide", "conflict")), (
                f"Input field `{name}` was consumed as a privileged Predict kwarg and the "
                f"resulting error does not explain the collision. Got: {msg}"
            )
            return
        assert seen.get(name) == "MY-VALUE", (
            f"Input field `{name}` never reached the adapter -- its value was silently "
            f"popped by Predict._forward_preprocess. Adapter saw: {seen}"
        )
