"""Tests for `BaseLM.__init_subclass__` signature detection and DeprecationWarning.

Validates the dual-contract evolution introduced in the in-place BaseLM plan:
v1 subclasses (legacy `forward(prompt, messages)`) emit DeprecationWarning at
class-definition time; v2 subclasses (typed `forward(request)`) do not.
"""

from __future__ import annotations

import warnings

import pytest

import dspy
from dspy.clients.base_lm import BaseLM, _detect_contract_version


def _make_class(name: str, module: str, forward):
    """Create a BaseLM subclass with the given module path and forward method."""
    ns = {"__module__": module, "forward": forward}
    return type(name, (BaseLM,), ns)


def test_detect_v1_signature_prompt_param():
    def forward(self, prompt=None, messages=None, **kw):
        return None

    cls = _make_class("V1", "user_code", forward)
    assert cls._lm_contract_version == 1
    assert _detect_contract_version(cls) == 1


def test_detect_v1_signature_messages_param():
    def forward(self, messages=None, **kw):
        return None

    cls = _make_class("V1Msg", "user_code", forward)
    assert cls._lm_contract_version == 1


def test_detect_v2_signature_typed_request():
    def forward(self, request):
        return dspy.LMResponse.from_text("ok", model="t/m")

    cls = _make_class("V2", "user_code", forward)
    assert cls._lm_contract_version == 2


def test_detect_v2_when_subclass_does_not_override_forward():
    cls = type("NoForward", (BaseLM,), {"__module__": "user_code"})
    # No override means BaseLM's inherited typed default — treat as v2.
    assert cls._lm_contract_version == 2


def test_v1_external_subclass_emits_deprecation_warning():
    def forward(self, prompt=None, messages=None, **kw):
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _make_class("V1Warn", "user_code", forward)

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation) == 1
    assert "legacy LM contract" in str(deprecation[0].message)
    assert "DSPy 4.0" in str(deprecation[0].message)
    assert "migration/baselm" in str(deprecation[0].message)


def test_v2_external_subclass_does_not_warn():
    def forward(self, request):
        return dspy.LMResponse.from_text("ok", model=request.model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _make_class("V2NoWarn", "user_code", forward)

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation == []


def test_in_tree_v1_subclass_is_exempt_from_warning():
    """dspy.LM is v1 but lives in dspy.* — should not warn at import time."""
    assert dspy.LM._lm_contract_version == 1
    # If LM had warned, this import would have raised under simplefilter=error
    # in conftest. The fact that the test module imports cleanly proves no warning.


def test_internal_opt_out_via_init_subclass_kw():
    """`class Foo(BaseLM, _internal=True)` bypasses the deprecation warning."""

    def forward(self, prompt=None, messages=None, **kw):
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        type(
            "Internal",
            (BaseLM,),
            {"__module__": "user_code", "forward": forward},
            _internal=True,
        )

    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation == []


def test_v1_call_legacy_string_positional_preserves_compat():
    """`lm("text")` on a v1 subclass continues to map to prompt="text"."""

    captured = {}

    class FakeChoice:
        class message:
            content = "echo"
            tool_calls = None

    class FakeResponse:
        model = "fake"
        choices = [FakeChoice()]
        usage = {}
        _hidden_params = {}

    def forward(self, prompt=None, messages=None, **kw):
        captured["prompt"] = prompt
        captured["messages"] = messages
        return FakeResponse()

    cls = _make_class("V1PositionalCompat", "user_code", forward)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lm = cls(model="fake")
    out = lm("hello world")
    assert captured["prompt"] == "hello world"
    assert captured["messages"] is None
    assert out == ["echo"]


def test_v1_subclass_rejects_positional_non_string_items():
    def forward(self, prompt=None, messages=None, **kw):
        return None

    cls = _make_class("V1NoItems", "user_code", forward)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lm = cls(model="fake")

    with pytest.raises(TypeError, match="legacy v1 LM contract"):
        lm(dspy.User("hello"))


def test_v2_subclass_returns_lm_response():
    class V2Echo(BaseLM):
        __module__ = "user_code"

        def forward(self, request):
            return dspy.LMResponse.from_text("v2-ok", model=request.model)

    lm = V2Echo(model="t/echo")
    response = lm("hello")
    assert isinstance(response, dspy.LMResponse)
    assert response.text == "v2-ok"
