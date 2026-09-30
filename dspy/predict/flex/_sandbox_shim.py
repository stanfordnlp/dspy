"""Sandbox side of the dspy.Flex bridge: a stand-in ``dspy`` module for optimizer-authored code.

``BridgeRuntime.forward`` executes this source in each per-forward interpreter before the bound
``module_src``, so the generated module runs sandboxed while predictors are built and called on
the host:

- ``dspy.Predict`` and other allowed modules return a ``_DspyPending``; construction waits for attribute assignment,
  because the attribute name is the predictor's host-side handle.
- ``_DspyModule.__setattr__`` has the host build the real predictor (``__dspy_construct__``) and
  binds a ``_DspyProxy`` in its place. Calling the proxy runs the predictor (``__dspy_call__``)
  with the proxy's ``fields`` (decision configuration, as on ``dspy.Predict``) and wraps the
  returned output fields in a ``_DspyPrediction``.
- Decision values arrive as JSON objects marked ``__dspy_decision__`` and become ``_DspyDecision``
  objects: still dicts, so they cross back unchanged, with the host types' ``bool``/``float``
  conversions and attribute access.
- Callables and signatures cannot cross the JSON boundary as themselves, so they travel as
  markers the host resolves: tools by name (``__dspy_tool__``), ``dspy.Signature(...)`` results
  as ``__dspy_sig__`` payloads.

Every name here is ``_dspy``-prefixed to stay clear of generated code; ``FlexContext`` rejects
user tool names in that namespace.
"""

import sys as _dspy_sys
import types as _dspy_types


def _dspy_host(_fn, **_kw):
    # Call a registered host tool by name (the CodeInterpreter.tools contract) and return its result.
    return globals()[_fn](**_kw)


class _DspyDecision(dict):
    """Sandbox-side stand-in for a Noul, Score, or Choice value.

    A dict of the decision's JSON fields plus its ``__dspy_decision__`` kind, so it serializes back
    to the host unchanged. ``bool()`` of a Noul and ``float()`` of a Score give its value, as on the host.
    """

    def __init__(self, _data):
        _data = dict(_data)
        if _data.get("__dspy_decision__") == "score" and isinstance(_data.get("probabilities"), dict):
            # JSON object keys are strings; the host keys a Score's distribution by level index.
            _data["probabilities"] = {int(_k): _v for _k, _v in _data["probabilities"].items()}
        dict.__init__(self, _data)

    def __getattr__(self, _name):
        if _name in self:
            return self[_name]
        if _name in ("value", "confidence", "probability", "probabilities", "level"):
            return None
        raise AttributeError(_name)

    def __bool__(self):
        if self.get("__dspy_decision__") == "noul":
            return bool(self["value"])
        return True

    def __float__(self):
        if self.get("__dspy_decision__") == "choice":
            raise TypeError("a Choice has no numeric value; use .value")
        return float(self["value"])

    def __repr__(self):
        _kind = str(self.get("__dspy_decision__", "decision")).capitalize()
        _shown = ", ".join(_k + "=" + repr(_v) for _k, _v in self.items() if _k != "__dspy_decision__")
        return _kind + "(" + _shown + ")"


def _dspy_decisions(_v):
    # Rebuild decision values the host marked, recursing into lists and dicts.
    if isinstance(_v, dict):
        if _v.get("__dspy_decision__") in ("noul", "score", "choice"):
            return _DspyDecision(_v)
        return {_k: _dspy_decisions(_x) for _k, _x in _v.items()}
    if isinstance(_v, list):
        return [_dspy_decisions(_x) for _x in _v]
    return _v


class _DspyPrediction:
    """Sandbox-side stand-in for dspy.Prediction; just holds output fields."""

    def __init__(self, **_fields):
        object.__setattr__(self, "_fields", dict(_fields))

    def __getattr__(self, _name):
        _f = object.__getattribute__(self, "_fields")
        if _name in _f:
            return _f[_name]
        raise AttributeError(_name)

    def __getitem__(self, _name):
        return object.__getattribute__(self, "_fields")[_name]

    def __repr__(self):
        return "Prediction(" + repr(object.__getattribute__(self, "_fields")) + ")"


class _DspyProxy:
    """Sandbox-side handle to a host predictor. Calling it runs the real predictor on the host."""

    def __init__(self, _handle):
        object.__setattr__(self, "_handle", _handle)
        # Decision configuration per output, as on dspy.Predict; sent with every call.
        object.__setattr__(self, "fields", {})

    def __call__(self, **_inputs):
        _h = object.__getattribute__(self, "_handle")
        _kw = {"handle": _h, "inputs": _inputs}
        if self.fields:
            _kw["fields"] = self.fields
        _out = _dspy_host("__dspy_call__", **_kw)
        return _DspyPrediction(**_dspy_decisions(_out or {}))


class _DspyPending:
    """Returned by a shim constructor before the attribute name is known (captured in __setattr__)."""

    def __init__(self, _kind, _sig, _kwargs):
        self.kind = _kind
        self.sig = _sig
        self.kwargs = _kwargs


class _DspyModule:
    def __init__(self, *_a, **_k):
        pass

    def __setattr__(self, _name, _value):
        if isinstance(_value, _DspyPending):
            _h = _dspy_host(
                "__dspy_construct__",
                kind=_value.kind,
                signature=_value.sig,
                attr_name=_name,
                kwargs=_value.kwargs,
            )
            _value = _DspyProxy(_h)
        object.__setattr__(self, _name, _value)

    def __call__(self, **_kw):
        return self.forward(**_kw)


def _dspy_enc(_v):
    # Tool references (e.g. tools=[shout]) are sandbox functions; send them to the host by name.
    if callable(_v) and hasattr(_v, "__name__"):
        return {"__dspy_tool__": _v.__name__}
    if isinstance(_v, (list, tuple)):
        return [_dspy_enc(_x) for _x in _v]
    if isinstance(_v, dict):
        return {_k: _dspy_enc(_x) for _k, _x in _v.items()}
    return _v


def _dspy_make_ctor(_kind):
    def _ctor(signature=None, **_kwargs):
        return _DspyPending(_kind, signature, {_k: _dspy_enc(_v) for _k, _v in _kwargs.items()})

    return _ctor


def _dspy_signature(signature, instructions=None, **_kw):
    return {"__dspy_sig__": True, "signature": signature, "instructions": instructions}


def _dspy_tool(func, **_kw):
    return func


_dspy = _dspy_types.ModuleType("dspy")
_dspy.Module = _DspyModule
_dspy.Prediction = _DspyPrediction
_dspy.Signature = _dspy_signature
_dspy.Tool = _dspy_tool
for _k in ("Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2"):
    setattr(_dspy, _k, _dspy_make_ctor(_k))
dspy = _dspy

# Register as the importable ``dspy`` only inside the sandbox, where the registered host tools are
# present in globals().
if "__dspy_construct__" in globals():
    _dspy_sys.modules["dspy"] = _dspy
