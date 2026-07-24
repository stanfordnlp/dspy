import sys as _dspy_sys
import types as _dspy_types


def _dspy_host(_fn, **_kw):
    # Call a registered host tool by name (the CodeInterpreter.tools contract) and return its result.
    return globals()[_fn](**_kw)


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

    def __call__(self, **_inputs):
        _h = object.__getattribute__(self, "_handle")
        _out = _dspy_host("__dspy_call__", handle=_h, inputs=_inputs)
        return _DspyPrediction(**(_out or {}))


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


_dspy = _dspy_types.ModuleType("dspy")
_dspy.Module = _DspyModule
_dspy.Prediction = _DspyPrediction
_dspy.Signature = _dspy_signature
for _k in ("Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2"):
    setattr(_dspy, _k, _dspy_make_ctor(_k))
dspy = _dspy

# Register as the importable ``dspy`` only inside the sandbox, where the registered host tools are
# present in globals().
if "__dspy_construct__" in globals():
    _dspy_sys.modules["dspy"] = _dspy
