"""Sandbox side of the dspy facade: a stand-in ``dspy`` module for sandboxed code.

Consumers such as ``dspy.Flex`` and ``dspy.RLM`` execute this source in the interpreter after
registering the host tools, so sandboxed code builds and calls predictors that actually run on
the host:

- ``dspy.Predict`` and other allowed modules return a ``_DspyPending``; construction waits for attribute assignment,
  because the attribute name is the predictor's host-side handle.
- ``_DspyModule.__setattr__`` has the host build the real predictor (``__dspy_construct__``) and
  binds a ``_DspyProxy`` in its place. Calling the proxy runs the predictor (``__dspy_call__``)
  and wraps the returned output fields in a ``_DspyPrediction``.
- Callables and signatures cannot cross the JSON boundary as themselves, so they travel as
  markers the host resolves: tools by name (``__dspy_tool__``), ``dspy.Signature(...)`` results
  as ``__dspy_sig__`` payloads.

Every name here is ``_dspy``-prefixed to stay clear of sandboxed code; consumers reject user tool
names in that namespace.
"""

import sys as _dspy_sys
import types as _dspy_types
import typing as _dspy_typing


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


_dspy_anon_count = 0


class _DspyPending:
    """Returned by a shim constructor before the attribute name is known (captured in __setattr__)."""

    def __init__(self, _kind, _sig, _kwargs):
        self.kind = _kind
        self.sig = _sig
        self.kwargs = _kwargs
        self.proxy = None

    def __call__(self, **_inputs):
        if self.proxy is None:
            global _dspy_anon_count
            _dspy_anon_count += 1
            _handle = _dspy_host(
                "__dspy_construct__",
                kind=self.kind,
                signature=self.sig,
                attr_name="_dspy_anon_" + str(_dspy_anon_count),
                kwargs=self.kwargs,
            )
            self.proxy = _DspyProxy(_handle)
        return self.proxy(**_inputs)


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


def _dspy_tool_name(_v):
    # Host tools are referenced by the sandbox global they are bound to: backends may bind them as
    # anonymous proxies, so the callable's own __name__ is only a fallback.
    for _k, _x in globals().items():
        if _x is _v and not _k.startswith("_"):
            return _k
    return getattr(_v, "__name__", type(_v).__name__)


def _dspy_enc(_v):
    # Tool references (e.g. tools=[shout]) are sandbox callables; send them to the host by name.
    if callable(_v):
        return {"__dspy_tool__": _dspy_tool_name(_v)}
    if isinstance(_v, (list, tuple)):
        return [_dspy_enc(_x) for _x in _v]
    if isinstance(_v, dict):
        return {_k: _dspy_enc(_x) for _k, _x in _v.items()}
    return _v


def _dspy_make_ctor(_kind):
    def _ctor(signature=None, **_kwargs):
        return _DspyPending(_kind, _dspy_sig(signature), {_k: _dspy_enc(_v) for _k, _v in _kwargs.items()})

    return _ctor


class _DspyField:
    def __init__(self, _io, _kwargs):
        self.io = _io
        self.kwargs = _kwargs


class _DspySignature:
    # dspy.Signature("a -> b", "instructions") returns a __dspy_sig__ payload; subclasses are class-based signatures.
    def __new__(cls, signature, instructions=None, **_kw):
        return {"__dspy_sig__": True, "signature": signature, "instructions": instructions}


def _dspy_type_str(_t):
    if isinstance(_t, str):
        # Under `from __future__ import annotations`, a quoted annotation arrives with its quotes.
        return _t[1:-1] if len(_t) > 1 and _t[0] == _t[-1] and _t[0] in "'\"" else _t
    if isinstance(_t, type) and not _dspy_typing.get_args(_t):
        return _t.__name__
    return repr(_t).replace("typing.", "")


def _dspy_sig(_s):
    # A class-based signature crosses as a string signature plus each field's InputField/OutputField kwargs.
    if not (isinstance(_s, type) and issubclass(_s, _DspySignature)):
        return _s
    _fields, _annotated = {}, set()
    for _c in reversed(_s.__mro__):
        _ann = getattr(_c, "__annotations__", {})
        _annotated.update(_ann)
        for _k, _v in vars(_c).items():
            if isinstance(_v, _DspyField):
                _fields[_k] = (_v, _dspy_type_str(_ann.get(_k, str)))
    _missing = sorted(_annotated - set(_fields))
    if _missing:
        raise TypeError(f"Field(s) {_missing} in {_s.__name__} must be declared with InputField or OutputField")
    _sides = {"input": [], "output": []}
    for _k, (_v, _t) in _fields.items():
        _sides[_v.io].append(_k + ": " + _t)
    return {
        "__dspy_sig__": True,
        "signature": ", ".join(_sides["input"]) + " -> " + ", ".join(_sides["output"]),
        "instructions": _s.__doc__,
        "fields": {_k: _v.kwargs for _k, (_v, _t) in _fields.items()},
    }


def _dspy_tool(func, **_kw):
    return func


_dspy = _dspy_types.ModuleType("dspy")
_dspy.Module = _DspyModule
_dspy.Prediction = _DspyPrediction
_dspy.Signature = _DspySignature
_dspy.InputField = lambda **_kw: _DspyField("input", _kw)
_dspy.OutputField = lambda **_kw: _DspyField("output", _kw)
_dspy.Tool = _dspy_tool
for _k in ("Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2"):
    setattr(_dspy, _k, _dspy_make_ctor(_k))
dspy = _dspy

# Register as the importable ``dspy`` only inside the sandbox, where the registered host tools are
# present in globals(). An interpreter that runs code in the host's memory (the host process or a fork
# of it) shares the host's sys.modules, so there ``import dspy`` resolves to the shim through this
# namespace's own __import__ and the host's ``dspy`` module stays in place.
if "__dspy_construct__" in globals():
    _dspy_host_facade = getattr(_dspy_sys, "modules", {}).get("dspy.primitives.facade")
    if getattr(_dspy_host_facade, "_HOST_PROCESS_TOKEN", None) == __dspy_host_token:  # noqa: F821 - host-injected
        _dspy_builtins = globals()["__builtins__"]
        _dspy_builtins = dict(_dspy_builtins if isinstance(_dspy_builtins, dict) else vars(_dspy_builtins))
        _dspy_import_orig = _dspy_builtins["__import__"]

        def _dspy_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name.split(".")[0] == "dspy":
                if name != "dspy":
                    raise ModuleNotFoundError(f"No module named {name!r}; 'dspy' is not a package", name=name)
                return _dspy
            return _dspy_import_orig(name, globals, locals, fromlist, level)

        _dspy_builtins["__import__"] = _dspy_import
        globals()["__builtins__"] = _dspy_builtins
    else:
        _dspy_sys.modules["dspy"] = _dspy
