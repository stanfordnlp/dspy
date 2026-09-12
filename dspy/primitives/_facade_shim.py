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
# Nested code-executing sub-agents take this in place of a real factory; the host substitutes its own.
dspy_interpreter_factory = "__dspy_interpreter_factory__"

# Register as the importable ``dspy`` only inside the sandbox, where the registered host tools are
# present in globals(). A sandbox whose ``dspy`` is the host's own module image (the host process or a
# fork of it) would hand generated code the host's memory, so the facade refuses it.
if "__dspy_construct__" in globals():
    _dspy_host_facade = getattr(_dspy_sys, "modules", {}).get("dspy.primitives.facade")
    if getattr(_dspy_host_facade, "_HOST_PROCESS_TOKEN", None) == __dspy_host_token:  # noqa: F821 - host-injected
        raise RuntimeError(
            "This interpreter runs code in the host's memory (the host process or a fork of it); "
            "the dspy facade needs an isolated interpreter."
        )
    _dspy_sys.modules["dspy"] = _dspy
