"""Guest bridge for Monty's native classes; no Python object-model emulation.

The compiler supplies the attribute hooks Monty does not implement. Only
predictor construction/calls and registered tools cross to the host.
"""

_dspy_native_getattr = getattr
_dspy_native_isinstance = isinstance
_dspy_native_type = type
_dspy_slice = slice
_dspy_module_types = []
_dspy_tool_entries = []
_dspy_next_handle = [0]


class _DspyPrediction:
    def __init__(self, **fields):
        self._fields = fields


class _DspyModule:
    def __init__(self, *args, **kwargs):
        pass


def _dspy_module_init(*args, **kwargs):
    pass


def _dspy_check_base(base):
    if base is not _DspyModule:
        raise TypeError("Monty Flex supports only the dspy.Module base")


def _dspy_getattr(obj, name, *default):
    if len(default) > 1:
        raise TypeError("getattr expected at most 3 arguments")
    try:
        return _dspy_native_getattr(obj, name)
    except AttributeError:
        if _dspy_native_isinstance(obj, _DspyPrediction) and name in obj._fields:
            return obj._fields[name]
        if default:
            return default[0]
        raise AttributeError(
            "Attribute '" + name + "' is unavailable in Monty. "
            "If this is a native method, call it directly or use a named helper "
            "instead of taking the method as a value."
        )


def _dspy_hasattr(obj, name):
    try:
        _dspy_getattr(obj, name)
        return True
    except AttributeError:
        return False


def _dspy_proxy(handle):
    def call(**inputs):
        fields = __dspy_call__(handle=handle, inputs=inputs)  # noqa: F821 - host tool
        return _DspyPrediction(**(fields or {}))

    return call


def _dspy_getitem(obj, key):
    if _dspy_native_isinstance(obj, _DspyPrediction):
        return obj._fields[key]
    return obj[key]


def _dspy_isinstance(value, cls):
    if cls is _DspyModule and _dspy_native_type(value) in _dspy_module_types:
        return True
    if _dspy_native_isinstance(cls, tuple):
        return any(_dspy_isinstance(value, item) for item in cls)
    return _dspy_native_isinstance(value, cls)


def _dspy_enc(value):
    for tool, name in _dspy_tool_entries:
        if value is tool:
            return {"__dspy_tool__": name}
    if _dspy_native_type(value) is _dspy_native_type(_dspy_enc):
        raise TypeError("Only provided tools can be passed to a bridged predictor")
    if _dspy_native_isinstance(value, (list, tuple)):
        return [_dspy_enc(item) for item in value]
    if _dspy_native_isinstance(value, dict):
        return {key: _dspy_enc(item) for key, item in value.items()}
    return value


def _dspy_make_ctor(kind):
    def ctor(signature=None, **kwargs):
        name = "_dspy_anon_" + str(_dspy_next_handle[0])
        _dspy_next_handle[0] += 1
        handle = __dspy_construct__(  # noqa: F821 - host tool
            kind=kind, signature=signature, attr_name=name, kwargs=_dspy_enc(kwargs),
        )
        return _dspy_proxy(handle)

    return ctor


def _dspy_signature(signature, instructions=None, **kwargs):
    return {"__dspy_sig__": True, "signature": signature, "instructions": instructions}


def _dspy_tool(func, **kwargs):
    return func


_dspy_namespace = _DspyModule()
_dspy_namespace.Module = _DspyModule
_dspy_namespace.Prediction = _DspyPrediction
_dspy_namespace.Signature = _dspy_signature
_dspy_namespace.Tool = _dspy_tool
for _dspy_kind in ("Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2"):
    setattr(_dspy_namespace, _dspy_kind, _dspy_make_ctor(_dspy_kind))

dspy = _dspy_namespace
getattr = _dspy_getattr
hasattr = _dspy_hasattr
isinstance = _dspy_isinstance
