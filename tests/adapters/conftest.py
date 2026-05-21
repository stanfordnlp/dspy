import dspy


class StopAdapterCallCapture(BaseException):
    """Stop adapter execution after capturing the LM call.

    The exact-format tests assert the adapter-to-LM boundary: the messages and
    keyword arguments passed to the LM. Raising here avoids needing to craft a
    parseable LM response for every signature under test.
    """


class CapturingLM(dspy.BaseLM):
    def __init__(self, source_lm=None):
        source_lm = source_lm or dspy.utils.DummyLM([{}])
        super().__init__(
            model=source_lm.model,
            model_type=source_lm.model_type,
            cache=source_lm.cache,
            **source_lm.kwargs,
        )
        self.source_lm = source_lm
        self.calls = []

    @property
    def supports_function_calling(self):
        return self.source_lm.supports_function_calling

    @property
    def supports_reasoning(self):
        return self.source_lm.supports_reasoning

    @property
    def supports_response_schema(self):
        return self.source_lm.supports_response_schema

    @property
    def supported_params(self):
        return self.source_lm.supported_params

    def __call__(self, messages=None, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        raise StopAdapterCallCapture


def format_messages_and_lm_kwargs(adapter, signature, demos, inputs, lm_kwargs=None, lm=None):
    capturing_lm = CapturingLM(lm)
    try:
        adapter(capturing_lm, dict(lm_kwargs or {}), signature, demos, inputs)
    except StopAdapterCallCapture:
        pass

    assert len(capturing_lm.calls) == 1
    call = capturing_lm.calls[0]
    return call["messages"], call["kwargs"]
