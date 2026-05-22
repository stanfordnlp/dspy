import dspy


class StatefulBaseLM(dspy.BaseLM):
    def __init__(self, model="test/base", *, label="base", **kwargs):
        super().__init__(model=model, cache=False, **kwargs)
        self.label = label

    def dump_state(self):
        return {"model": self.model, "label": self.label}

    def forward(self, prompt=None, messages=None, **kwargs):
        raise NotImplementedError


class StatefulLanguageModel(dspy.BaseLM):
    def __init__(self, model="test/language-model", *, label="normalized", cache=False, **kwargs):
        super().__init__(model=model, cache=cache, **kwargs)
        self.label = label

    def dump_state(self):
        state = super().dump_state()
        state["label"] = self.label
        return state

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text(self.label, model=request.model)


def test_predict_load_state_round_trips_concrete_baselm_subclass():
    original = dspy.Predict("q -> a")
    original.lm = StatefulBaseLM(label="custom-base")

    loaded = dspy.Predict("q -> a").load_state(original.dump_state())

    assert isinstance(loaded.lm, StatefulBaseLM)
    assert loaded.lm.model == "test/base"
    assert loaded.lm.label == "custom-base"


def test_predict_load_state_round_trips_concrete_language_model_subclass():
    original = dspy.Predict("q -> a")
    original.lm = StatefulLanguageModel(label="custom-normalized", temperature=0.2)

    loaded = dspy.Predict("q -> a").load_state(original.dump_state())

    assert isinstance(loaded.lm, StatefulLanguageModel)
    assert loaded.lm.model == "test/language-model"
    assert loaded.lm.label == "custom-normalized"
    assert loaded.lm.kwargs["temperature"] == 0.2
    assert loaded.lm("hello").text == "custom-normalized"


def test_predict_load_state_without_lm_class_marker_uses_legacy_lm_for_backward_compatibility():
    predict = dspy.Predict("q -> a")
    state = predict.dump_state()
    state["lm"] = {"model": "openai/gpt-4o-mini", "model_type": "text", "cache": False}

    loaded = dspy.Predict("q -> a").load_state(state)

    assert isinstance(loaded.lm, dspy.clients.lm.LM)
    assert loaded.lm.model == "openai/gpt-4o-mini"
    assert loaded.lm.model_type == "text"
    assert loaded.lm.cache is False
