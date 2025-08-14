import dspy
from dspy.utils.callback import BaseCallback


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.events = []

    def on_retry_start(self, call_id, instance, attempt, reason=None, parent_call_id=None):
        self.events.append(("start", attempt, reason, type(instance).__name__))

    def on_retry_end(self, call_id, outputs, exception):
        self.events.append(("end", outputs is not None, exception is not None))


def test_refine_retry_hooks(monkeypatch):
    cb = RecordingCallback()
    dspy.settings.configure(callbacks=[cb], lm=dspy.LM("openai/gpt-4o-mini"))

    class SimpleModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return dspy.Prediction(answer="x")

        def get_lm(self):
            return dspy.settings.lm

        def named_predictors(self):
            return []

    def reward_fn(args, pred):
        return 0.0

    mod = SimpleModule()
    ref = dspy.Refine(module=mod, N=2, reward_fn=reward_fn, threshold=1.0)

    ref()
    
    starts = [e for e in cb.events if e[0] == "start"]
    ends = [e for e in cb.events if e[0] == "end"]
    assert len(starts) >= 1
    assert len(ends) >= 1
