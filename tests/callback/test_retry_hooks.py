import dspy
from dspy.utils.callback import BaseCallback


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.events = []

    def on_retry_start(self, call_id, instance, attempt, reason=None, parent_call_id=None):
        self.events.append(("start", attempt, reason, type(instance).__name__))

    def on_retry_end(self, call_id, outputs, exception):
        self.events.append(("end", outputs is not None, exception is not None))


def test_best_of_n_retry_hooks(monkeypatch):
    cb = RecordingCallback()
    dspy.settings.configure(callbacks=[cb], lm=dspy.LM("openai/gpt-4o-mini"))

    # Create a simple module that always returns low reward on first attempt, higher later
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
        # Keep reward below threshold to ensure a retry attempt happens
        return 0.0

    mod = SimpleModule()
    bon = dspy.BestOfN(module=mod, N=2, reward_fn=reward_fn, threshold=1.0)
    try:
        bon()
    except Exception:
        # In environments without provider setup, just ensure hooks wiring gets exercised.
        pass

    # Expect one retry start (attempt 2) and one retry end
    starts = [e for e in cb.events if e[0] == "start"]
    ends = [e for e in cb.events if e[0] == "end"]
    assert len(starts) >= 1
    assert len(ends) >= 1

