import dspy
from dspy.utils.callback import BaseCallback


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.events = []

    def on_retry_start(self, call_id, instance, attempt, reason=None, parent_call_id=None):
        self.events.append(("start", attempt, reason, type(instance).__name__))

    def on_retry_end(self, call_id, outputs, exception):
        self.events.append(("end", outputs is not None, exception is not None))


def test_program_of_thought_retry_hooks(monkeypatch):
    cb = RecordingCallback()
    dspy.settings.configure(callbacks=[cb], lm=dspy.LM("openai/gpt-4o-mini"))

    # Use a tiny signature; interpreter may raise, so we focus on hook wiring
    pot = dspy.ProgramOfThought("question -> answer", max_iters=2)
    
    pot(question="what is 1+1?")

    starts = [e for e in cb.events if e[0] == "start"]
    assert len(starts) >= 1
