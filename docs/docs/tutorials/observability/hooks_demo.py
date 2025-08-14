# %% [markdown]
# This notebook-like script demonstrates how to use DSPy callback hooks for
# raw provider responses, raw completion text before parsing, and retry events
# in multi-try modules. It runs offline with a dummy LM for reliability.

# %%
import types
import dspy
from dspy.utils.callback import BaseCallback

# %% [markdown]
# We define a callback that records events from three key hook families. It
# captures the raw provider response object, the raw completion text right
# before adapter parsing, and retry start/end signals with attempt metadata.

# %%
class DemoCallback(BaseCallback):
    def __init__(self):
        self.events = []

    def on_lm_raw_response(self, call_id, instance, response):
        content = getattr(response.choices[0].message, "content", "")
        self.events.append(("lm.raw", getattr(instance, "model", None), len(content)))
        print({"event": "lm.raw", "model": getattr(instance, "model", None)})

    def on_adapter_parse_start(self, call_id, instance, inputs):
        c = inputs.get("completion", "") if isinstance(inputs, dict) else ""
        self.events.append(("adapter.parse.start", len(c)))
        print({"event": "adapter.parse.start", "chars": len(c)})

    def on_retry_start(self, call_id, instance, attempt, reason=None, parent_call_id=None):
        self.events.append(("retry.start", attempt, reason))
        print({"event": "retry.start", "attempt": attempt, "reason": reason})

    def on_retry_end(self, call_id, outputs, exception):
        self.events.append(("retry.end", exception is None))
        print({"event": "retry.end", "success": exception is None})

# %% [markdown]
# Next, we define a tiny provider-compatible response and a dummy LM. The LM
# returns a single chat message containing a JSON string so the JSON adapter
# can parse to the expected output field format reliably in offline runs.

# %%
class DummyResponse:
    def __init__(self, text):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1}
        self.model = "dummy-model"

class DummyLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        return DummyResponse(text='{"a": "hello"}')

# %% [markdown]
# We configure DSPy to use the dummy LM and the JSON adapter to ensure that the
# raw completion string is valid JSON and maps to the signature output fields.
# We also register our callback to observe the hook events during execution.

# %%
dspy.configure(
    lm=DummyLM(model="dummy-model"),
    adapter=dspy.JSONAdapter(),
    callbacks=[DemoCallback()],
)

# %% [markdown]
# The raw provider response hook fires as soon as the LM returns, before any
# processing or parsing is performed. Calling the LM directly triggers it.

# %%
lm = dspy.settings.lm
_ = lm(prompt="hi")

# %% [markdown]
# The raw completion hook is emitted right before adapters parse the LM output.
# Running a simple predictor produces a valid JSON output that can be parsed.

# %%
pred = dspy.Predict("q->a")
res = pred(q="hello")
print(res)

# %% [markdown]
# Retry hooks are emitted by multi-try modules such as BestOfN and Refine. We
# force retries by using a reward function that always returns a low score.

# %%
mod = dspy.Predict("q->a")
bon = dspy.BestOfN(module=mod, N=2, reward_fn=lambda args, pr: 0.0, threshold=1.0)
try:
    _ = bon(q="force")
except Exception:
    pass