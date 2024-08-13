# How to make the prompt visible

## Langfuse
We have now integrated langfuse as one of the tracker.

How to configure langfuse?[langfuse details](https://langfuse.com/docs/deployment/self-host) .

### install langfuse.

```shell
pip install langfuse
```

After that, you will get three environment variables, they are `LANGFUSE_SECRET_KEY`、`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_HOST`.

Just write the environment variables and langfuse will automatically read them.

If you are using **openai** or **azure_openai**, then your preparations are now complete.

for other modules, you need to manually configure and call.

### example

```python
import dspy
from dsp.trackers.langfuse_tracker import LangfuseTracker
# e.g:
# Assuming the environment variables have been set
langfuse = LangfuseTracker()
turbo = dspy.OllamaLocal()
dspy.settings.configure(lm=turbo)

completions =  turbo("Hi,how's it going today？")
turbo.tracker_call(tracker=langfuse)
```

## Custom Tracker

We provide  `BaseTracker`, just inherit it and override the call() method
```python
# custom_tracker.py
from dsp.trackers.base import BaseTracker

class CustomTracker(BaseTracker):
    
    def call(self, *args, **kwargs):
        pass

```
