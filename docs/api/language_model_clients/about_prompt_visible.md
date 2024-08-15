# Prompt Visiblity through Langfuse

We have now integrated Langfuse as one of the trackers.

How to configure Langfuse?[Langfuse details](https://langfuse.com/docs/deployment/self-host) .

### Install Langfuse.

```shell
pip install langfuse
```

Additionally, configure the following environment variables: `LANGFUSE_SECRET_KEY`、`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_HOST`.

If you are using **openai** or **azure_openai**, your LMs are configured.

For other LM providers, you need to configure the Langfuse tracker and call it manually.

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

We also provide  `BaseTracker`: simply inherit it and override the `call()` method
```python
# custom_tracker.py
from dsp.trackers.base import BaseTracker

class CustomTracker(BaseTracker):
    
    def call(self, *args, **kwargs):
        pass

```
