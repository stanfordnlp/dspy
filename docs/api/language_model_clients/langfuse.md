# Prompt Visiblity through Langfuse

We have now integrated Langfuse as one of the trackers.

## What is Langfuse?

Langfuse ([GitHub](https://github.com/langfuse/langfuse)) is an open-source LLM engineering and observability platform for model [tracing](https://langfuse.com/docs/tracing), [prompt management](https://langfuse.com/docs/prompts/get-started), and application [evaluation](https://langfuse.com/docs/scores/overview). Langfuse enables teams to collaboratively debug, analyze, and refine their LLM applications.

To get started, you can [self-host](https://langfuse.com/docs/deployment/self-host) Langfuse or use the Langfuse [cloud](https://cloud.langfuse.com/). 

### Install Langfuse.

```shell
pip install langfuse
```

Next, configure the following environment variables: `LANGFUSE_SECRET_KEY`、`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_HOST`. You can get your Langfuse API keys by navigating to **Project** -> **Settings** within Langfuse.

```python
import os
 
# Get keys for your project from the project settings page
# https://cloud.langfuse.com
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
```

If you are using **openai** or **azure_openai**, your LMs are configured.

For other LM providers, you need to configure the Langfuse tracker and call it manually.

### Example

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

Example trace in Langfuse: https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/5abece99-91bf-414e-a952-407ba8401c98

### Custom Tracker

We also provide  `BaseTracker`: simply inherit it and override the `call()` method
```python
# custom_tracker.py
from dsp.trackers.base import BaseTracker

class CustomTracker(BaseTracker):
    
    def call(self, *args, **kwargs):
        pass

```

### Resources

Have a look at the [DSPy cookbook](https://langfuse.com/docs/integrations/dspy) in the Langfuse docs for an end-to-end example.
