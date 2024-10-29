import os
import litellm
import dspy
import pytest
from dsp.trackers.langfuse_tracker import LangfuseTracker
import importlib.util

# langfuse docs: https://langfuse.com/docs/integrations/litellm/tracing

# set env variables
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

# Langfuse host
os.environ["LANGFUSE_HOST"] = ""

os.environ["OPENAI_API_KEY"] = ""
os.environ["COHERE_API_KEY"] = ""

# set callbacks
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)


def test_lm():
    result = lm("What is 2+2?", temperature=0.9)
    print(result)

    # The langfuse indicator collection has been completed
    # The following steps are optional (if you need to use the LangfuseTracker method, you need to instantiate it)
    if importlib.util.find_spec("langfuse"):
        dspy_langfuse = LangfuseTracker()

        quota = dspy_langfuse.get_all_quota()
        print(f"quota: {quota}")

        top_costs = quota.get_top_costs()
        print(f"top_costs: {top_costs}")

        top_latencies = quota.get_top_latencies()
        print(f"top_latencies: {top_latencies}")


if __name__ == '__main__':
    test_lm()




