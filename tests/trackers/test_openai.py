import os
import dspy
import pytest
from dsp.trackers.langfuse_tracker import LangfuseTracker
import importlib.util

# set env variables
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

turbo = dspy.OpenAI(api_key="your api key")


class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        answer = self.generate_answer(question=question)
        return answer


def test_langfuse():
    prompt = "what is the meaning of life?"
    result = turbo(prompt)
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
    test_langfuse()

