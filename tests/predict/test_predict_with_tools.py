import os

import pytest

import dspy
from dspy.predict.predict_with_tools import PredictWithTools


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OpenAI API key is not set")
def test_basic_predict_with_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        predict = PredictWithTools("question -> answer", tools=tools)
        outputs = predict(question="what's the weather in Paris?", tools=tools)
        assert "tool_calls" in outputs