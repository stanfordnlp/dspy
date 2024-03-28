import os
import pytest

from dspy import Example
from dsp import OpenAIVectorizer


@pytest.mark.skipif(os.getenv('OPENAI_API_KEY') is None, reason="Skipping this test because OPENAI_API_KEY is not set")
def test__call__():
    vectorizer = OpenAIVectorizer()
    input_examples = [
        example.with_inputs('question')
        for example in [
            Example(question="Why?", answer="Who knows?"),
            Example(question="Why not?", answer="I don't know"),
        ]
    ]
    vectorizer(inp_examples=input_examples)


@pytest.mark.skipif(os.getenv('OPENAI_API_KEY') is None, reason="Skipping this test because OPENAI_API_KEY is not set")
def test__call__with_str():
    vectorizer = OpenAIVectorizer()
    vectorizer("Hello world!")
