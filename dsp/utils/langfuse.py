from typing import Optional
import dspy

import wonderwords
word_generator = wonderwords.RandomWord()

from langfuse.model import CreateTrace


def random_phrase_generator() -> str:
    return f"{word_generator.word()}-{word_generator.word()}"

def create_trace(name: Optional[str] = None) -> str:
    return dspy.settings.langfuse.trace(CreateTrace(name=name or random_phrase_generator()))