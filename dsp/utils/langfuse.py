import os
from typing import Optional, Any

import wonderwords
word_generator = wonderwords.RandomWord()

from langfuse.model import CreateTrace
from langfuse import Langfuse
    

def random_phrase_generator() -> str:
    return f"{word_generator.word()}-{word_generator.word()}"

class LangfuseHandler:
    langfuse_client: Any
    langfuse_trace: Optional[Any]
    langfuse_in_context_call: bool
    
    def __init__(self):
        super().__init__()
        
        self.langfuse_client = Langfuse(debug=False) if os.environ.get("LANGFUSE_PUBLIC_KEY") else None
        self.langfuse_trace = None
        self.langfuse_in_context_call = False
    
    def create_new_trace(self, name: Optional[str] = None, reset_in_context: bool = False) -> None:
        if reset_in_context:
            self.langfuse_in_context_call = False
        self.langfuse_trace = self.langfuse_client.trace(CreateTrace(name=name or random_phrase_generator()))
