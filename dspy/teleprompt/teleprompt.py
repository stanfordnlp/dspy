import tqdm
import random
import dsp

from dspy.evaluate.evaluate import Evaluate
from langfuse.model import CreateTrace
import uuid
import dspy
import wonderwords
r = wonderwords.RandomWord()

class Teleprompter:
    def __init__(self):
        if dspy.settings.langfuse:
            self.trace_id = uuid.uuid4().hex
            dspy.settings.langfuse_trace = dspy.settings.langfuse.trace(CreateTrace(name=f"{r.word()}-{r.word()}"))
        pass
