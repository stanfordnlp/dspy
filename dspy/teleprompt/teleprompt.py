import dspy
from dsp.utils.langfuse import create_trace

class Teleprompter:
    def __init__(self):
        if dspy.settings.langfuse:
            dspy.settings.langfuse_trace = create_trace()
        pass
