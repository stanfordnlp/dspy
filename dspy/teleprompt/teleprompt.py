import dspy

class Teleprompter:
    def __init__(self):
        if dspy.settings.langfuse.langfuse_client and not dspy.settings.langfuse.langfuse_in_context_call:
            dspy.settings.langfuse.create_new_trace(reset_in_context=True)
        pass
