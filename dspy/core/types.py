"""Migration notice for the retired DSPy 3.3 experimental LM types."""

raise ImportError(
    "DSPy's 3.3 experimental LM types have been removed. "
    "Import Request, Response, Message, Config and content parts from dspy.lm15 instead. "
    "These are different objects, not drop-in aliases: system instructions use Request.system, "
    "tool arguments use ToolCallPart.input, and each Response contains one Message. "
    "lm('text') remains a convenience; lm(Request) returns a Response. "
    "See https://dspy.ai/community/normalized-lm-api-migration/."
)
