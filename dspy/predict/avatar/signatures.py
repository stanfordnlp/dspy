import dspy

from dspy.predict.avatar.models import Action


class Actor(dspy.Signature):
    """You will be given `Tools` which will have tool name, input values, and description. Given the user query, your task is to decide which tool to use and what input values to provide.

You will be given `Actions` which will have tool name and input values. You will respond with the output of the tool."""

    goal: str = dspy.InputField(
        prefix="Goal:",
        desc="Task to be accomplished.",
    )
    tools: str = dspy.InputField(
        prefix="Tools:",
        desc="list of tools to use",
        format=lambda x: str([str(tool) for tool in x]),
    )
    action_1: Action = dspy.OutputField(
        prefix="Action 1:",
        desc="1st action to taken",
    )