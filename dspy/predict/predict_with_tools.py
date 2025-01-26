from dspy.predict.predict import Predict
from dspy.primitives.prediction import Prediction


class PredictWithTools(Predict):
    """Predict with tool calling support.

    This class is a thin wrapper around the Predict class that explicit has tool calling support.
    """

    def __init__(self, signature, callbacks=None, tools=None, tool_choice="auto", **configs):
        """Initialize the PredictWithTools class.

        Args:
            signature (str): The signature of `PredictWithTools`.
            callbacks (list): The callbacks that are called before and after the Predict call.
            tools (list): The list of tools to use. For more details, please refer to
                https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
            tool_choice (str): The tool choice strategy, defaults to "auto". For more details, please refer to
                https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
            **configs: Additional configurations.
        """
        super().__init__(signature, callbacks, **configs)
        self.tools = tools
        self.tool_choice = tool_choice

    def __call__(self, tools=None, tool_choice="auto", **kwargs):
        """Call the PredictWithTools class.

        This method performs prediction using the configured language model, with the ability
        to invoke tools if specified. It can either return tool calling instructions or
        direct prediction results.

        Args:
            tools (list, optional): List of available tools for the LLM to choose from.
                If provided, overrides the `tools` set during initialization.
            tool_choice (str, optional): Strategy for tool selection, defaults to "auto".
                If provided, overrides the `tool_choice` set during initialization.
            **kwargs: Additional arguments passed to the underlying Predict class.

        Returns:
            dspy.Prediction: One of two types:
                - Tool calling scenario: Contains a single field 'tool_calls' with a list
                  of tool invocations and their arguments.
                - Direct prediction: Contains fields as specified by self.signature.
        """
        kwargs["tools"] = tools or self.tools
        kwargs["tool_choice"] = tool_choice or self.tool_choice

        pred = super().__call__(**kwargs).toDict()
        return Prediction(**pred)
