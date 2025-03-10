from ._common import BaseConversationalTask


class FireworksAIConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="fireworks-ai", base_url="https://api.fireworks.ai/inference")
