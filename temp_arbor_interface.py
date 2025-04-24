import requests
import json

from typing import TypedDict, List, Literal, Union

import dspy


class Message(TypedDict):
    role: Union[Literal["user"], Literal["assistant"], Literal["system"]]
    content: str

class MessageAssistant(TypedDict):
    role: Literal["assistant"]
    content: str

class GRPOChatData(TypedDict):
    messages: List[Message]
    completion: MessageAssistant
    reward: int

GRPOGroup = List[GRPOChatData]


class ReinforceInterface:
    DEFAULT_TRAIN_KWARGS = {
        "update_interval": 25,
        "temperature": 0.9,
        "beta": 0.04,
    }

    def __init__(self):
        if not hasattr(dspy.settings, "arbor_api_base"):
            raise ValueError("The Arbor base URL is not set in dspy settings. Set it using dspy.settings.arbor_base_url, e.g., dspy.settings.arbor_base_url = 'http://localhost:8000'")
        self.api_base = dspy.settings.arbor_api_base
        self.api_key = "local"
    
    def from_arbor_model(self, model):
        if model.startswith("openai/arbor:"):
            return model[len("openai/arbor:"):]
        if model.startswith("arbor:"):
            return model[len("arbor:"):]
        raise ValueError(f"Model {model} is not a valid Arbor model.") 

    def to_arbor_model(self, model):
        return "openai/arbor:" + model

    def initialize(self, model, train_kwargs=None):
        # TODO(GRPO Team):
        # * It would be nice if the server gave us a model ID to use
        train_kwargs = {} if train_kwargs is None else train_kwargs
        num_generations = train_kwargs.get("num_generations")  # The teleprompter must ensure this is set
        update_interval = train_kwargs.get("update_interval", self.DEFAULT_TRAIN_KWARGS["update_interval"])
        temperature = train_kwargs.get("temperature", self.DEFAULT_TRAIN_KWARGS["temperature"])
        beta = train_kwargs.get("beta", self.DEFAULT_TRAIN_KWARGS["beta"])

        suffix = "dspy"
        finetune_model = self.from_arbor_model(model)
        data = {
            'model': finetune_model,
            'suffix': suffix,
            'num_generations': num_generations,
            'update_interval': update_interval,
            'temperature': temperature,
            'beta': beta,
        }
        url = f"{self.api_base}fine_tuning/grpo/initialize"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            url=url,
            headers=headers,
            json=data
        )
        assert response.status_code == 200, f"Failed to initialize GRPO: {response}"
        current_model = self.to_arbor_model(finetune_model)  # TODO: To be updated
        return current_model

    def _run_grpo_step_one_group(self, model, train_group: GRPOGroup, train_kwargs=None):
        train_kwargs = {} if train_kwargs is None else train_kwargs
        finetune_model = self.from_arbor_model(model)
        data = {
            'model': finetune_model,
            'update_inference_model': True,
            'batch': train_group
        }
        url = f"{self.api_base}fine_tuning/grpo/step"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, f"Failed to run a GRPO step: {response.text}"
        response = response.json()
        assert "current_model" in response, f"Response does not contain the next model ID to be used: {response}"
        current_model = response["current_model"]
        current_model = self.to_arbor_model(current_model)
        return current_model
    
    def step(self, model, train_data: List[GRPOGroup]):
        current_model = model
        for group in train_data:
            current_model = self._run_grpo_step_one_group(current_model, group)
        return current_model

    def terminate(self):
        # TODO(GRPO Team):
        # * Update after the server starts returning the saved model ID
        # * Why do we need to send a payload?
        data = {
            'status': 'success'
        }
        url = f"{self.api_base}fine_tuning/grpo/terminate"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, f"Failed to run a GRPO step: {response.text}"
        # response = response.json()  # TODO: To be updated
        # current_model = response["current_model"]
        # current_model = self.to_arbor_model(current_model)
        # return current_model
