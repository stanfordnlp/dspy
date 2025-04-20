import requests
import json

from typing import TypedDict, List, Literal, Union

class Message(TypedDict):
    role: Union[Literal["user"], Literal["assistant"], Literal["system"]]
    content: str

class Completion(TypedDict):
    role: Literal["assistant"]
    content: str

class GRPOGroupMember(TypedDict):
    messages: List[Message]
    completion: Completion
    reward: int

GRPOGroup = List[GRPOGroupMember]

GRPOBatch = List[GRPOGroup]

# TODO(Lakshya, Noah): This currently assumes the server is already running locally
class ArborGRPOTrainer:
    def __init__(self, lm, suffix, base_url='http://127.0.0.1:8000/v1'):
        # TODO(Lakshya): This is a temporary hack for testing
        model = lm.model
        assert model.startswith("openai/arbor:")
        model = model[len("openai/arbor:"):]
        print("Initializing GRPO train job for model:", model)
        self.model = model
        self.base_url = base_url
        self.suffix = suffix
        self.lm = lm

    def initialize(self):
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': self.model,
            'suffix': self.suffix,
        }
        response = requests.post(f"{self.base_url}/fine_tuning/grpo/initialize", headers=headers, json=data)
        assert response.status_code == 200, f"Failed to initialize GRPO: {response.text}"
        return response

    def run_grpo_step_one_group(self, batch: GRPOGroup):
        url = f"{self.base_url}/fine_tuning/grpo/step"
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': self.model,
            'update_inference_model': True,
            "batch": batch
        }
        response = requests.post(url, headers=headers, json=data).json()
        assert "status" in response and response['status'] == "success"
        assert "current_model" in response
        self.model = response["current_model"]
        self.lm.model = "openai/arbor:" + self.model
        return {
            "response": response,
            "current_model": self.model
        }

    def run_grpo_step(self, batch: GRPOBatch):
        responses = []
        for group in batch:
            response = self.run_grpo_step_one_group(group)['response']
            responses.append(response)
        return {
            "responses": responses,
            "current_model": self.model
        }

    def terminate_grpo(self):
        url = f"{self.base_url}/fine_tuning/grpo/terminate"
        headers = {'Content-Type': 'application/json'}
        data = {
            'status': 'success'
        }
        response = requests.post(url, headers=headers, json=data)
        return response
