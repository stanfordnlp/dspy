import requests
import json

from typing import TypedDict, List, Literal

class Message(TypedDict):
    role: Literal["user"]
    content: str

class Completion(TypedDict):
    role: Literal["assistant"]
    content: str
    reward: int

class Input(TypedDict):
    messages: List[Message]

class GRPOTrainInstance(TypedDict):
    input: Input
    completions: List[Completion]

GRPOTrainBatch = List[GRPOTrainInstance]

# TODO(Lakshya, Noah): This currently assumes the server is already running locally
class ArborGRPOTrainer:
    def __init__(self, model, suffix, base_url='http://127.0.0.1:8000/v1'):
        # TODO(Lakshya): This is a temporary hack for testing
        if model == "openai/gpt-4o":
            model = "Qwen/Qwen2-0.5B-Instruct"
            print("Using Qwen/Qwen2-0.5B-Instruct model for GRPO training.")
        self.model = model
        self.base_url = base_url
        self.suffix = suffix

    def initialize(self):
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': self.model,
            'suffix': self.suffix,
        }
        response = requests.post(f"{self.base_url}/fine_tuning/grpo/initialize", headers=headers, json=data)
        assert response.status_code == 200, f"Failed to initialize GRPO: {response.text}"
        return response

    def run_grpo_step(self, batch: GRPOTrainBatch):
        url = f"{self.base_url}/fine_tuning/grpo/step"
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': self.model,
            'update_inference_model': True,
            "batch": batch
        }
        response = requests.post(url, headers=headers, json=data)
        return response

    def terminate_grpo(self):
        url = f"{self.base_url}/fine_tuning/grpo/terminate"
        headers = {'Content-Type': 'application/json'}
        data = {
            'status': 'success'
        }
        response = requests.post(url, headers=headers, json=data)
        return response
