import json
import logging
import os
import re
from typing import Any, Literal

import requests

from .base_lm import BaseLM

logger = logging.getLogger(__name__)


class LM(BaseLM):
    """
    A simplified language model supporting AWS Bedrock (default) and OpenAI API requests.
    """

    def __init__(
        self,
        model: str = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        api_key: str = None,
        api_base: str = None,
        region: str = "us-east-1",
        **kwargs,
    ):
        """
        Create a new language model instance.

        Args:
            model: The model to use (defaults to Claude 3 Sonnet on Bedrock)
            model_type: The type of the model, either "chat" or "text"
            temperature: The sampling temperature to use
            max_tokens: The maximum number of tokens to generate
            api_key: API key (only needed for OpenAI models)
            api_base: API base URL (for custom endpoints)
            region: AWS region for Bedrock (default: us-east-1)
        """
        self.model = model
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base
        self.region = region
        self.kwargs = kwargs
        
        # Check if this is a Bedrock model (default) or OpenAI model
        if model.startswith("bedrock/") or not model.startswith(("gpt-", "openai/")):
            self.provider = "bedrock"
            # For Bedrock, we don't need an API key - it uses AWS credentials
            self.api_key = None
            # If no bedrock/ prefix, add it
            if not model.startswith("bedrock/"):
                self.model = f"bedrock/{model}"
        else:
            self.provider = "openai"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required for OpenAI models. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    def forward(self, prompt=None, messages=None, **kwargs):
        """Forward pass for the language model."""
        if self.provider == "bedrock":
            return self._bedrock_completion(prompt, messages, **kwargs)
        elif self.model_type == "chat":
            return self._chat_completion(prompt, messages, **kwargs)
        else:
            return self._text_completion(prompt, **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        """Async forward pass for the language model."""
        # For simplicity, just call the sync version
        return self.forward(prompt, messages, **kwargs)

    def _chat_completion(self, prompt=None, messages=None, **kwargs):
        """Make a chat completion request."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **kwargs
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
        return response.json()

    def _text_completion(self, prompt, **kwargs):
        """Make a text completion request."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **kwargs
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
        return response.json()

    def _bedrock_completion(self, prompt=None, messages=None, **kwargs):
        """Make a Bedrock completion request."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for Bedrock support. Install with: pip install boto3")
        
        # Initialize Bedrock client
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region
        )
        
        # Prepare the request body for Claude
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Filter out system messages and convert to Bedrock format
        claude_messages = []
        system_content = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Collect system content to prepend to first user message
                system_content.append(msg["content"])
            elif msg["role"] in ["user", "assistant"]:
                content = msg["content"]
                
                # If this is the first user message and we have system content, prepend it
                if msg["role"] == "user" and system_content and not claude_messages:
                    content = "\n\n".join(system_content) + "\n\n" + content
                
                claude_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": content}]
                })
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": claude_messages
        }
        
        # Extract model ID from the full model name
        model_id = self.model.replace("bedrock/", "")
        
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response.get('body').read())
            
            # Convert Bedrock response to OpenAI-like format for compatibility
            if 'content' in response_body and len(response_body['content']) > 0:
                content = response_body['content'][0].get('text', '')
                return {
                    'choices': [{
                        'message': {
                            'content': content,
                            'role': 'assistant'
                        },
                        'finish_reason': 'stop'
                    }],
                    'model': self.model,
                    'usage': response_body.get('usage', {})
                }
            else:
                raise Exception("Invalid response format from Bedrock")
            
        except Exception as e:
            raise Exception(f"Bedrock API error: {str(e)}")

    def __call__(self, prompt=None, messages=None, **kwargs):
        """Make the LM callable for direct use."""
        return self.forward(prompt, messages, **kwargs)

    def acall(self, prompt=None, messages=None, **kwargs):
        """Async call method."""
        return self.aforward(prompt, messages, **kwargs)

    def __repr__(self):
        return f"LM(model='{self.model}', provider='{self.provider}', temperature={self.temperature})" 