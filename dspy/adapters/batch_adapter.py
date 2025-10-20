import json
import logging

from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


class StandardLMResponse:
    """Normalizes provider batch formats to DSPy-compatible structure.
    
    NOTE: Currently uses OpenAI's response format as standard. Other providers 
    must convert their responses to match this structure (not optimal).
    """
    def __init__(self, response_dict: dict, custom_id: str):
        # Convert dict to object attributes for DSPy compatibility  
        for key, value in response_dict.items():
            setattr(self, key, value)
        # Keep custom_id for result routing to correct worker thread
        self.custom_id = custom_id


class BatchAdapter:
    """Simple adapter for OpenAI batch API format conversion."""
    
    def format(self, call_data_list: list[dict], lm) -> tuple[list[dict], str]:
        """Convert call data to batch request format."""
        if 'openai' in lm.model.lower() or '/' not in lm.model:
            return self.format_batch_openai(call_data_list, lm)
        else:
            raise NotImplementedError(f"Batch adapter for {lm.model} not implemented")
    
    def parse(self, completion: str, lm) -> list:
        """Parse provider batch results to standard format."""  
        if 'openai' in lm.model.lower() or '/' not in lm.model:
            return self.parse_batch_openai(completion)
        else:
            raise NotImplementedError(f"Batch adapter for {lm.model} not implemented")
    
    def format_batch_openai(self, call_data_list: list[dict], lm) -> tuple[list[dict], str]:
        """Format to OpenAI batch API structure."""
        batch_requests = []
        for call_data in call_data_list:
            messages = call_data.get('messages', [])
            # Extract model name from provider/model format if present
            model_name = lm.model.split("/")[-1] if "/" in lm.model else lm.model
            
            request = {
                'custom_id': call_data['id'],
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {
                    'model': model_name,
                    'messages': messages,
                    **lm.kwargs
                }
            }
            batch_requests.append(request)
        
        # Extract provider from model string for litellm routing
        provider = lm.model.split('/')[0] if '/' in lm.model else 'openai'
        return batch_requests, provider
    
    def parse_batch_openai(self, completion: str) -> list:
        """Parse OpenAI JSONL results to standard LM response format."""
        results = []
        
        for line in completion.strip().split("\n"):
            if line.strip():
                try:
                    result = json.loads(line)
                    response_body = result["response"]["body"]
                    
                    # Add "text" key for DSPy compatibility
                    if "choices" in response_body:
                        for choice in response_body["choices"]:
                            if "message" in choice and "content" in choice["message"]:
                                choice["text"] = choice["message"]["content"]
                    
                    results.append(StandardLMResponse(response_body, result.get('custom_id')))
                    
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="BatchAdapter",
                        signature=None,
                        lm_response=str(result) if 'result' in locals() else line,
                        message=f"Failed to parse OpenAI batch result: {e}"
                    )
        
        return results
