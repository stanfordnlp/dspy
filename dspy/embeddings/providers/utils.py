import hashlib
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def generate_cache_key(text: str) -> str:
    """Generate a cache key for text."""
    return hashlib.md5(text.encode()).hexdigest()

def validate_api_response(response: Dict[str, Any], dimensions: Optional[int] = None) -> List[List[float]]:
    """Validate API response and extract embeddings."""
    if not isinstance(response, dict) or 'data' not in response:
        raise ValueError("Invalid API response format")
        
    embeddings = [data.get('embedding') for data in response['data']]
    
    if any(emb is None for emb in embeddings):
        raise ValueError("Missing embeddings in response")
        
    if dimensions and any(len(emb) != dimensions for emb in embeddings):
        raise ValueError(f"Incorrect embedding dimensions")
        
    return embeddings