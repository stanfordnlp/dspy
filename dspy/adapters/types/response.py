"""Response type for adapters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AdapterResponse:
    """Response from an adapter."""
    
    text: str
    logprobs: Optional[dict] = None 