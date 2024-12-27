from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os

class BingConfig(BaseModel):
    """
    Config setup for Bing API.
    This class will automatically pull the API key and base URI from the environment
    or they can be passed manually. For most use cases, the base_uri is not necessary
    to specify as it should not change for most users.

    Args:
        key (str): The Bing API key.
        base_uri (str, Optional): The base URI for the Bing API.
        endpoints (dict, Optional): A dictionary of Bing API endpoints.
    """
    key: str = Field(..., description="BING_API_KEY")
    base_uri: str = Field(..., description="BING_BASE_URI")
    endpoints: dict = {
        "web_search": "v7.0/search", # Alias for search
        "search": "v7.0/search",
        "news": "v7.0/news",
    }

    @classmethod
    def from_env(cls) -> "BingConfig":
        key = os.getenv("BING_API_KEY")
        base_uri = os.getenv("BING_BASE_URI", "https://api.bing.microsoft.com")

        for req in [key, base_uri]:
            assert req, f"Key/base url missing; key: {key}, base_url: {base_uri}"
        
        return cls(
            key=key
            , base_uri=base_uri
        )
    
    def endpoint(self, endpoint: str) -> Optional[str]:
        if endpoint not in self.endpoints:
            wrn = f"Endpoint {endpoint} is currently unsupported"
            logging.warning(wrn)
            return None
        
        path = os.path.join(
            self.base_uri
            , self.endpoints[endpoint]
        )

        return path
        
