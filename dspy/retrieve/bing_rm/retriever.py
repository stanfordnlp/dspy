from dspy.retrieve.bing_rm.base import BingClient
import dspy
from pydantic import BaseModel, Field
from typing import Optional

class BingSource(BaseModel):
    headline: Optional[str] = Field(None, description="The headline of the passage")
    url: str = Field(..., description="The URL of the passage")
    published: Optional[str] = Field(None, description="The date the passage was published")
    text: str = Field(..., description="The text of the passage")
    provider: Optional[str] = Field(None, description="The provider of the passage")

    def __str__(self):
        attributes = []
        
        if self.headline:
            attributes.append(f"Headline: {self.headline}")
        if self.published:
            attributes.append(f"Published: {self.published}")
        if self.url:
            attributes.append(f"URL: {self.url}")
        if self.provider:
            attributes.append(f"Provider: {self.provider}")
        
        # Join metadata with newlines, add text at the end
        metadata = "\n".join(attributes)
        return f"{metadata}\n\n{self.text}" if self.text else metadata

    @property
    def long_text(self) -> str:
        return str(
            self
        )

class BingRM(dspy.Retrieve):
    """A retrieval module that uses the Bing API to return the top passages for a given query.
    While there is free tier available, you will need to sign up for a Bing API key to use this module.
    https://portal.azure.com/#create/Microsoft.BingSearch

    Please set the environment variable `BING_API_KEY` to your Bing API key.

    Args:
        api (str): The Bing API to use. Defaults to "search". ["search", "news"]
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
    1. Retrieve via settings (recommended)
        ```python
        import dspy
        from dspy.retrieve.bing_rm import BingRM

        dspy.settings.configure(
            rm=BingRM()
        )

        bing = dspy.Retrieve(k=3)
        results = bing("Current interest rates in the USA")
        # Returns a list of strings
        ```
    2. Simple Retrieve; topk=3
        ```python
        import dspy
        from dspy.retrieve.bing_rm import BingRM

        bing = BingRM(api="news")
        results = bing("OpenAI o3 model", k=3)
        # Returns a list of BingSource objects, top 3; easily parsable
        ```
    """

    def __init__(self, api: str="search"):
        self.client = BingClient()
        self.set_api(api)

    def set_api(self, api: str):
        err = f"API {api} is currently unsupported"
        assert api in self.client.config.endpoints, err

        self.api = api
            
    def forward(self, query: str, k:int=3) -> dspy.Prediction:
        response = self.client.search(
            api=self.api
            , query=query
        )

        passages = [
            BingSource(
                **passage
            )
            for passage in 
            response.data[:k]
        ]

        return passages