import requests 
import aiohttp
import asyncio
from dspy.retrieve.bing_rm.config import BingConfig
from concurrent.futures import ThreadPoolExecutor
from pydantic import Field, BaseModel
from typing import (
    Optional
    , List
    , Coroutine
    , Dict
)
import logging

class BingResponse(BaseModel):
    """
    A response class for the Bing API.
    Contains many class methods to handle different types of responses.
    """
    status: int = Field(..., description="HTTP status code")
    data: list = Field(..., description="The results of the search")
    err_message: Optional[str] = Field(None, description="Any error message")
    
    @classmethod
    def from_search_json(
        cls
        , results: dict
        , status:int=200
    ) -> "BingResponse":
        pages = [ # kinda ugly but I promise it makes sense
            {
                "headline": page.get("name")
                , "url": page.get("url")
                , "published": page.get("datePublished")
                , "text": page.get("snippet")
            } for page in
            results.get("webPages", {}).get("value", [])
        ]

        return cls(
            status=status
            , data=pages
        )

    @classmethod
    def from_news_json(
        cls
        , results: dict
        , status:int=200
    ) -> "BingResponse":
        pages = [
            {
                "headline": page.get("name")
                , "url": page.get("url")
                , "published": page.get("datePublished")
                , "text": page.get("description")
                , "provider": page.get("provider")[0].get("name")
            } for page in 
            results.get("value", [])
        ]

        return cls(
            status=status
            , data=pages
        )

    @classmethod
    def from_search(cls, response: requests.Response) -> "BingResponse":
        response.raise_for_status()
        status = response.status_code
        results = response.json()
        pages = [ # kinda ugly but I promise it makes sense
            {
                "headline": page.get("name")
                , "url": page.get("url")
                , "published": page.get("datePublished")
                , "text": page.get("snippet")
                , "provider": None
            } for page in
            results.get("webPages", {}).get("value", [])
        ]

        return cls(
            status=status
            , data=pages
        )

    @classmethod
    def from_news(
        cls
        , response: requests.Response
    ) -> "BingResponse":
        response.raise_for_status()
        status = response.status_code
        results = response.json()
        pages = [
            {
                "headline": page.get("name")
                , "url": page.get("url")
                , "published": page.get("datePublished")
                , "text": page.get("description")
                , "provider": page.get("provider")[0].get("name")
            } for page in 
            results.get("value", [])
        ]

        return cls(
            status=status
            , data=pages
        )

    @classmethod
    def from_error(cls, status, err_message: Optional[str]) -> "BingResponse":
        return cls(
            status=status
            , data=[]
            , err_message=err_message
        )

    @property
    def warnings(self) -> bool:
        return self.status != 200
    
class BingClient(BaseModel):
    """
    A client for the Bing API.
    Supports searching for web pages and news articles.
    Current API support: 
        - News ("news")
        - Web ("search")

    Async support is also available, although it is not the default.
    Likewise, async may not be currently compatible with DSPy.
    """
    config: BingConfig = Field(
        BingConfig.from_env()
        , description="The configuration for the bing api"
    )
    method_matrix: dict = {
        "search": BingResponse.from_search
        , "web_search": BingResponse.from_search
        , "news": BingResponse.from_news
    }

    def generate_reqs(
        self
        , api: str
        , query: str
        , mkt: str="en-US"
        , pargs: dict={
            "sortBy": "Relevance"
        }
    ) -> Optional[dict]:
        params = {
            "q": query
            , "mkt": mkt
            , **pargs
        }
        headers = {
            'Ocp-Apim-Subscription-Key': self.config.key 
        }
        url = self.config.endpoint(api)
        if not url:
            return BingResponse(
                status=404
                , err_message=f"Invalid API / API not found: {api}"
            )

        return {
            "url": url
            , "params": params
            , "headers": headers
        }

    def search(self, api: str, query: str) -> BingResponse:
        """
        Searches Bing for a query using the specified api.

        Args:
            api (str): The api to use
            query (str): The query to search for
        Returns:
            BingResponse: The response from the search
        """
        reqs = self.generate_reqs(api, query)
        if isinstance(reqs, BingResponse):
            return reqs
        params = reqs.get("params")
        headers = reqs.get("headers")
        url = reqs.get("url")

        logging.info(f"GET @ {url}")
        response = requests.get(
            url
            , headers=headers
            , params=params
        )
        parser = self.method_matrix.get(api)
        
        try:
            return parser(
                response
            )
        except requests.exceptions.HTTPError as err:
            return BingResponse.from_error(
                err.response.status_code, str(err)
            )

    async def async_search(self, api: str, query: str) -> BingResponse:
        """
        Asynchronous version of search
        """
        reqs = self.generate_reqs(api, query)
        if isinstance(reqs, BingResponse):
            return reqs
        params = reqs.get("params")
        headers = reqs.get("headers")
        url = reqs.get("url")

        logging.info(f"GET @ {url}")
        
        parser = self.method_matrix.get(api)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                try:
                    status = response.status
                    if status != 200:
                        return BingResponse.from_error(
                            status=status
                            , err_message=f"Error {status}"
                        )
                    json = await response.json()
                    return BingResponse.from_search_json(
                        json
                    )
                except aiohttp.ClientResponseError as err:
                    err = f"Error "
                    return BingResponse.from_error(
                        err.status, str(err)
                    )

    def search_all(
        self
        , api: str
        , queries: List[str]
    ) -> Optional[Dict[str, BingResponse]]:
        """
        Searches for multiple queries in parallel.
        Be wary of rate limiting when using this. 

        Args:
            api (str): The api to use
            queries (List[str]): The queries to search for
        Returns:
            Dict[str, BingResponse]: A dictionary of responses; keys are queries
        """

        async def async_search_all():
            tasks = [self.async_search(api, query) for query in queries]
            return await asyncio.gather(*tasks)

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_search_all())

        try:
            with ThreadPoolExecutor() as executor:
                results = executor.submit(run_in_thread).result()
                return dict(zip(queries, results))
        except Exception as e:
            logging.error(f"Error in search_all: {str(e)}")
            return None