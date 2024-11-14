import os
import warnings
from typing import Literal, Optional, Union, List
from tavily import TavilyClient, MissingAPIKeyError, InvalidAPIKeyError, UsageLimitExceededError, BadRequestError

import dspy
from dsp.utils import dotdict


class TavilyRM(dspy.Retrieve):
    """A DSPy retriever module that uses Tavily's Search API to perform web searches and return relevant content.

    This retriever supports both basic and advanced search modes, general and news topics, and can include 
    both synthesized answers and raw content from web pages. Results are returned with source URLs for 
    reference.

    Args:
        api_key (Optional[str]): Tavily API key. If not provided, will look for TAVILY_API_KEY in environment 
            variables. Defaults to None.
        k (int): Maximum number of results to return (including the answer if include_answer=True). 
            Defaults to 5.
        search_depth (Literal["basic", "advanced"]): The depth of search to perform. For concrete definitions of each
            please review the Tavily documentation. Defaults to "basic".
        topic (Literal["general", "news"]): Type of search to perform. "general" for regular web search, 
            "news" for news articles. Defaults to "general".
        days (int): For news searches, the maximum age of articles in days. Only used when topic="news". 
            Defaults to 3.
        include_answer (bool): Whether to include Tavily's synthesized answer as the first result. The 
            answer will include references to the source URLs. Defaults to True.
        include_raw_content (bool): Whether to return the full raw content of pages instead of snippets. 
            When True, raw_content replaces regular content where available. Defaults to False.
        include_images (bool): Whether to include image results. Note: Even if images are returned by Tavily, 
            they are not included in the DSPy retriever output. Use the Tavily API directly for image results. 
            Defaults to False.
        include_image_descriptions (bool): Whether to include descriptions for returned images. Only used 
            when include_images=True. See above note about images. Defaults to False.
        include_domains (Optional[List[str]]): List of domains to restrict the search to. Defaults to None.
        exclude_domains (Optional[List[str]]): List of domains to exclude from the search. Defaults to None.
        include_urls (bool): Whether to append source URLs to content and answers. Defaults to False.

    Returns:
        dspy.Prediction: A list of dotdict objects, each containing a 'long_text' field with either:
            - A synthesized answer with source URLs (if include_answer=True and include_urls=True)
            - Content snippets with their source URLs (if include_urls=True)
            - Raw content with source URLs (if include_raw_content=True and include_urls=True)

    Example:
        ```python
        import dspy
        from dspy.retrieve import TavilyRM

        # Initialize with default settings
        retriever = TavilyRM(api_key="your-api-key")

        # Or customize the behavior
        retriever = TavilyRM(
            api_key="your-api-key",
            k=3,
            search_depth="advanced",
            topic="news",
            days=7,
            include_answer=True,
            include_domains=["example.com", "trusteddomain.com"]
        )

        # Use as the default retriever
        dspy.settings.configure(rm=retriever)

        # Or use directly
        results = retriever("What are the latest developments in AI?")
        ```

    Note:
        The retriever requires a valid Tavily API key. The API key can be obtained from 
        https://tavily.com/. For detailed API documentation, see:
        https://docs.tavily.com/docs/python-sdk/tavily-search/api-reference

    Author:
        @RamXX (Ramiro Salas)
    """

    def __init__(
        self,
        api_key: Optional[str] = os.getenv("TAVILY_API_KEY"),
        k: int = 5, # default for max_results
        search_depth: Literal["basic", "advanced"] = "basic", 
        topic: Literal["general", "news"] = "general", 
        days: int = 3, # Only taken into account if topic="news". Irrelevant otherwise.
        include_answer=True, # Tavily defaults to False here
        include_raw_content=False, 
        include_images=False,
        include_image_descriptions=False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_urls: bool = False,
    ):
        super().__init__(k=k)

        # Data validation
        if not api_key and not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError('You must supply `api_key` or set environment variable "TAVILY_API_KEY"')

        if search_depth not in ("basic", "advanced"):
            raise ValueError('`search_depth` must be either "basic" or "advanced"')
        
        if topic not in ("general", "news"):
            raise ValueError('`topic` must be either "general" or "news"')

        if include_images:
            warnings.warn(
                "Note: Even if images are returned by Tavily, they are not included in the DSPy "
                "retriever output. Use the Tavily API directly for image results.",
                UserWarning
            )

        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.max_results = k
        self.search_depth = search_depth
        self.topic = topic
        self.days = days
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.include_image_descriptions = include_image_descriptions
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.include_urls = include_urls


    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None) -> dspy.Prediction:
        k = k if k is not None else self.max_results

        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries

        client = TavilyClient(api_key=self.api_key)

        all_docs = []
        for query in queries:
            try:
                response = client.search(query=query, 
                         search_depth=self.search_depth,  # type: ignore
                         topic=self.topic,  # type: ignore
                         days=self.days, 
                         max_results=k, 
                         include_domains=self.include_domains,  # type: ignore
                         exclude_domains=self.exclude_domains,  # type: ignore
                         include_answer=self.include_answer, 
                         include_raw_content=self.include_raw_content, 
                         include_images=self.include_images,
                )
                
                query_docs = []
                if "results" in response:
                    for result in response["results"][:k]:
                        if self.include_raw_content and "raw_content" in result and result["raw_content"]:
                            content = result["raw_content"]
                        else:
                            content = result["content"]
                        
                        # Append URL as reference if include_urls is True
                        if self.include_urls and "url" in result:
                            content = f"{content}\nReference: {result['url']}"
                            
                        query_docs.append(content)
                
                # If include_answer is True and we got an answer, add it as the first document for this query
                if self.include_answer and "answer" in response and response["answer"]:
                    answer_text = response["answer"]
                    # If there are results URLs and include_urls is True, add them as references to the answer
                    if self.include_urls and "results" in response and response["results"]:
                        urls = [f"- {r['url']}" for r in response["results"][:3] if "url" in r]
                        if urls:
                            answer_text = f"{answer_text}\n\nReferences:\n" + "\n".join(urls)
                    query_docs.insert(0, answer_text)
                    
                query_docs = query_docs[:k] # Ensure we don't exceed k documents for this query
                
                all_docs.extend(query_docs)
                
            except MissingAPIKeyError:
                raise MissingAPIKeyError
            except InvalidAPIKeyError:
                raise InvalidAPIKeyError
            except UsageLimitExceededError:
                raise UsageLimitExceededError("Tavily usage limit exceeded")
            except BadRequestError:
                raise BadRequestError("Bad request made against the Tavily API")

        # Return all documents wrapped in dotdict with long_text field
        return [dotdict({"long_text": doc}) for doc in all_docs[:k]] # type: ignore