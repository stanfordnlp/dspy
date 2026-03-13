# Bing Retriever
The Bing Retriever leverages the Bing API to perform web based searches as a DSPy compatible retriever. 

Various APIs exist within Azure's Bing Search, although only the following are (currently) supported:
- `search` (a.k.a `web_search`); default
- `news`

## Setup
The Bing Retriever requires an API key from Azure. Various tiers for the API exist, including a free tier. 

1. Get an [API key](https://portal.azure.com/#create/Microsoft.BingSearch)
2. Set the API as an environment variable: `BING_API_KEY`
    ```bash
    export BING_API_KEY='your_api_key_here'
    ```

## Example Usage
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
2. Simple News Retrieve; topk=3
    ```python
    import dspy
    from dspy.retrieve.bing_rm import BingRM

    bing = BingRM(api="news")
    results = bing("OpenAI o3 model", k=3)
    # Returns a list of BingSource objects, top 3; easily parsable
    ```

### Parsing BingSource Objects
BingSource objects are used to format results from Bing. When retrieving from Bing via the `dspy.Retrieve` function, results are returned as strings. When retrieving directly from `BingRM`, results are returned as BingSource objects. BingSource objects also contain structured metadata specific to the article retrieved from Bing. This metadata is formatted into the string results which are returned from `dspy.Retrieve`.
BingSource objects can be easily cast into strings. 
For example (beginning where example 2 ends):
```python
str_results = [
    str(result) #auto casts and formats to string
    for result in results
]
```

## Additional Notes
- Reranking is done by default by the Bing API itself. 
- For more information on the underlying API, please see its [documentation](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/). 
- For more information on the internals of the API itself, please see [base.py](./base) and [config.py](./config.py)
- An SDK does exist for Bing Search, however it is not longer maintained [according to PyPi](https://pypi.org/project/azure-cognitiveservices-search-websearch/). For this reason, the (custom) modules referenced above are used to interact with the API.
- The underlying classes used to query Bing do support async, but because DSPy does not currently suppport async retrieval via their retriever modules this has not been integrated into the BingRM class.
    - For more information regarding async search with Bing, see the methods `async_search` and `search_all` within `BingClient`.
