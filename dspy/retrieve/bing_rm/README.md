# Bing Retriever
The Bing Retriever leverages the Bing API to perform web based searches as a DSPy compatible retriever. 

Various APIs exist within Azure's Bing Search, although only the following is (currently) supported:
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

    dspy.settings.configure(rm=BingRM())

    bing = dspy.Retrieve(k=3)
    results = bing("Current interest rates in the USA")
    # Returns a list of strings
    ```
2. Simple Retrieve; k=3
    ```python
    import dspy
    from dspy.retrieve.bing_rm import BingRM

    bing = BingRM()
    results = bing("Current interest rates in the USA", k=3)
    # Returns a list of BingSource objects, top 3; easily parsable
    ```

## Additional Notes
- Reranking is done by default by the Bing API itself. 
- For more information on the underlying API, please see its [documentation](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/). 
- For more information on the internals of the API itself, please see [base.py](./base) and [config.py](./config.py)
