---
sidebar_position: 3
---

# retrieve.AzureCognitiveSearch

### Constructor

The constructor initializes an instance of the `AzureCognitiveSearch` class and sets up parameters for sending queries and retreiving results with the Azure Cognitive Search server.

```python
class AzureCognitiveSearch:
    def __init__(
        self,
        search_service_name: str,
        search_api_key: str,
        search_index_name: str,
        field_text: str,
        field_score: str, # required field to map with "score" field in dsp framework
    ):
```

**Parameters:**

- `search_service_name` (_str_): Name of Azure Cognitive Search server.
- `search_api_key` (_str_): API Authentication token for accessing Azure Cognitive Search server.
- `search_index_name` (_str_): Name of search index in the Azure Cognitive Search server.
- `field_text` (_str_): Field name that maps to DSP "content" field.
- `field_score` (_str_): Field name that maps to DSP "score" field.

### Methods

Refer to [ColBERTv2](/api/retrieval_model_clients/ColBERTv2) documentation. Keep in mind there is no `simplify` flag for AzureCognitiveSearch.

AzureCognitiveSearch supports sending queries and processing the received results, mapping content and scores to a correct format for the Azure Cognitive Search server.

### Deprecation Notice

This module is scheduled for removal in future releases. Please use the AzureAISearchRM class from dspy.retrieve.azureaisearch_rm instead.For more information, refer to the updated documentation(docs/docs/deep-dive/retrieval_models_clients/Azure.mdx).
