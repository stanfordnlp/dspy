from typing import Any, Union

from dsp.utils import dotdict

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents._paging import SearchItemPaged
except ImportError:
    raise ImportError(
        "You need to install azure-search-documents library"
        "Please use the command: pip install azure-search-documents",
    )

# Deprecated: This module is scheduled for removal in future releases.
# Please use the AzureAISearchRM class from dspy.retrieve.azureaisearch_rm instead.
# For more information, refer to the updated documentation.

class AzureCognitiveSearch:
    """Wrapper for the Azure Cognitive Search Retrieval."""

    def __init__(
        self,
        search_service_name: str,
        search_api_key: str,
        search_index_name: str,
        field_text: str, # required field to map with "content" field in dsp framework
        field_score: str, # required field to map with "score" field in dsp framework

    ):
        self.search_service_name = search_service_name
        self.search_api_key = search_api_key
        self.search_index_name = search_index_name
        self.endpoint=f"https://{self.search_service_name}.search.windows.net"
        self.field_text = field_text # field name of the text content
        self.field_score = field_score # field name of the search score
        # Create a client
        self.credential = AzureKeyCredential(self.search_api_key)
        self.client = SearchClient(endpoint=self.endpoint,
                        index_name=self.search_index_name,
                        credential=self.credential)

    def __call__(self, query: str, k: int = 10) -> Union[list[str], list[dotdict]]:
        print("""# Deprecated: This module is scheduled for removal in future releases.
                Please use the AzureAISearchRM class from dspy.retrieve.azureaisearch_rm instead.
                For more information, refer to the updated documentation.""")

        topk: list[dict[str, Any]] = azure_search_request(self.field_text, self.field_score, self.client, query, k)
        topk = [{**d, "long_text": d["text"]} for d in topk]

        return [dotdict(psg) for psg in topk]

def azure_search_request(key_content: str, key_score: str,  client: SearchClient, query: str, top: int =1):
    '''
    Search in Azure Cognitive Search Index
    '''
    results = client.search(search_text=query,top=top)
    results = process_azure_result(results, key_content, key_content)

    return results

def process_azure_result(results:SearchItemPaged, content_key:str, content_score: str):
    '''
    process received result from Azure Cognitive Search as dictionary array and map content and score to correct format
    '''
    res = []
    for result in results:
        tmp = {}
        for key, value in result.items():
            if(key == content_key):
                tmp["text"] = value # assign content
            elif(key == content_score):
                tmp["score"] = value
            else:
                tmp[key] = value
        res.append(tmp)
    return res
