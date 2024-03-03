import dspy

class elastic_rm(dspy.Retrieve):
    def __init__(self, es_client, es_index, es_field, k=3):
        """"
        A retrieval module that uses Elastic simple vector search to return the top passages for a given query.
        Assumes that you already have instanciate your ESClient.

        The code has been tested with ElasticSearch 8.12
        For more information on how to instanciate your ESClient, please refer to the official documentation.
        Ref: https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html

        Args:
            es_client (Elasticsearch): An instance of the Elasticsearch client.
            es_index (str): The name of the index to search.
            es_field (str): The name of the field to search.
            k (Optional[int]): The number of context strings to return. Default is 3.
        """
        super().__init__()
        self.k=k
        self.es_index=es_index
        self.es_client=es_client
        self.field=es_field
        

    def forward(self, query) -> dspy.Prediction:
        """Search with Elastic Search - local or cloud for top k passages for query or queries
   

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of context strings to return, if not already specified in self.k

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """

        passages = []

        # Define the index to search
        index_name = self.es_index #the name of the index of your elastic-search-dump

        # Define the search query
        search_query = {
            "query": {
                "match": {
                    self.field: query  #took for granted that your index has : title, text as document format
                }
            }
        }

        # Perform the search
        response = self.es_client.search(index=index_name, body=search_query)

        for hit in response['hits']['hits']:

            #Uncomment for debug...
            # Retrieve the score
            #score = hit["_score"]
            # Retrieve other fields from the source
            #title = hit["_source"]["title"]
            text = hit["_source"]["text"]
            #print("Score: %.2f | Tile: %s | Text: %s" % (score,title, text))
            passages.append(text)
            if len(passages) == self.k:  # Break the loop once k documents are retrieved
                break

        return dspy.Prediction(passages=passages)