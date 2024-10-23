import os
import dspy
from dsp.utils import dotdict
import warnings

from typing import List, Optional, Union, Dict
from r2r import R2RClient
from shared.abstractions import R2RException
from uuid import UUID

class R2RRetrieve(dspy.Retrieve):
    """A DSPy retriever implementation for R2R (RAG-to-Riches) system: https://r2r-docs.sciphi.ai/introduction
    
    This class extends DSPy's Retrieve class to provide integration with R2R, supporting 
    vector and hybrid search as well as knowledge graph-based retrieval capabilities. It can operate in two modes:
    'search' for direct retrieval and 'rag' for full retrieval-augmented generation.

    Arguments
    ---------
    mode : str, optional (default='search')
        Operating mode for the retriever:
        - 'search': Direct retrieval of relevant passages
        - 'rag': Retrieval-augmented generation mode
    model : str, optional (default='gpt-4o')
        The language model to use for RAG and knowledge graph operations in the server. The API key must be
        available in the environmental variables for the session. This is unrelated to the DSPy LM settings
    use_vector : bool, optional (default=True)
        Whether to enable vector-based search
    use_hybrid : bool, optional (default=True)
        Whether to enable hybrid search (combining vector and traditional keyword-based search).
        Note: hybrid is exposed as an attribute of vector, so vector must be True for hybrid to be enabled
    use_kg_search : bool, optional (default=True)
        Whether to enable knowledge graph search
    collection_names : str or List[str], optional (default='Default')
        Human-readable name(s) of the R2R collection(s) used for search.
    collection_ids: str or List[str], optional
        R2R-generated UUID corresponding to the collection(s) used for search.
    k : int, optional (default=3)
        Number of passages to retrieve

    Returns
    -------
    List[dotdict]
        List of retrieved passages

    Raises
    ------
    ValueError
        If R2R health check fails, collection is not found, or invalid mode is specified
    R2RException
        If R2R system is not available or returns an error

    Examples
    --------
    >>> rm = R2RRetrieve(collection_name="My Collection")
    >>> dspy.configure(lm=lm, rm=rm) # lm previously defined
    >>> retriever = dspy.Retrieve(k=10)
    >>> query="What is machine learning?"
    >>> topK_passages = retriever(query).passages

    Notes
    -----
    - The URI for the R2R server needs to be in the environment in R2R_URI. 
      Otherwise, it will default to "http://localhost:7272
    - Default search mode is "search" (returning k passages), with all methods enabled
    - When 'rag' is selected, also all search methods are enabled, but k will be ignored and a single entry returned
      with generated content from R2R created with model specified in 'model'
    - R2R is currently used in default user mode. Complete user management could be added later.
    - It is possible to define most parameters at instantiation of the class, or through the forward 
      method of the instance for flexibility
    - If no results are found, returns a single passage with "No relevant passages found"
    - You can completely customize the output by passing the R2R parameters as dictionaries via kwargs.
      Review the R2R documentation at: https://r2r-docs.sciphi.ai/api-reference/endpoint/search

    See Also
    --------
    dspy.Retrieve : Base retriever class
    R2RClient : Client for interacting with R2R system

    Current build and testing
    -------------------------
    Python: 3.12.7
    R2R Server: 3.2.17
    R2R Python SDK: 3.2.17
    DSPy: 2.5.15

    Author
    ------
    @RamXX (Ramiro Salas)

    """
    def __init__(
        self,
        mode: str = "search",
        model: str = "gpt-4o",
        use_vector: bool = True,
        use_hybrid: bool = True,
        use_kg_search: bool = True,
        collection_names: Union[str, List[str]] = "Default",
        collection_ids: Optional[Union[str, List[str]]] = None,
        k: int = 3,
    ):
        super().__init__(k=k)
        
        # Validate search parameters
        if not (use_vector or use_hybrid or use_kg_search):
            warnings.warn("All search methods are disabled. Searches will return no results.")
            
        if not use_vector and use_hybrid:
            warnings.warn("Hybrid search requires vector search. Disabling hybrid search.")
            use_hybrid = False
            
        self.default_params = {
            "mode": "rag" if mode == "rag" else "search",
            "model": model,
            "use_vector": use_vector,
            "use_hybrid": use_hybrid,
            "use_kg_search": use_kg_search,
            "collection_names": collection_names if isinstance(collection_names, list) else [collection_names],
            "collection_ids": collection_ids if isinstance(collection_ids, list) else [collection_ids] if collection_ids else []
        }
        
        try:
            self.client = R2RClient(os.getenv("R2R_URI", "http://localhost:7272"))
            if (health_response := self.client.health().get("results", {"response": "error"}).get("response")) != "ok": # type: ignore
                raise ValueError(f"R2R failed returning an ok health response: {health_response}")
        except R2RException as e:
            raise ValueError(f"R2R System not ready: {e}")
    

    def _get_collections(
        self, 
        collection_names: List[str], 
        collection_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Get collection IDs from both names and direct IDs.

        Args:
            collection_names: List of collection names to search for.
            collection_ids: Optional list of collection UUIDs to include.

        Returns:
            List of unique collection IDs.
        """
        resolved_ids = set()
        not_found = []

        # Validate and add direct collection IDs if provided
        if collection_ids:
            for cid in collection_ids:
                if cid:
                    try:
                        # Validate UUID format
                        UUID(cid)
                        resolved_ids.add(cid)
                    except ValueError:
                        warnings.warn(f"Invalid collection ID format: {cid}")

        # Resolve collection names to IDs, handling duplicates
        collections_list = self.client.list_collections()
        name_to_ids_map: Dict[str, List[str]] = {}

        # Build a map of names to a list of their corresponding collection IDs
        for col in collections_list.get('results', []):  # Handle missing 'results' safely
            name = col['name']
            col_id = col['collection_id']
            if name not in name_to_ids_map:
                name_to_ids_map[name] = []
            name_to_ids_map[name].append(col_id)

        # Add all IDs matching the provided names
        for name in collection_names:
            if name in name_to_ids_map:
                resolved_ids.update(name_to_ids_map[name])
            else:
                not_found.append(name)

        if not_found:
            warnings.warn(f"The following collections were not found: {', '.join(not_found)}")

        if not resolved_ids:
            raise ValueError("No valid collections found to search in")

        return list(resolved_ids)
    

    def forward(
        self,
        query: str,
        k: Optional[int] = None,
        mode: Optional[str] = None,
        model: Optional[str] = None,
        use_vector: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        use_kg_search: Optional[bool] = None,
        collection_names: Optional[Union[str, List[str]]] = None,
        collection_ids: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> List[dotdict]:
        """[previous docstring remains the same]"""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or contain only whitespace")
        
        # Merge default parameters with any provided overrides
        params = self.default_params.copy()
        if mode is not None:
            params["mode"] = "rag" if mode == "rag" else "search"
        if model is not None:
            params["model"] = model
        if use_vector is not None:
            params["use_vector"] = use_vector
        if use_hybrid is not None:
            params["use_hybrid"] = use_hybrid and (use_vector if use_vector is not None else params["use_vector"])
        if use_kg_search is not None:
            params["use_kg_search"] = use_kg_search
        if collection_names is not None:
            params["collection_names"] = collection_names if isinstance(collection_names, list) else [collection_names]
        if collection_ids is not None:
            params["collection_ids"] = collection_ids if isinstance(collection_ids, list) else [collection_ids]
            
        k = k if k is not None and k > 0 else self.k
        
        # Resolve collections
        try:
            collections = self._get_collections(params["collection_names"], params["collection_ids"])
        except ValueError as e:
            return [dotdict({'text': str(e), 'long_text': str(e)})]
  
        vector_search_settings = {
            "use_vector_search": params["use_vector"],
            "selected_collection_ids": collections,
            "search_limit": k,
            "use_hybrid_search": params["use_hybrid"]
        } if params["use_vector"] else None
        
        kg_search_settings = {
            "use_kg_search": params["use_kg_search"],
            "kg_search_type": "local",
            "kg_search_level": "0",
            "selected_collection_ids": collections,
            "generation_config": {
                "model": params["model"],
                "temperature": 0.7,
            },
            "local_search_limits": {
                "__Entity__": k,
                "__Relationship__": k,
                "__Community__": k,
            },
            "max_community_description_length": 65536,
            "max_llm_queries_for_global_search": 250
        } if params["use_kg_search"] else None
            
        rag_generation_config = {
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 150
        } if params["mode"] == "rag" else None

        # Handle user overrides
        if vector_search_settings and 'vector_search_settings' in kwargs:
            vector_search_settings.update(kwargs['vector_search_settings'])
        if kg_search_settings and 'kg_search_settings' in kwargs:
            kg_search_settings.update(kwargs['kg_search_settings'])
        if rag_generation_config and 'rag_generation_config' in kwargs:
            rag_generation_config.update(kwargs['rag_generation_config'])

        # Build search arguments
        search_args = {
            "query": query,
            **kwargs
        }
        if vector_search_settings:
            search_args["vector_search_settings"] = vector_search_settings
        if kg_search_settings:
            search_args["kg_search_settings"] = kg_search_settings
        if rag_generation_config:
            search_args["rag_generation_config"] = rag_generation_config

        # Perform actual search/rag API call to the R2R server URI
        if params["mode"] == "search":
            result = self.client.search(**search_args)
        elif params["mode"] == "rag":
            result = self.client.rag(**search_args)
        else:
            raise ValueError(f"Query mode must be 'search' or 'rag' only. Value received: {params['mode']}")

        passages = []
        
        if params["mode"] == "search":
            # Handle vector search results if enabled. Hybrid is included in vector
            if params["use_vector"] and 'results' in result:
                vector_results = result.get('results', {}).get('vector_search_results', [])               # type: ignore
                if vector_results:
                    for item in vector_results:
                        if isinstance(item, dict) and 'text' in item:
                            passages.append(dotdict({'text': item['text'], 'long_text': item['text']}))
            
            # Handle KG search results if enabled
            if params["use_kg_search"] and 'results' in result:
                kg_results = result.get('results', {}).get('kg_search_results', [])                      # type: ignore
                if kg_results:
                    for item in kg_results:
                        if isinstance(item, dict):
                            if item.get('result_type') == 'entity':
                                desc = item.get('content', {}).get('description', '')
                                if desc:
                                    passages.append(dotdict({'text': desc, 'long_text': desc}))
                            elif item.get('result_type') == 'community':
                                findings = item.get('content', {}).get('findings', [])
                                if findings:
                                    passages.extend([dotdict({'text': finding, 'long_text': finding}) 
                                                   for finding in findings])
        else:  # RAG mode
            if ('results' in result and 
                'completion' in result.get('results', {}) and                                           # type: ignore
                'choices' in result['results']['completion'] and                                        # type: ignore
                len(result['results']['completion']['choices']) > 0 and                                 # type: ignore
                'message' in result['results']['completion']['choices'][0]):                            # type: ignore
                content = result['results']['completion']['choices'][0]['message']['content']           # type: ignore
                passages = [dotdict({'text': content, 'long_text': content})]

        if not passages:
            passages = [dotdict({'text': "No relevant passages found.", 
                               'long_text': "No relevant passages found."})]

        return passages[:k]