import json
import re
from typing import Any, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError

import dspy
from dsp.utils import dotdict

try:
    from snowflake.core import Root
    from snowflake.snowpark.functions import col
except ImportError:
    raise ImportError(
        "The snowflake-snowpark-python library is required to use SnowflakeRM. Install it with dspy-ai[snowflake]",
    )


class SnowflakeRM(dspy.Retrieve):
    """A retrieval module that uses Snowlfake's Cortex Search service to return the top relevant passages for a given query.

    Assumes that a Snowflake Cortex Search endpoint has been configured by the use.

    For more information on configuring the Cortex Search service, visit: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview

    Args:
        snowflake_sesssion (object): Snowflake Snowpark session for accessing the service.
        cortex_search_service(str): Name of the Cortex Search service to be used.
        snowflake_database (str): The name of the Snowflake table containing document embeddings.
        snowflake_schema (str): The name of the Snowflake table containing document embeddings.
        auto_filter (bool): Auto generate metadata filter based on user query and push it down prior to retrieving Cortex Search results.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
    """

    def __init__(
        self,
        snowflake_session: object,
        cortex_search_service: str,
        snowflake_database: str,
        snowflake_schema: str,
        auto_filter=False,
        k: int = 3,
        max_retries=3,
    ):
        super().__init__(k=k)
        self.k = k
        self.auto_filter = auto_filter
        self.max_retries = max_retries
        self.cortex_search_service_name = cortex_search_service
        self.client = self._fetch_cortex_service(
            snowflake_session, snowflake_database, snowflake_schema, cortex_search_service
        )

        if self.auto_filter:
            self.sample_values = self._get_sample_values(
                snowpark_session=snowflake_session, cortex_search_service=cortex_search_service
            )
            self.optimized_filter_gen = SmartSearch()
            self.filter_lm = dspy.Snowflake(session=snowflake_session, model="mixtral-8x7b")

    def forward(
        self,
        query_or_queries: Union[str, list[str]],
        retrieval_columns: list[str],
        filter: Optional[dict] = None,
        k: Optional[int] = None,
    ) -> dspy.Prediction:
        """Query Cortex Search endpoint for top k relevant passages.
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            retrieval_columns (List[str]): Columns to include in response.
            filter (Optional[json]):Filter query.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = self.k if k is None else k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages = []

        for cortex_query in queries:
            if self.auto_filter:
                response_chunks = self._cortex_search_with_auto_filter(
                    cortex_search_service=self.client, query=cortex_query, columns=retrieval_columns, k=k
                )

            else:
                response_chunks = self._query_cortex_search(
                    cortex_search_service=self.client,
                    query=cortex_query,
                    columns=retrieval_columns,
                    filter=filter,
                    k=k,
                )

            if len(retrieval_columns) == 1:
                passages.extend(
                    dotdict({"long_text": passage[retrieval_columns[0]]}) for passage in response_chunks["results"]
                )
            else:
                passages.extend(dotdict({"long_text": str(passage)}) for passage in response_chunks["results"])

        return passages

    def _fetch_cortex_service(self, snowpark_session, snowflake_database, snowflake_schema, cortex_search_service_name):
        """Fetch the Cortex Search service to be used"""
        snowpark_session.query_tag = {"origin": "sf_sit", "name": "dspy", "version": {"major": 1, "minor": 0}}
        root = Root(snowpark_session)

        # fetch service
        search_service = (
            root.databases[snowflake_database]
            .schemas[snowflake_schema]
            .cortex_search_services[cortex_search_service_name]
        )

        return search_service

    def _query_cortex_search(self, cortex_search_service, query, columns, filter, k):
        """Query Cortex Search endpoint for top relevant passages .

        Args:
            cortex_search_service (object): cortex search service for querying
            query (str): The query or queries to search for.
            columns (Optional[list]): A comma-separated list of columns to return for each relevant result in the response. These columns must be included in the source query for the service.
            filter (Optional[json]): A filter object for filtering results based on data in the ATTRIBUTES columns. See Filter syntax.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        # query service
        resp = cortex_search_service.search(query=query, columns=columns, filter=filter, limit=k)

        return json.loads(resp.to_json())

    def _cortex_search_with_auto_filter(self, cortex_search_service, query, columns, k):
        """Cortex Search Query with automatic metadata filter generation."""

        search_response = None
        with dspy.settings.context(lm=self.filter_lm):
            if self.auto_filter:
                for _ in range(self.max_retries):
                    raw_filter_query = self.optimized_filter_gen(
                        query=query, attributes=str([*self.sample_values]), sample_values=str(self.sample_values)
                    )["answer"]

                    try:
                        filter_query = json.loads(raw_filter_query)
                        search_response = self._query_cortex_search(
                            cortex_search_service=cortex_search_service,
                            query=query,
                            columns=columns,
                            filter=filter_query,
                            k=k,
                        )
                        break
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue

            else:
                search_response = self._query_cortex_search(
                    cortex_search_service=cortex_search_service, query=query, columns=columns, filter=None, k=k
                )

        return search_response

    def _get_search_attributes(self, snowpark_session, search_service_name):
        df = snowpark_session.sql("SHOW CORTEX SEARCH SERVICES")
        raw_atts = (
            df.where(col('"name"') == search_service_name).select('"attribute_columns"').to_pandas().loc[0].values[0]
        )
        attribute_list = raw_atts.split(",")

        return attribute_list

    def _get_search_table(self, snowpark_session, search_service_name):
        df = snowpark_session.sql("SHOW CORTEX SEARCH SERVICES")
        table_def = df.where(col('"name"') == search_service_name).select('"definition"').to_pandas().loc[0].values[0]

        pattern = r"FROM\s+([\w\.]+)"
        match = re.search(pattern, table_def)

        if match:
            from_value = match.group(1)
            return from_value
        else:
            print("No match found.")

        return table_def

    def _get_sample_values(self, snowpark_session, cortex_search_service, max_samples=10):
        sample_values = {}
        attributes = self._get_search_attributes(
            snowpark_session=snowpark_session, search_service_name=cortex_search_service
        )
        table_name = self._get_search_table(
            snowpark_session=snowpark_session, search_service_name=cortex_search_service
        )

        for attribute in attributes:
            query = f"""SELECT DISTINCT({attribute}) FROM {table_name} LIMIT {max_samples}"""
            sample_values[attribute] = list(snowpark_session.sql(query).to_pandas()[attribute].values)

        return sample_values


class JSONFilter(BaseModel):
    answer: str = Field(description="The filter_query in valid JSON format")

    @classmethod
    def model_validate_json(cls, json_data: str, *, strict: bool | None = None, context: dict[str, Any] | None = None):
        __tracebackhide__ = True
        try:
            return cls.__pydantic_validator__.validate_json(json_data, strict=strict, context=context)
        except ValidationError:
            min_length = get_min_length(cls)
            for substring_length in range(len(json_data), min_length - 1, -1):
                for start in range(len(json_data) - substring_length + 1):
                    substring = json_data[start : start + substring_length]
                    try:
                        res = cls.__pydantic_validator__.validate_json(substring, strict=strict, context=context)
                        return res
                    except ValidationError:
                        pass
        raise ValueError("Could not find valid json")


class GenerateFilter(dspy.Signature):
    """
    Given a query, attributes in the data, and example values of each attribute, generate a filter in valid JSON format.
    Ensure the filter only uses valid operators: @eq, @contains,@and,@or,@not
    Ensure only the valid JSON is output with no other reasoning.

    ---
    Query: What was the sentiment of CEOs between 2021 and 2024?
    Attributes: industry,hq,date
    Sample Values: {"industry":["biotechnology","healthcare","agriculture"],"HQ":["NY, US","CA,US","FL,US"],"date":["01/01,1999","01/01/2024"]}
    Answer: {"@or":[{"@eq":{"year":"2021"}},{"@eq":{"year":"2022"}},{"@eq":{"year":"2023"}},{"@eq":{"year":"2024"}}]}

    Query: Wha is the sentiment of Biotech CEO's of companies based in New York?
    Attributes: industry,hq,date
    Sample Values: {"industry":["biotechnology","healthcare","agriculture"],"HQ":["NY, US","CA,US","FL,US"],"date":["01/01,1999","01/01/2024"]}
    Answer: {"@and": [ { "@eq": { "industry"": "biotechnology" } }, { "@eq": { "HQ": "NY,US" } }]}

    Query: What is the sentiment of Biotech CEOs outside of California?
    Attributes: industry,hq,date
    Sample Values: {"industry":["biotechnology","healthcare","agriculture"],"HQ":["NY, US","CA,US","FL,US"],"date":["01/01,1999","01/01/2024"]}
    Answer: {"@and":[{ "@eq": { "industry": "biotechnology" } },{"@not":{"@eq":{"HQ":"CA,US"}}}]}

    Query: What is the sentiment of Biotech CEOs outside of California?
    Attributes: industry,hq,date
    Sample Values: {"industry":["biotechnology","healthcare","agriculture"],"HQ":["NY, US","CA,US","FL,US"],"date":["01/01,1999","01/01/2024"]}
    Answer: {"@and":[{ "@eq": { "industry": "biotechnology" } },{"@not":{"@eq":{"HQ":"CA,US"}}}]}

    Query: What is sentiment towards ag and biotech companies based outside of the US?
    Attributes: industry,hq,date
    Sample Values: {"industry"":["biotechnology","healthcare","agriculture"],"COUNTRY":["United States","Ireland","Russia","Georgia","Spain"],"month":["01","02","03","06","11","12""],""year"":["2022","2023","2024"]}
    Answer:{"@and": [{ "@or": [{"@eq":{ "industry": "biotechnology" } },{"@eq":{"industry":"agriculture"}}]},{ "@not": {"@eq": { "COUNTRY": "United States" } }}]}

    """

    query = dspy.InputField(desc="user query")
    attributes = dspy.InputField(desc="attributes to filter on")
    sample_values = dspy.InputField(desc="examples of values per attribute")
    answer: JSONFilter = dspy.OutputField(
        desc="filter query in valid JSON format. ONLY output the filter query in JSON, no reasoning"
    )


class SmartSearch(dspy.Module):
    def __init__(self):
        super().__init__()
        self.filter_gen = dspy.ChainOfThought(GenerateFilter)

    def forward(self, query, attributes, sample_values):
        filter_query = self.filter_gen(query=query, attributes=attributes, sample_values=sample_values)

        return filter_query


def get_min_length(model: Type[BaseModel]):
    min_length = 0
    for key, field in model.model_fields.items():
        if issubclass(field.annotation, BaseModel):
            min_length += get_min_length(field.annotation)
        min_length += len(key)
    return min_length
