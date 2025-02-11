import pytest
from unittest.mock import Mock, patch
from datetime import timedelta
import os
from typing import List
from couchbase.options import ClusterOptions, KnownConfigProfiles
from couchbase.auth import PasswordAuthenticator
from couchbase.management.search import SearchIndex
import json
import time
from dspy.retrieve.couchbase_rm import CouchbaseSearchRM, Embedder
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.bucket import Bucket
from couchbase.scope import Scope
from couchbase.collection import Collection
from couchbase.result import SearchResult, GetResult
from couchbase.management.buckets import CreateBucketSettings
from couchbase.cluster import Cluster


########################################################################################
### Integration tests
########################################################################################

# Update TEST_DOCS to match the vector index schema
TEST_DOCS = [
    {
        "id": "doc1",
        "content": "Jorah Mormont fled to Essos after selling poachers into slavery to pay for his wife's expensive tastes.",
        "embedding": None,  # Will be populated by OpenAI
        "meta": {
            "name": "got_doc_1"
        },
        "type": "document"
    },
    {
        "id": "doc2",
        "content": "Daenerys Targaryen is known as the Mother of Dragons after hatching three dragon eggs.",
        "embedding": None,  # Will be populated by OpenAI
        "meta": {
            "name": "got_doc_2"
        },
        "type": "document"
    }
]

# Add this constant with the vector index configuration
VECTOR_INDEX_CONFIG = {
    "name": "vector_search",
    "type": "fulltext-index",
    "sourceType": "gocbcore",
    "sourceName": "dspy_test",
    "planParams": {
        "indexPartitions": 1,
        "numReplicas": 0
    },
    "params": {
        "doc_config": {
            "docid_prefix_delim": "",
            "docid_regexp": "",
            "mode": "scope.collection.type_field",
            "type_field": "type"
        },
        "mapping": {
            "default_analyzer": "standard",
            "default_datetime_parser": "dateTimeOptional",
            "index_dynamic": True,
            "store_dynamic": True,
            "default_mapping": {
                "dynamic": True,
                "enabled": False
            },
            "types": {
                "got.got_collection": {
                    "dynamic": False,
                    "enabled": True,
                    "properties": {
                        "content": {
                            "enabled": True,
                            "fields": [{
                                "docvalues": True,
                                "include_in_all": False,
                                "include_term_vectors": False,
                                "index": True,
                                "name": "content",
                                "store": True,
                                "type": "text"
                            }]
                        },
                        "embedding": {
                            "enabled": True,
                            "dynamic": False,
                            "fields": [{
                                "vector_index_optimized_for": "recall",
                                "docvalues": True,
                                "dims": 3072,
                                "include_in_all": False,
                                "include_term_vectors": False,
                                "index": True,
                                "name": "embedding",
                                "similarity": "dot_product",
                                "store": True,
                                "type": "vector"
                            }]
                        },
                        "meta": {
                            "dynamic": True,
                            "enabled": True,
                            "properties": {
                                "name": {
                                    "enabled": True,
                                    "fields": [{
                                        "docvalues": True,
                                        "include_in_all": False,
                                        "include_term_vectors": False,
                                        "index": True,
                                        "name": "name",
                                        "store": True,
                                        "analyzer": "keyword",
                                        "type": "text"
                                    }]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

# Add these constants for index names
GLOBAL_INDEX_NAME = "vector_search_global"
SCOPED_INDEX_NAME = "vector_search"

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY environment variable not set")
class TestCouchbaseRMIntegration:
    """Integration tests for CouchbaseSearchRM using real Couchbase connection"""
    
    @staticmethod
    def create_vector_indexes(cluster: Cluster, scope: Scope = None) -> None:
        """Create both global and scoped vector search indexes if they don't exist."""
        try:
            # Get search indexes manager
            search_mgr = cluster.search_indexes()
            
            # Create global index
            try:
                search_mgr.get_index(GLOBAL_INDEX_NAME)
                print(f"Global vector index '{GLOBAL_INDEX_NAME}' already exists")
            except Exception:
                # Create a copy of the config and modify for global index
                global_config = json.loads(json.dumps(VECTOR_INDEX_CONFIG))  # Deep copy
                global_config["name"] = GLOBAL_INDEX_NAME
                
                global_index = SearchIndex.from_json(global_config)
                search_mgr.upsert_index(global_index)
                time.sleep(10)
                print(f"Global vector index '{GLOBAL_INDEX_NAME}' created successfully")
            
            # Create scoped index if scope is provided
            if scope:
                try:
                    scope.search_indexes().get_index(SCOPED_INDEX_NAME)
                    print(f"Scoped vector index '{SCOPED_INDEX_NAME}' already exists")
                except Exception:
                    # Create a copy of the config and modify for scoped index
                    scoped_config = json.loads(json.dumps(VECTOR_INDEX_CONFIG))  # Deep copy
                    scoped_config["name"] = SCOPED_INDEX_NAME
                    scoped_config["params"]["doc_config"]["mode"] = "scope.collection.type_field"
                    
                    scoped_index = SearchIndex.from_json(scoped_config)
                    scope.search_indexes().upsert_index(scoped_index)
                    time.sleep(10)
                    print(f"Scoped vector index '{SCOPED_INDEX_NAME}' created successfully")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to create vector indexes: {str(e)}") from e

    @pytest.fixture(scope="class")
    def openai_embedder(self):
        """Fixture for OpenAI embedder"""
        api_key = os.getenv("OPENAI_API_KEY")
        return Embedder(provider="openai", model="text-embedding-3-large")

    @pytest.fixture(scope="class")
    def couchbase_config(self):
        """Fixture for Couchbase configuration"""
        cluster_options = ClusterOptions(
            authenticator=PasswordAuthenticator(
                username=os.getenv("COUCHBASE_USER", "admin"),
                password=os.getenv("COUCHBASE_PASSWORD", "Password")
            )
        )
        cluster_options.apply_profile(KnownConfigProfiles.WanDevelopment)
        return {
            "cluster_connection_string": os.getenv("COUCHBASE_HOST", "couchbase://localhost"),
            "bucket": "dspy_test",
            "scope": "got",
            "collection": "got_collection",
            "index_name": "vector_search",
            "cluster_options": cluster_options
        }

    @pytest.fixture(scope="class")
    def setup_couchbase(self, couchbase_config, openai_embedder):
        """Fixture to set up test data in Couchbase"""
        from couchbase.cluster import Cluster
        print(couchbase_config)
        # Connect to cluster
        cluster = Cluster(
            couchbase_config["cluster_connection_string"],
            couchbase_config["cluster_options"]
        )
        cluster.wait_until_ready(timeout=timedelta(seconds=30))
        
        # Create test bucket, scope, collection
        try:
            bucket = cluster.bucket(couchbase_config["bucket"])
        except Exception:
            cluster.buckets().create_bucket(
                settings=CreateBucketSettings(
                    name=couchbase_config["bucket"],
                    ram_quota_mb=250,
                )
            )
            time.sleep(5)
            bucket = cluster.bucket(couchbase_config["bucket"])
        mgr = bucket.collections()
        try:
            mgr.create_scope(couchbase_config["scope"])
            time.sleep(5)
        except Exception:
            pass
        
        try:
            print(f"Creating collection {couchbase_config['collection']} in scope {couchbase_config['scope']}")
            mgr.create_collection(
                scope_name=couchbase_config["scope"],
                collection_name=couchbase_config["collection"],
            )
            time.sleep(5)
        except Exception as e:
            pass
        
        # Create both vector search indexes
        scope = bucket.scope(couchbase_config["scope"])
        self.create_vector_indexes(cluster, scope)
        
        # Generate embeddings and insert test documents
        collection = scope.collection(couchbase_config["collection"])
        
        # Generate embeddings for all documents at once
        texts = [doc["content"] for doc in TEST_DOCS]
        embeddings = openai_embedder(texts)
        
        for doc, embedding in zip(TEST_DOCS, embeddings):
            doc_with_embedding = doc.copy()
            doc_with_embedding["embedding"] = embedding
            collection.upsert(doc["id"], doc_with_embedding)
        
        yield cluster
        
        # Cleanup
        try:
            cluster.buckets().drop_bucket(couchbase_config["bucket"])
        except Exception:
            pass

    def test_initialization(self, setup_couchbase,couchbase_config):
        """Test successful initialization with real Couchbase"""
        rm = CouchbaseSearchRM(
            cluster_connection_string=couchbase_config["cluster_connection_string"],
            bucket=couchbase_config["bucket"],
            scope=couchbase_config["scope"],
            collection=couchbase_config["collection"],
            index_name=couchbase_config["index_name"],
            cluster_options=couchbase_config["cluster_options"],
            embedding_model="text-embedding-3-large",
            text_field="content"
        )
        assert rm is not None
        assert rm.bucket_name == couchbase_config["bucket"]
        assert rm.scope_name == couchbase_config["scope"]
        assert rm.collection_name == couchbase_config["collection"]

    def test_vector_search(self, setup_couchbase, couchbase_config):
        """Test vector search with real Couchbase and OpenAI embeddings"""
        rm = CouchbaseSearchRM(
            cluster_connection_string=couchbase_config["cluster_connection_string"],
            bucket=couchbase_config["bucket"],
            scope=couchbase_config["scope"],
            collection=couchbase_config["collection"],
            index_name=couchbase_config["index_name"],
            cluster_options=couchbase_config["cluster_options"],
            embedding_model="text-embedding-3-large",
            text_field="content",
            k=2
        )
        
        results = rm.forward("Who is Daenerys?")
        assert len(results) > 0
        assert isinstance(results[0].long_text, str)
        assert "Daenerys" in results[0].long_text

    def test_global_vector_search(self, setup_couchbase, couchbase_config):
        """Test global vector search with real Couchbase"""
        rm = CouchbaseSearchRM(
            cluster_connection_string=couchbase_config["cluster_connection_string"],
            bucket=couchbase_config["bucket"],
            scope=couchbase_config["scope"],
            collection=couchbase_config["collection"],
            index_name=GLOBAL_INDEX_NAME,
            cluster_options=couchbase_config["cluster_options"],
            embedding_model="text-embedding-3-large",
            text_field="content",
            is_global_index=True,
            k=2
        )
        
        results = rm.forward("Who is Daenerys?")
        assert len(results) > 0
        assert isinstance(results[0].long_text, str)
        assert "Daenerys" in results[0].long_text

    def test_scoped_vector_search(self, setup_couchbase, couchbase_config):
        """Test scoped vector search with real Couchbase"""
        rm = CouchbaseSearchRM(
            cluster_connection_string=couchbase_config["cluster_connection_string"],
            bucket=couchbase_config["bucket"],
            scope=couchbase_config["scope"],
            collection=couchbase_config["collection"],
            index_name=SCOPED_INDEX_NAME,
            cluster_options=couchbase_config["cluster_options"],
            embedding_model="text-embedding-3-large",
            text_field="content",
            is_global_index=False,
            k=2
        )
        
        results = rm.forward("Who is Daenerys?")
        assert len(results) > 0
        assert isinstance(results[0].long_text, str)
        assert "Daenerys" in results[0].long_text

    def test_kv_get_text(self, setup_couchbase, couchbase_config):
        """Test KV get operation with real Couchbase"""
        rm = CouchbaseSearchRM(
            cluster_connection_string=couchbase_config["cluster_connection_string"],
            bucket=couchbase_config["bucket"],
            scope=couchbase_config["scope"],
            collection=couchbase_config["collection"],
            index_name=couchbase_config["index_name"],
            cluster_options=couchbase_config["cluster_options"],
            embedding_model="text-embedding-3-large",
            text_field="content",
            use_kv_get_text=True
        )
        
        results = rm.forward("Who is Daenerys?")
        assert len(results) > 0
        assert isinstance(results[0].long_text, str)
        assert "Daenerys" in results[0].long_text

    def test_error_handling(self, couchbase_config):
        """Test error handling with invalid configuration"""
        with pytest.raises(ConnectionError):
            CouchbaseSearchRM(
                cluster_connection_string="couchbase://invalid-host",
                bucket="invalid_bucket",
                scope="invalid_scope",
                collection="invalid_collection",
                index_name="invalid_index",
                cluster_options=couchbase_config["cluster_options"],
                embedding_model="text-embedding-3-large",
                text_field="content"
            ) 