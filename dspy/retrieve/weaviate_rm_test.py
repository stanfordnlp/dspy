# ToDo

# Start Weaviate Embedded

# Add Toy Data

# Connect DSPy

# Test this API
"""
from dspy.retrieve.weaviate_rm import WeaviateRM
retriever_model = WeaviateRM("WeaviateBlogChunk", weaviate_client=weaviate_client)
dspy.settings.configure(rm=retriever_model)

retrieve = dspy.Retrieve(k=1)
topK_passages = retrieve("what are the stages in planning, sanctioning and execution of public works").passages

assert(type(topK_passages))
"""