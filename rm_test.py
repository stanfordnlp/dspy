# import dspy.retrieve
# from dspy.retrieve.chromadb_rm import ChromadbRM
# from dotenv import load_dotenv
# import chromadb.utils.embedding_functions as embedding_functions
# from chromadb.utils.batch_utils import create_batches
# import os
# import dspy


# load_dotenv(override=True)
# emb_fn = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key=os.environ['OPENAI_API_KEY'],
#                 model_name="text-embedding-3-small")

# crm = ChromadbRM(
#     collection_name="rows",
#     persist_directory="/home/athekunal/DSPy-contributions/Text-to-SQL/India_TABLE",
#     embedding_function=emb_fn
# )
# # reranker = dspy.
# dspy.settings.configure(rm=crm)

# retriever = dspy.Retrieve(k=2)

# print(retriever(["Software Internet"],by_prob=False,where={"table_name":"capexIndia"}))
# print("-"*100)
# print(retriever(["Software Internet","Packaging"],by_prob=False,where={"table_name":"capexIndia"}))
import dspy

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts,reranker=colbertv2_wiki17_abstracts)

#Define Retrieve Module
retriever = dspy.RetrieveThenRerank(k=3)

query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
topK_passages = retriever([query])

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')