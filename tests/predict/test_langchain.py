import pytest
pytest.importorskip("langchain")

import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings


from dspy.predict.langchain import LangChainModule, LangChainPredict

def test_copying_module():
    os.environ["OPENAI_API_KEY"] = "fake-key"
    llm = ChatOpenAI(model="gpt-4o-mini")
    docs = [Document(
        page_content="Hello, world!",
        metadata={"source": "https://example.com"}
    )]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=FakeEmbeddings(size=5))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | LangChainPredict(prompt, llm)
        | StrOutputParser()
    )
    # Now we wrap it in LangChainModule.
    rag_dspy_module = LangChainModule(rag_chain)

    copied_module = rag_dspy_module.reset_copy()
    assert len(copied_module.chain.steps) == len(rag_dspy_module.chain.steps)
    for (module, copied_module) in zip(rag_dspy_module.chain.steps, copied_module.chain.steps):
        if isinstance(module, LangChainPredict):
            # The LangChainPredict modules are deep copied.
            assert module != copied_module
            assert module.langchain_llm.model_name == copied_module.langchain_llm.model_name
        else:
            # The rest of the modules are just copied by reference.
            assert module == copied_module
    # Clean up.
    os.environ["OPENAI_API_KEY"] = None
