import asyncio

from pydantic import BaseModel

import dspy


class Document(BaseModel):
    doc_id: str
    text: str


class Request(BaseModel):
    question: str
    documents: list[Document]


class Citation(BaseModel):
    doc_id: str
    start_token_index: int
    end_token_index: int


class CitationWithText(Citation):
    cited_text: str


class ResponseChunk(BaseModel):
    answer_chunk: str
    citations: list[Citation] | None


class Response(BaseModel):
    answer_chunks: list[ResponseChunk]


class AnswerWithCitation(dspy.Signature):
    """
    You are a helpful assistant that answers questions based on the provided documents.
    To prove the answer is fully grounded, you must cite the content from source documents used to generate the answer chunk immediately after each chunk.
    Do not use [1], [2] in the generated text and put citations at the very end.
    Instead, output separate answer chunks with citations attached to each chunk.
    Keep both the answer and its citations very concise.
    The token index within each document starts with 0.
    The end index is exclusive.
    """

    question: str = dspy.InputField()
    documents: list[Document] = dspy.InputField()
    answer_chunks: list[Response] = dspy.OutputField(
        desc="answer with citations. If no citation is found, the `Response.citations` field is None. Make sure citation is only applied to its supported text"
    )


lm = dspy.LM("openai/gpt-4o-mini", cache=False)

dspy.configure(lm=lm)

# Put some space
documents = [
    Document(
        doc_id="1",
        text="Databricks provides a unified platform for data engineering, data science, and machine learning.",
    ),
    Document(doc_id="2", text="Databricks supports open source projects like Spark and MLflow."),
    Document(doc_id="3", text="Snowflake is a competitor to Databricks."),
    Document(
        doc_id="4",
        text="Agent Bricks provides a simple approach to build and optimize domain-specific, high-quality AI agent systems for common AI use cases. Agent Bricks streamlines the implementation of AI agent systems so that users can focus on the problem, data, and metrics instead. Agent Bricks is a Databricks Designated Service, which means that it uses Databricks Geos to manage data residency when processing customer content.",
    ),
]


predict = dspy.Predict(AnswerWithCitation)

streamed_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer_chunks")],
)

output = streamed_predict(question="What does databricks do?", documents=documents)


async def main():
    concated = []
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            concated.append(chunk.chunk)

    return "".join(concated)


concat_output = asyncio.run(main())
print(concat_output)
