import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import dspy
from dspy.predict.langchain import LangChainModule, LangChainPredict

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

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
# Now we wrap it in LangChainModule
rag_dspy_module = LangChainModule(rag_chain)  # then wrap the chain in a DSPy module.


questions = [
    "What are the main components of a Large Language Model (LLM)-powered autonomous agent system?",
    "How does self-reflection improve an autonomous agent's performance?",
    "What techniques can be used for task decomposition in LLM-powered agents?",
    "What role does memory play in LLM-powered autonomous agents?",
    "What are some case studies of LLM-powered agents in specialized domains?",
]

answers = [
    (
        "The main components are Planning, Memory, and Tool Use. Planning involves task decomposition and "
        "self-reflection, Memory includes short-term and long-term memory management, and Tool Use involves "
        "integrating external tools or APIs to enhance capabilities."
    ),
    (
        "Self-reflection allows agents to learn from past mistakes, refine actions, and iteratively improve "
        "decision-making by analyzing reasoning trajectories and using frameworks like ReAct and Reflexion."
    ),
    (
        "Techniques include 'Chain of Thought' prompting, 'Tree of Thoughts,' and using external planners like "
        "LLM+P for complex, long-horizon tasks."
    ),
    (
        "Memory helps agents retain information over time, with short-term memory handling immediate tasks and "
        "long-term memory enabling the recall of relevant information from external vector stores."
    ),
    (
        "Examples include ChemCrow for scientific discovery in chemistry and Generative Agents Simulation, where "
        "virtual characters interact in a simulated environment."
    ),
]

train_dataset = []

for question, answer in zip(questions, answers):
    train_dataset.append(dspy.Example(question=question, answer=answer).with_inputs("question"))


# Define the signature for autoamtic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = dspy.InputField(desc="ignore if N/A")
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


judge = dspy.OpenAI(model="gpt-4o-mini", max_tokens=1000, model_type="chat")


def metric(gold, pred, trace=None):
    question, answer = gold.question, gold.answer
    chain_output = pred.output
    context = retriever.invoke(question)

    faithful = (
        "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    )
    correct = f"The text above is should answer `{question}`. The gold answer is `{answer}`."
    correct = f"{correct} Does the assessed text above contain the gold answer?"

    with dspy.context(lm=judge):
        faithful = dspy.Predict(Assess)(context=context, assessed_text=chain_output, assessment_question=faithful)
        correct = dspy.Predict(Assess)(context="N/A", assessed_text=chain_output, assessment_question=correct)

    is_faithful = faithful.assessment_answer.split()[0].lower() == "yes"
    is_correct = correct.assessment_answer.split()[0].lower() == "yes"
    return is_faithful + is_correct


from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up the optimizer. We'll use very minimal hyperparameters for this example.
# Just do random search with ~3 attempts, and in each attempt, bootstrap <= 3 traces.
optimizer = BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=3, num_candidate_programs=3)

optimized_chain = optimizer.compile(rag_dspy_module, trainset=train_dataset)
