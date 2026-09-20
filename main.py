import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing...")

embeddings = OpenAIEmbeddings()
llm = ChatOllama(model="gemma4:e4b-mlx")
vstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

retriever = vstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer this question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)


def retrieval_chain_without_lcel(query: str) -> str:
    docs = retriever.invoke(query)

    context = format_docs(docs)

    msgs = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(msgs)

    return str(response.content)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved docs as a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    print("Retrieving...")
    question = "What is Pinecone in machine learning?"

    print("\n" + "=" * 50)
    print("IMPLEMENTATION 0: NO RAG")
    result = llm.invoke([HumanMessage(question)])
    print("Answer:")
    print(result.content)

    print("\n" + "=" * 50)
    print("IMPLEMENTATION 1: W/OUT LCEL")
    result = retrieval_chain_without_lcel(question)
    print("Answer:")
    print(result)
