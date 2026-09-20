import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
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


def format_docs(docs) -> str:
    """Format retrieved docs as a single string."""
    pass


if __name__ == "__main__":
    print("Retrieving...")
