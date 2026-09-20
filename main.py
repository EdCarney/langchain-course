import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

load_dotenv()

print("Initializing...")

lf_client = get_client()
lf_callback = CallbackHandler()

embeddings = OllamaEmbeddings(model="embeddinggemma:latest", dimensions=768)
llm = ChatOllama(model="gemma4:e4b-mlx")
vstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

retriever = vstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer this question based only on the following context:

    {context}

    Question: {question}

    Provide concise, one-paragraph answer with some bullet points:"""
)


def retrieval_chain_without_lcel(query: str) -> str:
    docs = retriever.invoke(query)

    context = format_docs(docs)

    msgs = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(msgs)

    return str(response.content)


@observe(name="LangChain with RAG")
def retrieval_chain_with_lcel() -> Runnable:
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


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

    print("\n" + "=" * 50)
    print("IMPLEMENTATION 2: WITH LCEL")
    chain = retrieval_chain_with_lcel()
    result = chain.invoke(
        input={"question": question}, config={"callbacks": [lf_callback]}
    )
    print("Answer:")
    print(result)
