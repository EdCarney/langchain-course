import os
from dotenv import load_dotenv
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def main():
    print("ingesting...")
    loader = UnstructuredLoader(
        partition_via_api=True,
        file_path="./mediumblog1.txt",
        url="http://localhost:8000/",
        api_key="",
        strategy="hi_res",
        max_characters=1_000_000,
        encoding="utf-8",
    )
    docs = loader.load()

    print("splitting...")
    txt_splitter = CharacterTextSplitter(chunk_size=1_000, chunk_overlap=0)
    chunks = txt_splitter.split_documents(docs)

    print("loading...")

    embeddings = OllamaEmbeddings(model="embeddinggemma:latest", dimensions=768)

    print("uploading...")

    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.environ.get("INDEX_NAME")
    )

    print("done")


if __name__ == "__main__":
    main()
