import asyncio
import os
import ssl
import certifi
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# configure SSL context to use certifi cert (for Tavily API)
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="embeddinggemma:latest", dimensions=768)
vec_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)
tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)


async def main():
    print("Hello!")


if __name__ == "__main__":
    asyncio.run(main())
