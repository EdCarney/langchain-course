import asyncio
import os
import ssl
import certifi
import logging
from typing import Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    logging.info("DOCUMENTATION INGESTION PIPELINE")

    logging.info(
        "TavilyCrawl: starting to crawl documentation from https://python.langchain.com/"
    )

    resp = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com",
            "max_depth": 1,
            "extract_depth": "advanced",
        }
    )
    all_docs = [
        Document(page_content=res["raw_content"], metadata={"source": res["url"]})
        for res in resp["results"]
    ]

    logging.info(
        f"TavilyCrawl: successfully crawled {len(all_docs)} URLs from documentation site."
    )


if __name__ == "__main__":
    asyncio.run(main())
