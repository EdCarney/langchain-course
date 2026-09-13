from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    A tool that searches the internet for the specified query.
    Args:
        query: The query to search for.
    Returns:
        A human-readable string representing the search result.
    """
    print(f"searching for query: {query}")
    return tavily.search(query=query)["results"]


def main():
    llm = ChatOllama(model="gemma4:e4b-mlx")
    llm = ChatAnthropic(model="claude-sonnet-5")
    agent = create_agent(model=llm, tools=[TavilySearch()])
    agent_data = {
        "messages": HumanMessage(
            content="Search for 3 job postings for a LangChain AI engineer in the Bay Area. Please also list all of their related details."
        ),
    }
    result = agent.invoke(agent_data)
    print(result)


if __name__ == "__main__":
    main()
