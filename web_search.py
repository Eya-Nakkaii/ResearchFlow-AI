
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def search_web(query: str) -> list[dict[str, str]]:
    """
    Search the web using Tavily API.

    Args:
        query: The search question.

    Returns:
        List of sources with title, URL and snippet.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY manquante dans .env")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=2,
        include_answer=False
    )

    sources = []

    for item in response.get("results", []):
        sources.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", "")
        })

    return sources
