#!/usr/bin/env python3
"""
Web Search Integration for Local LLMs.

Provides multiple search backends:
- DuckDuckGo (default, no API key)
- Google Custom Search (requires API key)
- Bing Search (requires API key)
- SearXNG (self-hosted)

Usage:
    from agent.web_search import WebSearch

    search = WebSearch(backend="duckduckgo")
    results = search.search("Python programming", num_results=5)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus, urlencode


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    total_results: int
    search_time: float
    backend: str


class WebSearch:
    """
    Unified web search interface with multiple backends.

    Features:
    - Multiple search backends
    - Result caching
    - Rate limiting
    - Fallback mechanisms
    """

    def __init__(
        self,
        backend: str = "duckduckgo",
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize web search.

        Args:
            backend: Search backend (duckduckgo, google, bing, searxng)
            api_key: API key for paid backends
            cache_enabled: Enable result caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.backend = backend
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, SearchResponse]] = {}

    def search(
        self, query: str, num_results: int = 10, **kwargs
    ) -> SearchResponse:
        """
        Perform web search.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            SearchResponse with results
        """
        import time

        start_time = time.time()

        # Check cache
        if self.cache_enabled:
            cached = self._get_from_cache(query)
            if cached:
                return cached

        # Perform search based on backend
        search_func = {
            "duckduckgo": self._search_duckduckgo,
            "google": self._search_google,
            "bing": self._search_bing,
            "searxng": self._search_searxng,
        }.get(self.backend, self._search_duckduckgo)

        try:
            results = search_func(query, num_results, **kwargs)
        except Exception as e:
            print(f"Search error with {self.backend}: {e}")
            # Fallback to DuckDuckGo
            if self.backend != "duckduckgo":
                results = self._search_duckduckgo(query, num_results)
            else:
                results = []

        search_time = time.time() - start_time

        response = SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time=search_time,
            backend=self.backend,
        )

        # Cache results
        if self.cache_enabled:
            self._store_in_cache(query, response)

        return response

    def _search_duckduckgo(
        self, query: str, num_results: int, **kwargs
    ) -> list[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        import requests

        results = []

        # Method 1: DuckDuckGo Instant Answer API
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract abstract
                if data.get("Abstract"):
                    results.append(
                        SearchResult(
                            title=data.get("Heading", "DuckDuckGo Result"),
                            snippet=data.get("Abstract", ""),
                            url=data.get("AbstractURL", ""),
                            metadata={"source": data.get("AbstractSource", "")},
                        )
                    )

                # Extract answer
                if data.get("Answer"):
                    results.append(
                        SearchResult(
                            title="Direct Answer",
                            snippet=data.get("Answer", ""),
                            url=f"https://duckduckgo.com/?q={quote_plus(query)}",
                        )
                    )

                # Extract related topics
                for topic in data.get("RelatedTopics", []):
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(
                            SearchResult(
                                title=topic.get("Text", "")[:100],
                                snippet=topic.get("Text", ""),
                                url=topic.get("FirstURL", ""),
                            )
                        )
                    if len(results) >= num_results:
                        break

        except Exception as e:
            print(f"DuckDuckGo API error: {e}")

        # Method 2: DuckDuckGo HTML search (fallback)
        if len(results) < num_results:
            try:
                import requests
                from bs4 import BeautifulSoup

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    for result in soup.select(".result")[:num_results]:
                        title_el = result.select_one(".result__title a")
                        snippet_el = result.select_one(".result__snippet")

                        if title_el:
                            results.append(
                                SearchResult(
                                    title=title_el.text.strip(),
                                    snippet=snippet_el.text.strip() if snippet_el else "",
                                    url=title_el.get("href", ""),
                                )
                            )
            except Exception as e:
                print(f"DuckDuckGo HTML error: {e}")

        return results[:num_results]

    def _search_google(
        self, query: str, num_results: int, **kwargs
    ) -> list[SearchResult]:
        """Search using Google Custom Search API."""
        import requests

        if not self.api_key:
            raise ValueError("Google API key required. Set SEARCH_API_KEY env var.")

        cx = os.getenv("GOOGLE_CX", "")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": cx,
            "q": query,
            "num": min(num_results, 10),
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Google API error: {response.status_code}")

        data = response.json()
        results = []

        for item in data.get("items", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                    metadata={"displayLink": item.get("displayLink", "")},
                )
            )

        return results

    def _search_bing(
        self, query: str, num_results: int, **kwargs
    ) -> list[SearchResult]:
        """Search using Bing Search API."""
        import requests

        if not self.api_key:
            raise ValueError("Bing API key required. Set SEARCH_API_KEY env var.")

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": num_results, "mkt": "en-US"}

        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Bing API error: {response.status_code}")

        data = response.json()
        results = []

        for item in data.get("webPages", {}).get("value", []):
            results.append(
                SearchResult(
                    title=item.get("name", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("url", ""),
                    metadata={"dateLastCrawled": item.get("dateLastCrawled", "")},
                )
            )

        return results

    def _search_searxng(
        self, query: str, num_results: int, **kwargs
    ) -> list[SearchResult]:
        """Search using SearXNG (self-hosted)."""
        import requests

        base_url = os.getenv("SEARXNG_URL", "http://localhost:8888")
        url = f"{base_url}/search"
        params = {"q": query, "format": "json", "categories": "general"}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"SearXNG error: {response.status_code}")

        data = response.json()
        results = []

        for item in data.get("results", [])[:num_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    url=item.get("url", ""),
                    metadata={"engine": item.get("engine", "")},
                )
            )

        return results

    def _get_from_cache(self, query: str) -> Optional[SearchResponse]:
        """Get cached results if available and not expired."""
        import time

        if query in self._cache:
            cached_time, cached_response = self._cache[query]
            if time.time() - cached_time < self.cache_ttl:
                return cached_response
            else:
                del self._cache[query]
        return None

    def _store_in_cache(self, query: str, response: SearchResponse):
        """Store results in cache."""
        import time

        self._cache[query] = (time.time(), response)

    def clear_cache(self):
        """Clear the search cache."""
        self._cache.clear()

    def format_results(self, response: SearchResponse, max_snippet_len: int = 200) -> str:
        """Format search results as readable text."""
        lines = [f"Search: {response.query} ({response.total_results} results in {response.search_time:.2f}s)\n"]

        for i, result in enumerate(response.results, 1):
            snippet = result.snippet[:max_snippet_len]
            if len(result.snippet) > max_snippet_len:
                snippet += "..."
            lines.append(f"{i}. {result.title}")
            lines.append(f"   {snippet}")
            lines.append(f"   URL: {result.url}\n")

        return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Web search")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="Search the web")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument(
        "--backend",
        default="duckduckgo",
        choices=["duckduckgo", "google", "bing", "searxng"],
        help="Search backend",
    )
    p.add_argument("--api-key", default=None, help="API key for paid backends")
    p.add_argument("--num-results", type=int, default=10, help="Number of results")
    p.add_argument("--no-cache", action="store_true", help="Disable result caching")

    args = parser.parse_args(argv)

    try:
        if args.command == "search":
            search = WebSearch(
                backend=args.backend,
                api_key=args.api_key,
                cache_enabled=not args.no_cache,
            )
            response = search.search(args.query, num_results=args.num_results)
            data = {
                "query": response.query,
                "total_results": response.total_results,
                "search_time": response.search_time,
                "backend": response.backend,
                "results": [
                    {
                        "title": r.title,
                        "snippet": r.snippet,
                        "url": r.url,
                        "metadata": r.metadata,
                    }
                    for r in response.results
                ],
            }
            print(json.dumps(data, indent=2, default=str))
            return 0
        else:
            parser.error("Unknown command")
            return 2
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
