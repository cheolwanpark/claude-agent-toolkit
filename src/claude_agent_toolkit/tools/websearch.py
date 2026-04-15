#!/usr/bin/env python3
# websearch.py - Web search tool with You.com Search API provider

import os
from typing import Any, Dict, List, Optional

import httpx

from claude_agent_toolkit import BaseTool, tool


class WebSearchTool(BaseTool):
    """Web search tool with You.com as primary provider and DuckDuckGo fallback."""

    def __init__(
        self,
        *,
        ydc_api_key: Optional[str] = None,
        default_count: int = 5,
        default_country: str = "US",
        default_language: str = "EN",
        fallback_to_duckduckgo: bool = True,
        timeout_s: int = 20,
    ):
        super().__init__()
        self.ydc_api_key = ydc_api_key or os.getenv("YDC_API_KEY")
        self.default_count = max(1, min(default_count, 20))
        self.default_country = default_country
        self.default_language = default_language
        self.fallback_to_duckduckgo = fallback_to_duckduckgo
        self.timeout_s = max(3, timeout_s)

    async def _search_youcom(
        self,
        query: str,
        *,
        count: int,
        country: str,
        language: str,
        freshness: Optional[str],
    ) -> Dict[str, Any]:
        if not self.ydc_api_key:
            raise RuntimeError("YDC_API_KEY is not configured")

        params: Dict[str, Any] = {
            "query": query,
            "count": max(1, min(count, 100)),
            "country": country,
            "language": language,
        }
        if freshness:
            params["freshness"] = freshness

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(
                "https://ydc-index.io/v1/search",
                params=params,
                headers={"X-API-Key": self.ydc_api_key},
            )

        if resp.status_code == 401:
            raise RuntimeError("YDC_API_KEY is invalid (401)")
        if resp.status_code == 429:
            raise RuntimeError("You.com Search API rate limit hit (429)")
        if resp.status_code >= 400:
            raise RuntimeError(f"You.com Search API error ({resp.status_code}): {resp.text[:200]}")

        payload = resp.json()
        results = payload.get("results", {})
        web_items = results.get("web", []) or []
        news_items = results.get("news", []) or []

        normalized: List[Dict[str, Any]] = []
        for item in web_items:
            normalized.append(
                {
                    "source": "web",
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("description") or " ".join(item.get("snippets", [])[:2]),
                    "page_age": item.get("page_age"),
                }
            )
        for item in news_items:
            normalized.append(
                {
                    "source": "news",
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("description"),
                    "page_age": item.get("page_age"),
                }
            )

        return {
            "provider": "you.com",
            "query": query,
            "results": normalized,
            "metadata": payload.get("metadata", {}),
            "total_results": len(normalized),
        }

    async def _search_duckduckgo(self, query: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"DuckDuckGo fallback error ({resp.status_code})")

        payload = resp.json()
        results: List[Dict[str, Any]] = []

        abstract_url = payload.get("AbstractURL")
        if abstract_url:
            results.append(
                {
                    "source": "instant_answer",
                    "title": payload.get("Heading") or "DuckDuckGo Instant Answer",
                    "url": abstract_url,
                    "snippet": payload.get("AbstractText"),
                    "page_age": None,
                }
            )

        for topic in payload.get("RelatedTopics", [])[:8]:
            if "FirstURL" in topic:
                results.append(
                    {
                        "source": "related",
                        "title": topic.get("Text", "DuckDuckGo related result")[:120],
                        "url": topic.get("FirstURL"),
                        "snippet": topic.get("Text"),
                        "page_age": None,
                    }
                )
            for nested in topic.get("Topics", [])[:4]:
                if "FirstURL" in nested:
                    results.append(
                        {
                            "source": "related",
                            "title": nested.get("Text", "DuckDuckGo related result")[:120],
                            "url": nested.get("FirstURL"),
                            "snippet": nested.get("Text"),
                            "page_age": None,
                        }
                    )

        return {
            "provider": "duckduckgo-fallback",
            "query": query,
            "results": results,
            "metadata": {"note": "Fallback result set is less comprehensive than You.com Search API."},
            "total_results": len(results),
        }

    @tool()
    async def search(
        self,
        query: str,
        count: Optional[int] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the web with You.com Search API, with automatic fallback handling."""
        if not query.strip():
            return {
                "success": False,
                "provider": None,
                "query": query,
                "results": [],
                "error": "Query must not be empty.",
            }

        search_count = count if count is not None else self.default_count
        country_code = country or self.default_country
        lang_code = language or self.default_language

        try:
            response = await self._search_youcom(
                query,
                count=search_count,
                country=country_code,
                language=lang_code,
                freshness=freshness,
            )
            response["success"] = True
            response["fallback_used"] = False
            return response
        except Exception as primary_error:
            if not self.fallback_to_duckduckgo:
                return {
                    "success": False,
                    "provider": "you.com",
                    "query": query,
                    "results": [],
                    "error": str(primary_error),
                    "fallback_used": False,
                }

            try:
                fallback = await self._search_duckduckgo(query)
                fallback["success"] = True
                fallback["fallback_used"] = True
                fallback["primary_provider_error"] = str(primary_error)
                return fallback
            except Exception as fallback_error:
                return {
                    "success": False,
                    "provider": "you.com",
                    "query": query,
                    "results": [],
                    "error": str(primary_error),
                    "fallback_error": str(fallback_error),
                    "fallback_used": False,
                }
