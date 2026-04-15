#!/usr/bin/env python3
"""WebSearchTool demo for You.com Search API integration."""

import asyncio
import os

from claude_agent_toolkit import Agent
from claude_agent_toolkit.tools import WebSearchTool


async def main() -> None:
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        raise RuntimeError("Set CLAUDE_CODE_OAUTH_TOKEN before running this example")

    # Primary provider uses YDC_API_KEY; fallback remains available when enabled.
    search_tool = WebSearchTool(fallback_to_duckduckgo=True)

    agent = Agent(
        system_prompt="You are a research assistant. Use web search and cite URLs.",
        tools=[search_tool],
        model="haiku",
    )

    prompt = "Find recent sources about Model Context Protocol adoption and summarize in 5 bullets."
    response = await agent.run(prompt)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
