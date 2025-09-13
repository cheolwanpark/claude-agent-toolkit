#!/usr/bin/env python3
# abstract.py - Abstract base class for all MCP tools

from abc import ABC, abstractmethod
from typing import Union

# Import types from utils to avoid circular imports
try:
    from .utils import McpServerConfig
except ImportError:
    # Fallback type definition to handle import order issues
    from typing import Dict, List, Any
    McpServerConfig = Dict[str, Any]


class AbstractTool(ABC):
    """
    Abstract base class for all MCP tools.

    This base class defines the interface that all tools must implement,
    whether they are HTTP-based BaseTool instances or external MCP servers.

    The two key methods allow the agent system to:
    1. Get the MCP server configuration for connecting to the tool
    2. Get the tool/server name for identification and logging
    """

    @abstractmethod
    def config(self) -> McpServerConfig:
        """
        Get the MCP server configuration for this tool.

        Returns:
            McpServerConfig: Configuration dict with type and connection details.

        Example:
            For HTTP-based tools:
            {"type": "http", "url": "http://localhost:8080/mcp"}

            For stdio-based tools:
            {"command": "node", "args": ["server.js"], "env": {}}

            For SSE-based tools:
            {"type": "sse", "url": "http://localhost:8080/sse"}
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Get the name/identifier for this tool/server.

        This name is used for:
        - Server identification in logs
        - Tool ID generation (mcp__servername__toolname format)
        - Configuration management

        Returns:
            str: Unique name for this tool/server

        Example:
            "calculator", "weather", "playwright"
        """
        pass