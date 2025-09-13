#!/usr/bin/env python3
# mcp.py - MCP tool wrapper classes for external MCP servers

from typing import Dict, List, Optional
from .abstract import AbstractTool
from claude_code_sdk.types import (
    McpServerConfig,
    McpStdioServerConfig,
    McpHttpServerConfig,
)


class StdioMCPTool(AbstractTool):
    """
    Wrapper class for external MCP servers using stdio transport.

    This class allows you to integrate external MCP servers that communicate
    via stdin/stdout into the Claude Agent Toolkit framework.

    Example:
        # Connect to an external MCP server
        tool = StdioMCPTool(
            command="node",
            args=["server.js"],
            name="my-server"
        )

        # Use with agent
        agent = Agent(tools=[tool])
        result = await agent.run("Use the external tool")
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize StdioMCPTool with stdio server configuration.

        Args:
            command: Command to execute the MCP server (e.g., "node", "python")
            args: Arguments to pass to the command (e.g., ["server.js"])
            env: Environment variables for the server process
            name: Name identifier for the server (defaults to command name)
        """
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._name = name or command.split('/')[-1]  # Use command basename if no name provided

    def config(self) -> McpServerConfig:
        """Get MCP server configuration for stdio transport."""
        return {
            "command": self._command,
            "args": self._args,
            "env": self._env
        }

    def name(self) -> str:
        """Get the name/identifier for this tool/server."""
        return self._name


class HttpMCPTool(AbstractTool):
    """
    Wrapper class for external MCP servers using HTTP transport.

    This class allows you to integrate external MCP servers that communicate
    via HTTP into the Claude Agent Toolkit framework.

    Example:
        # Connect to an external HTTP MCP server
        tool = HttpMCPTool(
            url="http://localhost:8080/mcp",
            name="my-http-server"
        )

        # Use with agent
        agent = Agent(tools=[tool])
        result = await agent.run("Use the HTTP tool")
    """

    def __init__(self, url: str, name: Optional[str] = None):
        """
        Initialize HttpMCPTool with HTTP server configuration.

        Args:
            url: HTTP URL of the MCP server endpoint
            name: Name identifier for the server (defaults to hostname from URL)
        """
        self._url = url
        self._name = name or self._extract_name_from_url(url)

    def _extract_name_from_url(self, url: str) -> str:
        """Extract a reasonable name from the URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            # Use hostname or fallback to 'http-server'
            return parsed.hostname or 'http-server'
        except Exception:
            return 'http-server'

    def config(self) -> McpServerConfig:
        """Get MCP server configuration for HTTP transport."""
        return {
            "type": "http",
            "url": self._url
        }

    def name(self) -> str:
        """Get the name/identifier for this tool/server."""
        return self._name


