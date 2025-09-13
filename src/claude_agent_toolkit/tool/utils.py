#!/usr/bin/env python3
# utils.py - Utility functions for MCP tool discovery

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client, StdioServerParameters
from claude_code_sdk.types import (
    McpServerConfig,
    McpStdioServerConfig,
    McpSSEServerConfig,
    McpHttpServerConfig,
    McpSdkServerConfig
)

# Removed direct import to avoid circular import - imported dynamically where needed
from ..exceptions import ConnectionError, ExecutionError
from ..logging import get_logger

logger = get_logger("tool")


@dataclass
class ToolInfo:
    """Information about an MCP tool discovered from a server."""

    server_name: str  # MCP server name (for mcp__[servername]__[toolname])
    tool_name: str  # Tool name (for mcp__[servername]__[toolname])
    description: str  # Tool description
    input_schema: Dict[str, Any]  # JSON schema for tool inputs

    @property
    def mcp_tool_id(self) -> str:
        """Generate Claude Code compatible tool identifier."""
        return f"mcp__{self.server_name}__{self.tool_name}"


def _convert_to_tool_infos(tools_response, server_name: str) -> List[ToolInfo]:
    """Convert MCP tools response to ToolInfo objects."""
    tool_infos = []
    for mcp_tool in tools_response.tools:
        tool_info = ToolInfo(
            server_name=server_name,
            tool_name=mcp_tool.name,
            description=mcp_tool.description or "",
            input_schema=mcp_tool.inputSchema or {}
        )
        tool_infos.append(tool_info)
        logger.debug(f"Added tool: {tool_info.mcp_tool_id}")
    return tool_infos


# Removed tool_to_config function - replaced by tool.config() method


async def list_tools(tool) -> List[ToolInfo]:
    """
    List available tools from an MCP tool/server.

    Supports stdio, HTTP, and SSE transports via tool.config() method.

    Args:
        tool: AbstractTool instance with config() and name() methods

    Returns:
        List of ToolInfo objects with server_name and tool_name

    Raises:
        ConnectionError: If the tool configuration is invalid
        ExecutionError: If unable to retrieve tools from the server
        NotImplementedError: If config type is not supported

    Example:
        # Using a BaseTool (HTTP-based)
        my_tool = MyBaseTool()
        tools = await list_tools(my_tool)

        # Using external MCP server wrapper
        external_tool = ExternalMcpTool()
        tools = await list_tools(external_tool)
    """
    # Import here to avoid circular imports
    from .abstract import AbstractTool

    if not isinstance(tool, AbstractTool):
        raise ConnectionError("Tool must be an instance of AbstractTool")

    config = tool.config()
    server_name = tool.name()

    logger.debug(f"Connecting to MCP server for {server_name}")

    # Use isolated async context to avoid cross-task violations
    async def _isolated_tool_discovery():
        """Isolated async context for MCP session to avoid TaskGroup cross-task issues."""
        try:
            # Determine transport type and create appropriate client
            if "command" in config:  # McpStdioServerConfig
                params = StdioServerParameters(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env", {})
                )
                logger.debug(f"Using stdio transport: {config['command']} {config.get('args', [])}")
                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        logger.debug(f"Initialized MCP stdio session for {server_name}")
                        tools_response = await session.list_tools()
                        logger.info(f"Retrieved {len(tools_response.tools)} tools from {server_name}")
                        return _convert_to_tool_infos(tools_response, server_name)

            elif config.get("type") == "sse":  # McpSSEServerConfig
                logger.debug(f"Using SSE transport: {config['url']}")
                async with streamablehttp_client(config["url"]) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        logger.debug(f"Initialized MCP SSE session for {server_name}")
                        tools_response = await session.list_tools()
                        logger.info(f"Retrieved {len(tools_response.tools)} tools from {server_name}")
                        return _convert_to_tool_infos(tools_response, server_name)

            elif "url" in config:  # McpHttpServerConfig or fallback
                logger.debug(f"Using HTTP transport: {config['url']}")
                async with streamablehttp_client(config["url"]) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        logger.debug(f"Initialized MCP HTTP session for {server_name}")
                        tools_response = await session.list_tools()
                        logger.info(f"Retrieved {len(tools_response.tools)} tools from {server_name}")
                        return _convert_to_tool_infos(tools_response, server_name)

            else:  # McpSdkServerConfig or unknown
                raise NotImplementedError(f"Unsupported server config type: {config}")

        except Exception as e:
            logger.error(f"Tool discovery failed for {server_name}: {e}")
            raise ExecutionError(f"Failed to discover tools from {server_name}: {e}") from e

    # Execute the isolated tool discovery
    return await _isolated_tool_discovery()


