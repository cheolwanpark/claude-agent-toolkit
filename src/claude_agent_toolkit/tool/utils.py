#!/usr/bin/env python3
# utils.py - Utility functions for MCP tool discovery

from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .base import BaseTool
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


async def list_tools(tool: BaseTool) -> List[ToolInfo]:
    """
    List available tools from a BaseTool's MCP server.
    
    This function connects to the MCP server running for the given tool
    and retrieves all available tools from that server.
    
    Args:
        tool: A BaseTool instance with a running MCP server
        
    Returns:
        List of ToolInfo objects with server_name and tool_name
        that can be used to construct mcp__servername__toolname
        
    Raises:
        ConnectionError: If the tool's server is not running
        ExecutionError: If unable to retrieve tools from the server
    
    Example:
        my_tool = MyCustomTool()
        tools = await list_tools(my_tool)
        for t in tools:
            print(f"Tool ID: {t.mcp_tool_id}")  # mcp__MyCustomTool__method_name
    """
    if not hasattr(tool, 'connection_url'):
        raise ConnectionError("Tool must have 'connection_url' property")
    
    # Extract server name from tool class (lowercase to match Claude Code format)
    server_name = tool.__class__.__name__.lower()
    
    # Get the tool's MCP server URL
    server_url = tool.connection_url
    
    # Validate URL
    try:
        parsed_url = urlparse(server_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ConnectionError(f"Invalid server URL: {server_url}")
    except Exception as e:
        raise ConnectionError(f"Failed to parse server URL {server_url}: {e}") from e
    
    logger.debug(f"Connecting to MCP server at {server_url} for {server_name}")
    
    # Use isolated async context to avoid cross-task violations
    async def _isolated_tool_discovery():
        """Isolated async context for MCP session to avoid TaskGroup cross-task issues."""
        try:
            # Connect using streamable HTTP client in isolated context
            async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the connection
                    await session.initialize()
                    logger.debug(f"Initialized MCP session for {server_name}")
                    
                    # List available tools
                    tools_response = await session.list_tools()
                    logger.info(f"Retrieved {len(tools_response.tools)} tools from {server_name}")
                    
                    # Convert to ToolInfo objects
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
                    
        except Exception as e:
            logger.error(f"Tool discovery failed for {server_name}: {e}")
            raise ExecutionError(f"Failed to discover tools from {server_name}: {e}") from e
    
    # Execute the isolated tool discovery
    return await _isolated_tool_discovery()