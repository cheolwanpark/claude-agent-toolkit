#!/usr/bin/env python3
# tool_connector.py - Tool connection and URL management

from typing import Any, Dict

from ..logging import get_logger
from ..exceptions import ConfigurationError
from ..constants import DOCKER_LOCALHOST_MAPPINGS
from ..tool.utils import McpServerConfig

logger = get_logger('agent')


class ToolConnector:
    """Manages tool connections and MCP server configurations for executor access."""

    def __init__(self, is_docker: bool = True):
        """Initialize tool connector.

        Args:
            is_docker: Whether to convert localhost URLs for Docker container access.
                      Set to False when using subprocess executor.
        """
        self.is_docker = is_docker
        self.mcp_servers: Dict[str, McpServerConfig] = {}  # tool_name -> config mapping
        self.tools: Dict[str, Any] = {}  # tool_name -> tool instance mapping
    
    def connect_tool(self, tool: Any) -> str:
        """
        Connect to an MCP tool server.

        Args:
            tool: AbstractTool instance with config() and name() methods

        Returns:
            Tool name that was connected

        Raises:
            ConfigurationError: If tool doesn't have required methods
        """
        # Import here to avoid circular imports
        from ..tool.abstract import AbstractTool

        if not isinstance(tool, AbstractTool):
            raise ConfigurationError("Tool must be an instance of AbstractTool with config() and name() methods")

        # Get tool configuration and name
        config = tool.config()
        tool_name = tool.name()

        # Apply Docker host mapping for HTTP-based tools if needed
        if self.is_docker and config.get("type") == "http":
            url = config.get("url", "")
            for localhost, docker_host in DOCKER_LOCALHOST_MAPPINGS.items():
                url = url.replace(localhost, docker_host)
            # Create modified config for Docker environment
            config = config.copy()
            config["url"] = url

        self.mcp_servers[tool_name] = config
        self.tools[tool_name] = tool  # Store tool instance for discovery
        logger.info("Connected to %s (class: %s) with config: %s", tool_name, tool.__class__.__name__, config)

        return tool_name
    
    def get_connected_tools(self) -> Dict[str, McpServerConfig]:
        """Get all connected MCP server configurations."""
        return self.mcp_servers.copy()

    
    def get_connected_tool_instances(self) -> Dict[str, Any]:
        """Get all connected tool instances."""
        return self.tools.copy()
    
    def clear_connections(self):
        """Clear all tool connections."""
        self.mcp_servers.clear()
        self.tools.clear()