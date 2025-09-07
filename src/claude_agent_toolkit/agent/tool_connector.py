#!/usr/bin/env python3
# tool_connector.py - Tool connection and URL management

from typing import Any, Dict

from ..logging import get_logger
from ..exceptions import ConfigurationError
from ..constants import DOCKER_LOCALHOST_MAPPINGS

logger = get_logger('agent')


class ToolConnector:
    """Manages tool connections and URL mappings for Docker container access."""
    
    def __init__(self, is_docker: bool = True):
        """Initialize tool connector.
        
        Args:
            is_docker: Whether to convert localhost URLs for Docker container access.
                      Set to False when using subprocess executor.
        """
        self.is_docker = is_docker
        self.tool_urls: Dict[str, str] = {}  # tool_name -> url mapping
        self.tools: Dict[str, Any] = {}  # tool_name -> tool instance mapping
    
    def connect_tool(self, tool: Any) -> str:
        """
        Connect to an MCP tool server.
        
        Args:
            tool: Tool instance with connection_url property
            
        Returns:
            Tool name that was connected
            
        Raises:
            ConfigurationError: If tool doesn't have connection_url property
        """
        if not hasattr(tool, 'connection_url'):
            raise ConfigurationError("Tool must have 'connection_url' property")
        
        # Get tool name (class name in lowercase for consistency)
        tool_name = tool.__class__.__name__.lower()
        
        # Rewrite localhost URLs for Docker container access (only if using Docker)
        url = tool.connection_url
        if self.is_docker:
            for localhost, docker_host in DOCKER_LOCALHOST_MAPPINGS.items():
                url = url.replace(localhost, docker_host)
        
        self.tool_urls[tool_name] = url
        self.tools[tool_name] = tool  # Store tool instance for discovery
        logger.info("Connected to %s (class: %s) at %s", tool_name, tool.__class__.__name__, url)
        
        return tool_name
    
    def get_connected_tools(self) -> Dict[str, str]:
        """Get all connected tool URLs."""
        return self.tool_urls.copy()

    
    def get_connected_tool_instances(self) -> Dict[str, Any]:
        """Get all connected tool instances."""
        return self.tools.copy()
    
    def clear_connections(self):
        """Clear all tool connections."""
        self.tool_urls.clear()
        self.tools.clear()