#!/usr/bin/env python3
# tool_connector.py - Tool connection and URL management

from typing import Any, Dict

from ..logging import get_logger
from ..exceptions import ConfigurationError
from ..constants import DOCKER_LOCALHOST_MAPPINGS

logger = get_logger('agent')


class ToolConnector:
    """Manages tool connections and URL mappings for Docker container access."""
    
    def __init__(self):
        """Initialize tool connector."""
        self.tool_urls: Dict[str, str] = {}  # tool_name -> url mapping
    
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
        
        # Get tool name (class name)
        tool_name = tool.__class__.__name__
        
        # Rewrite localhost URLs for Docker container access
        url = tool.connection_url
        for localhost, docker_host in DOCKER_LOCALHOST_MAPPINGS.items():
            url = url.replace(localhost, docker_host)
        
        self.tool_urls[tool_name] = url
        logger.info("Connected to %s at %s", tool_name, url)
        
        return tool_name
    
    def get_connected_tools(self) -> Dict[str, str]:
        """Get all connected tool URLs."""
        return self.tool_urls.copy()
    
    def clear_connections(self):
        """Clear all tool connections."""
        self.tool_urls.clear()