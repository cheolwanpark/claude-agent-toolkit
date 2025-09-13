#!/usr/bin/env python3
# base.py - Abstract base executor class

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Import types to use in type hints
try:
    from ...tool.utils import McpServerConfig
except ImportError:
    # Fallback type definition to handle import order issues
    McpServerConfig = Dict[str, Any]


class BaseExecutor(ABC):
    """Abstract base class for all executor implementations."""
    
    @abstractmethod
    async def run(
        self,
        prompt: str,
        oauth_token: str,
        mcp_servers: Dict[str, McpServerConfig],
        allowed_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None
    ) -> str:
        """
        Execute prompt with connected MCP servers.

        Args:
            prompt: The instruction for Claude
            oauth_token: Claude Code OAuth token
            mcp_servers: Dictionary of server_name -> McpServerConfig mappings
            allowed_tools: List of allowed tool IDs (mcp__servername__toolname format)
            system_prompt: Optional system prompt to customize agent behavior
            verbose: If True, enable verbose output
            model: Optional model to use for this execution

        Returns:
            Response string from Claude

        Raises:
            ConfigurationError: If OAuth token or configuration is invalid
            ConnectionError: If connection fails
            ExecutionError: If execution fails
        """
        pass