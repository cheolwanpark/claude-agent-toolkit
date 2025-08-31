#!/usr/bin/env python3
# core.py - Main Agent class with simplified interface

import os
from typing import Any, Dict, Optional

from .docker_manager import DockerManager
from .tool_connector import ToolConnector
from .executor import ContainerExecutor


class Agent:
    """
    Docker-isolated Agent that runs Claude Code with MCP tool support.
    
    Usage:
        agent = Agent(oauth_token="...")
        agent.connect(tool1)
        agent.connect(tool2)
        result = await agent.run("Your prompt")
    """
    
    def __init__(self, oauth_token: Optional[str] = None):
        """
        Initialize the Agent.
        
        Args:
            oauth_token: Claude Code OAuth token (or use CLAUDE_CODE_OAUTH_TOKEN env var)
        """
        self.oauth_token = oauth_token or os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '')
        
        if not self.oauth_token:
            raise ValueError("OAuth token required: pass oauth_token or set CLAUDE_CODE_OAUTH_TOKEN")
        
        # Initialize components
        self.docker_manager = DockerManager()
        self.tool_connector = ToolConnector()
        self.executor = ContainerExecutor(
            self.docker_manager.client, 
            self.docker_manager.IMAGE_NAME
        )
        
        # Ensure Docker image exists
        self.docker_manager.ensure_image()
    
    def connect(self, tool: Any) -> 'Agent':
        """
        Connect to an MCP tool server. Can be called multiple times for multiple tools.
        
        Args:
            tool: Tool instance with connection_url property
            
        Returns:
            Self for chaining
        """
        self.tool_connector.connect_tool(tool)
        return self
    
    async def run(self, prompt: str) -> Dict[str, Any]:
        """
        Run the agent with the given prompt.
        
        Args:
            prompt: The instruction for Claude
            
        Returns:
            Dict with success status and response
        """
        return self.executor.execute(
            prompt=prompt,
            oauth_token=self.oauth_token,
            tool_urls=self.tool_connector.get_connected_tools()
        )