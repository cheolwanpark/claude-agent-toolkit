#!/usr/bin/env python3
# core.py - Main Agent class with simplified interface

import os
from typing import Any, Dict, List, Optional, Union, Literal

from .docker_manager import DockerManager
from .tool_connector import ToolConnector
from .executor import ContainerExecutor
from ..exceptions import ConfigurationError
from ..constants import ENV_CLAUDE_CODE_OAUTH_TOKEN


class Agent:
    """
    Docker-isolated Agent that runs Claude Code with MCP tool support.
    
    Usage:
        # Traditional pattern
        agent = Agent(oauth_token="...")
        agent.connect(tool1)
        agent.connect(tool2)
        result = await agent.run("Your prompt")
        
        # New pattern (cleaner)
        agent = Agent(
            oauth_token="...",
            system_prompt="You are a helpful assistant",
            tools=[tool1, tool2]
        )
        result = await agent.run("Your prompt")
    """
    
    def __init__(
        self, 
        oauth_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None
    ):
        """
        Initialize the Agent.
        
        Args:
            oauth_token: Claude Code OAuth token (or use ENV_CLAUDE_CODE_OAUTH_TOKEN env var)
            system_prompt: System prompt to customize agent behavior
            tools: List of tool instances to connect automatically
            model: Model to use ("opus", "sonnet", "haiku", or any Claude model name/ID)
        
        Note:
            Docker image version automatically matches the installed package version (__version__) for safety.
        """
        self.oauth_token = oauth_token or os.environ.get(ENV_CLAUDE_CODE_OAUTH_TOKEN, '')
        self.system_prompt = system_prompt
        self.model = model
        
        if not self.oauth_token:
            raise ConfigurationError(f"OAuth token required: pass oauth_token or set {ENV_CLAUDE_CODE_OAUTH_TOKEN}")
        
        # Initialize components
        self.docker_manager = DockerManager()
        self.tool_connector = ToolConnector()
        self.executor = ContainerExecutor(
            self.docker_manager.client, 
            self.docker_manager.IMAGE_NAME
        )
        
        # Ensure Docker image exists
        self.docker_manager.ensure_image()
        
        # Connect tools if provided
        if tools:
            for tool in tools:
                self.connect(tool)
    
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
    
    async def run(
        self, 
        prompt: str, 
        verbose: bool = False,
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None
    ) -> Dict[str, Any]:
        """
        Run the agent with the given prompt.
        
        Args:
            prompt: The instruction for Claude
            verbose: If True, print detailed message processing info
            model: Model to use for this run (overrides agent default)
            
        Returns:
            Dict with success status, response, and metadata
        """
        return self.executor.execute(
            prompt=prompt,
            oauth_token=self.oauth_token,
            tool_urls=self.tool_connector.get_connected_tools(),
            system_prompt=self.system_prompt,
            verbose=verbose,
            model=model or self.model
        )