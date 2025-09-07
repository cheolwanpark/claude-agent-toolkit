#!/usr/bin/env python3
# core.py - Main Agent class with simplified interface

import os
from typing import Any, Dict, List, Optional, Union, Literal

from .tool_connector import ToolConnector
from .executor import ExecutorType, create_executor
from ..exceptions import ConfigurationError
from ..constants import ENV_CLAUDE_CODE_OAUTH_TOKEN
from ..tool.utils import list_tools
from ..logging import get_logger

logger = get_logger('agent')


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
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None,
        executor: Optional[ExecutorType] = None
    ):
        """
        Initialize the Agent.
        
        Args:
            oauth_token: Claude Code OAuth token (or use ENV_CLAUDE_CODE_OAUTH_TOKEN env var)
            system_prompt: System prompt to customize agent behavior
            tools: List of tool instances to connect automatically
            model: Model to use ("opus", "sonnet", "haiku", or any Claude model name/ID)
            executor: Executor type to use (defaults to DOCKER)
        
        Note:
            Docker image version automatically matches the installed package version (__version__) for safety.
        """
        self.oauth_token = oauth_token or os.environ.get(ENV_CLAUDE_CODE_OAUTH_TOKEN, '')
        self.system_prompt = system_prompt
        self.model = model
        
        if not self.oauth_token:
            raise ConfigurationError(f"OAuth token required: pass oauth_token or set {ENV_CLAUDE_CODE_OAUTH_TOKEN}")
        
        # Create executor instance
        self.executor_type = executor or ExecutorType.DOCKER
        self.executor = create_executor(self.executor_type)
        
        # Initialize components with Docker-aware configuration
        is_docker = self.executor_type == ExecutorType.DOCKER
        self.tool_connector = ToolConnector(is_docker=is_docker)
        
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

    
    async def _discover_tools(self) -> List[str]:
        """
        Discover all available tools from connected MCP servers.
        
        Returns:
            List of tool IDs in the format mcp__servername__toolname
            
        Note:
            Uses graceful error handling - continues discovery even if some servers fail
        """
        all_tools = []
        tool_instances = self.tool_connector.get_connected_tool_instances()
        successful_discoveries = 0
        
        for tool_name, tool in tool_instances.items():
            try:
                logger.debug("Discovering tools from %s", tool_name)
                tool_infos = await list_tools(tool)
                
                for info in tool_infos:
                    all_tools.append(info.mcp_tool_id)
                    logger.debug("Discovered tool: %s", info.mcp_tool_id)
                
                logger.info("Discovered %d tools from %s", len(tool_infos), tool_name)
                successful_discoveries += 1
                
            except Exception as e:
                logger.warning("Failed to discover tools from %s: %s", tool_name, e)
                logger.debug("Tool discovery error details for %s", tool_name, exc_info=True)
                # Continue with other tools instead of failing completely
                continue
        
        logger.info("Total discovered tools: %d from %d/%d servers", 
                   len(all_tools), successful_discoveries, len(tool_instances))
        
        if successful_discoveries == 0 and len(tool_instances) > 0:
            logger.warning("No tools discovered from any connected servers - agent will have limited functionality")
        
        return all_tools
    
    async def run(
        self, 
        prompt: str, 
        verbose: bool = False,
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None
    ) -> str:
        """
        Run the agent with the given prompt.
        
        Args:
            prompt: The instruction for Claude
            verbose: If True, print detailed message processing info
            model: Model to use for this run (overrides agent default)
            
        Returns:
            Response string from Claude
            
        Raises:
            ConfigurationError: If OAuth token or configuration is invalid
            ConnectionError: If Docker connection fails
            ExecutionError: If agent execution fails
        """
        # Discover available tools from connected MCP servers
        allowed_tools = await self._discover_tools()
        
        # Use stored executor instance
        return await self.executor.run(
            prompt=prompt,
            oauth_token=self.oauth_token,
            tool_urls=self.tool_connector.get_connected_tools(),
            allowed_tools=allowed_tools,
            system_prompt=self.system_prompt,
            verbose=verbose,
            model=model or self.model
        )