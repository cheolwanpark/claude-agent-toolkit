#!/usr/bin/env python3
# base.py - Abstract base executor class

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseExecutor(ABC):
    """Abstract base class for all executor implementations."""
    
    @abstractmethod
    async def run(
        self, 
        prompt: str, 
        oauth_token: str, 
        tool_urls: Dict[str, str], 
        allowed_tools: Optional[List[str]] = None, 
        system_prompt: Optional[str] = None, 
        verbose: bool = False, 
        model: Optional[str] = None
    ) -> str:
        """
        Execute prompt with connected tools.
        
        Args:
            prompt: The instruction for Claude
            oauth_token: Claude Code OAuth token
            tool_urls: Dictionary of tool_name -> url mappings
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