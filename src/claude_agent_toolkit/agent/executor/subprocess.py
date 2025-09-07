#!/usr/bin/env python3
# subprocess.py - Subprocess executor without Docker dependency

import asyncio
import dataclasses
import json
import os
import tempfile
from typing import Dict, List, Optional

from claude_code_sdk import query, ClaudeCodeOptions

from .base import BaseExecutor
from ...constants import MODEL_ID_MAPPING
from ...exceptions import ConfigurationError, ExecutionError
from ...logging import get_logger
from ..response_handler import ResponseHandler

logger = get_logger('agent')


class SubprocessExecutor(BaseExecutor):
    """Subprocess-based executor that runs Claude Code SDK directly without Docker dependency."""
    
    def __init__(self):
        """
        Initialize subprocess executor.
        
        Note:
            No Docker dependency required - uses claude-code-sdk directly.
            Creates temporary directory for minimal file system isolation.
        """
        logger.debug("Initialized SubprocessExecutor")
    
    def run(
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
        Execute prompt using claude-code-sdk directly with connected tools.
        
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
            ExecutionError: If execution fails
        """
        logger.info("Running with prompt: %s...", prompt[:100])
        
        if not oauth_token:
            raise ConfigurationError("OAuth token is required")
        
        # Use asyncio to run the async execution
        return asyncio.run(self._run_claude_code_sdk(
            prompt=prompt,
            oauth_token=oauth_token,
            tool_urls=tool_urls,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
            verbose=verbose,
            model=model
        ))
    
    async def _run_claude_code_sdk(
        self,
        prompt: str,
        oauth_token: str,
        tool_urls: Dict[str, str],
        allowed_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None
    ) -> str:
        """Run claude-code-sdk directly with temporary directory isolation."""
        
        # Set up environment variables for claude-code-sdk
        original_env = os.environ.copy()
        try:
            # Set OAuth token in environment for SDK
            os.environ['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token
            
            # Apply model ID mapping if needed
            final_model = None
            if model:
                final_model = MODEL_ID_MAPPING.get(model, model)
            
            # Configure MCP servers using HTTP configuration
            mcp_servers = {}
            if tool_urls:
                for tool_name, tool_url in tool_urls.items():
                    # Use localhost URLs directly (no Docker host mapping needed)
                    mcp_servers[tool_name.lower()] = {
                        "type": "http",
                        "url": tool_url,
                        "headers": {}
                    }
                    if verbose:
                        logger.info("Configured HTTP MCP server %s at %s", tool_name, tool_url)
            
            if verbose:
                logger.info("Connected tools: %s", list(tool_urls.keys()) if tool_urls else [])
                if allowed_tools:
                    logger.info("Allowed tools: %d tools discovered", len(allowed_tools))
                logger.info("Using model: %s", final_model)
            
            # Create temporary directory for minimal isolation
            with tempfile.TemporaryDirectory(prefix="claude-agent-") as temp_dir:
                logger.debug("Created temporary directory: %s", temp_dir)
                
                # Setup Claude Code options with temporary directory as working directory
                options = ClaudeCodeOptions(
                    allowed_tools=allowed_tools if allowed_tools else None,
                    mcp_servers=mcp_servers if mcp_servers else {},
                    system_prompt=system_prompt,
                    model=final_model,
                    cwd=temp_dir
                )
                
                # Create response handler for processing messages
                handler = ResponseHandler()
                
                try:
                    logger.debug("Starting Claude Code query with %d MCP servers", len(mcp_servers))
                    
                    async for message in query(prompt=prompt, options=options):
                        # Convert message to JSON format for ResponseHandler
                        try:
                            # Convert dataclass message to dict for JSON serialization
                            try:
                                message_dict = dataclasses.asdict(message)
                            except (TypeError, AttributeError):
                                # Fallback for non-dataclass objects
                                message_dict = {'type': type(message).__name__, 'content': str(message)}
                            
                            # Serialize to JSON string for ResponseHandler
                            json_line = json.dumps(message_dict)
                            
                            # Process through ResponseHandler
                            result = handler.handle(json_line, verbose)
                            if result:
                                logger.info("Execution completed successfully")
                                return result
                                
                        except (TypeError, AttributeError) as e:
                            if verbose:
                                logger.debug("Failed to process message: %s", e)
                            continue
                                
                except Exception as e:
                    logger.error("Claude Code SDK execution failed: %s", e)
                    raise ExecutionError(f"Claude Code SDK execution failed: {e}") from e
                
                # If we get here, no ResultMessage was received
                if handler.text_responses:
                    logger.info("Execution completed with text responses")
                    return '\n'.join(handler.text_responses)
                else:
                    raise ExecutionError("No response received from Claude")
                    
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
