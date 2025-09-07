#!/usr/bin/env python3
# subprocess.py - Subprocess executor without Docker dependency

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
        
        # Directly await the async execution
        return await self._run_claude_code_sdk(
            prompt=prompt,
            oauth_token=oauth_token,
            tool_urls=tool_urls,
            allowed_tools=allowed_tools,
            system_prompt=system_prompt,
            verbose=verbose,
            model=model
        )
    
    def _serialize_message(self, message):
        """Convert a claude-code-sdk message to a serializable dict."""
        def default_serializer(obj):
            """Custom serializer for objects that aren't JSON serializable by default."""
            if hasattr(obj, '__dict__'):
                result = {"type": type(obj).__name__}
                result.update(obj.__dict__)
                return result
            return str(obj)
        
        return json.loads(json.dumps(message, default=default_serializer))
    
    async def _isolated_claude_query(
        self, 
        prompt: str, 
        options,
        handler,
        verbose: bool = False
    ) -> Optional[str]:
        """
        Isolated async context for claude-code-sdk query.
        
        This method ensures the entire claude-code-sdk async generator 
        lifecycle (including TaskGroup creation/cleanup) happens within 
        a single, contained async context to avoid cross-task violations.
        
        Args:
            prompt: The instruction for Claude
            options: ClaudeCodeOptions instance
            handler: ResponseHandler instance
            verbose: Enable verbose logging
            
        Returns:
            Response string from Claude if available
            
        Raises:
            ExecutionError: If query execution fails
        """
        
        query_result = None
        query_generator = None
        
        try:
            # Initialize generator in this async context
            query_generator = query(prompt=prompt, options=options)
            
            # Process all messages in same async context
            # Note: Don't break early - let generator complete naturally
            async for message in query_generator:
                try:
                    # Process message through handler
                    message_dict = self._serialize_message(message)
                    json_line = json.dumps(message_dict)
                    
                    result = handler.handle(json_line, verbose)
                    if result and not query_result:
                        # Store first result but continue processing
                        query_result = result
                        
                except Exception as e:
                    if verbose:
                        logger.debug("Failed to process message: %s", e)
                    continue
            
            # Generator completed naturally - TaskGroup cleanup happens here
            
        except GeneratorExit:
            # AnyIO best practice: Allow natural cleanup, don't interfere
            raise
            
        except Exception as e:
            logger.error("Claude Code query failed: %s", e)
            raise ExecutionError(f"Claude Code query failed: {e}") from e
            
        finally:
            # Ensure generator cleanup happens in same async context
            if query_generator:
                try:
                    await query_generator.aclose()
                except (AttributeError, RuntimeError):
                    # Generator may already be closed or not have aclose method
                    pass
        
        return query_result
    
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
                
                
                # Use isolated async context for query execution
                result = await self._isolated_claude_query(
                    prompt=prompt,
                    options=options,
                    handler=handler,
                    verbose=verbose
                )
                
                if result:
                    logger.info("Execution completed successfully")
                    return result
                
                # If we get here, no ResultMessage was received
                if handler.text_responses:
                    logger.info("Execution completed with text responses")
                    return '\n'.join(handler.text_responses)
                else:
                    raise ExecutionError("No response received from Claude")
                    
        finally:
            # Restore original environment - use try/except to protect critical cleanup
            try:
                os.environ.clear()
                os.environ.update(original_env)
            except Exception as e:
                logger.warning("Failed to restore environment: %s", e)
                # Don't raise - environment restoration shouldn't break execution
