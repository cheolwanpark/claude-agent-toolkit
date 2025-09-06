#!/usr/bin/env python3
# response_handler.py - Handles streaming responses from Docker container

import json
import sys
from typing import Any, Dict, List, Optional

from ..logging import get_logger

logger = get_logger('response_handler')


class ResponseHandler:
    """Handles streaming responses from the Docker container."""
    
    def __init__(self):
        """Initialize the response handler."""
        self.messages: List[Dict[str, Any]] = []
        self.text_responses: List[str] = []
        self.final_result: Optional[str] = None
        
    def handle(self, line: str, verbose: bool = False) -> Optional[str]:
        """
        Process a line of output from container.
        
        Args:
            line: A line of output from the container
            verbose: If True, print detailed processing info
            
        Returns:
            Final response string if ResultMessage received, None otherwise
        """
        if not line.strip():
            return None
            
        # Skip non-JSON lines
        if not line.startswith('{'):
            if verbose and line.strip():
                print(f"[Container] {line}", flush=True)
            return None
            
        try:
            msg = json.loads(line)
            self.messages.append(msg)
            
            # Handle different message types
            msg_type = msg.get('type', 'Unknown')
            
            if msg_type == 'AssistantMessage':
                self._handle_assistant_message(msg, verbose)
                
            elif msg_type == 'ResultMessage':
                return self._handle_result_message(msg, verbose)
                
            elif msg_type == 'UserMessage':
                if verbose:
                    print(f"[User] Message received", flush=True)
                    
            elif msg_type == 'SystemMessage':
                if verbose:
                    subtype = msg.get('subtype', 'unknown')
                    print(f"[System] {subtype}", flush=True)
                    
            else:
                if verbose:
                    print(f"[{msg_type}] Received", flush=True)
                    
        except json.JSONDecodeError as e:
            if verbose:
                print(f"[JSON Error] {e}: {line[:100]}...", flush=True)
            logger.warning("Failed to parse JSON: %s", e)
                
        return None
    
    def _handle_assistant_message(self, msg: Dict[str, Any], verbose: bool) -> None:
        """Handle AssistantMessage type."""
        content = msg.get('content', [])
        model = msg.get('model', 'unknown')
        
        if verbose:
            print(f"[Assistant] Model: {model}", flush=True)
        
        for block in content:
            if isinstance(block, dict):
                block_type = block.get('type', 'Unknown')
                
                if block_type == 'TextBlock':
                    text = block.get('text', '')
                    if text:
                        self.text_responses.append(text)
                        if verbose:
                            # Print with indentation for readability
                            for line in text.split('\n'):
                                print(f"  Text: {line}", flush=True)
                                
                elif block_type == 'ThinkingBlock':
                    if verbose:
                        thinking = block.get('thinking', '')[:200]
                        print(f"  Thinking: {thinking}...", flush=True)
                        
                elif block_type == 'ToolUseBlock':
                    if verbose:
                        name = block.get('name', 'unknown')
                        tool_id = block.get('id', 'unknown')
                        input_keys = list(block.get('input', {}).keys())
                        print(f"  ToolUse: {name}({tool_id}) with {input_keys}", flush=True)
                        
                elif block_type == 'ToolResultBlock':
                    if verbose:
                        tool_use_id = block.get('tool_use_id', 'unknown')
                        is_error = block.get('is_error', False)
                        status = "ERROR" if is_error else "OK"
                        print(f"  ToolResult[{status}]: {tool_use_id}", flush=True)
    
    def _handle_result_message(self, msg: Dict[str, Any], verbose: bool) -> str:
        """Handle ResultMessage type and return final result."""
        # Extract metadata
        duration_ms = msg.get('duration_ms', 0)
        total_cost_usd = msg.get('total_cost_usd', 0)
        is_error = msg.get('is_error', False)
        num_turns = msg.get('num_turns', 0)
        
        if verbose:
            print(f"[Result] Duration: {duration_ms}ms, Cost: ${total_cost_usd:.4f}, Turns: {num_turns}", flush=True)
            if is_error:
                print(f"[Result] Error flag set", flush=True)
        
        # Get final result from ResultMessage
        result = msg.get('result')
        if result:
            self.final_result = result
        else:
            # Fallback to accumulated text responses
            self.final_result = '\n'.join(self.text_responses) if self.text_responses else "No response generated"
            
        return self.final_result
    
    def get_response_summary(self) -> Dict[str, Any]:
        """Get a summary of processed messages."""
        return {
            "total_messages": len(self.messages),
            "text_responses": len(self.text_responses),
            "final_result": self.final_result,
            "message_types": [msg.get('type', 'Unknown') for msg in self.messages]
        }